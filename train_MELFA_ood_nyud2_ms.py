import argparse

from tqdm import tqdm
import numpy as np

import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter
from mindspore.experimental import optim
from mindspore.train.serialization import save_checkpoint as save, load_checkpoint as load
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common import dtype as mstype
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback

from sklearn.metrics import accuracy_score
from models.MELFA_ms import MELFA
from mindspore.train.model import Model
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from data.ood_conc_dataset_ms import oodConcDataset
from data.aligned_conc_dataset_ms import AlignedConcDataset
from utils.utils_ms import *
from utils.crl_utils_ms import *
from utils.logger import create_logger
import os
import mindspore.dataset as ds
import mindspore.dataset.engine.datasets as de


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=32)
    # FIXME: directory
    # parser.add_argument("--data_path", type=str, default="/media/zhangqingyang/dataset/mm/nyud2_trainvaltest/")
    parser.add_argument("--data_path", type=str, default=r"D:\downloads\nyud2_trainvaltest")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.1)
    # parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="s")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/MELFA/ood_nyud/pretrained_resnet18/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--df", type=int, default=1)
    parser.add_argument("--sample", type=float, default=0.3)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--m_in", type=float, default=10)
    parser.add_argument("--m_out", type=float, default=10)
    parser.add_argument("--CONTENT_MODEL_PATH", type=str,
                        # FIXME: directory
                        # default="/media/zhangqingyang/DF/checkpoint/resnet18_pretrained.pth")
                        default=r"D:\downloads\resnet18-f37072fd.pth")


def get_optimizer(model, args):
    optimizer = optim.Adam(model.trainable_params(), lr=args.lr, weight_decay=1e-5)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, factor=args.lr_factor
    )


def rank_loss(confidence, idx, history):
    # make input pair
    rank_input1 = confidence
    rank_input2 = P.roll(confidence, -1)
    idx2 = P.roll(idx, -1)

    # calc target, margin
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

    # ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                                    rank_input2,
                                                    - rank_target)
    return ranking_loss


def model_forward_train(_, model, args, batch, depth_history, rgb_history):
    rgb, depth, tgt = batch['A'], batch['B'], batch['label']
    idx = batch['idx']
    ID_1, ID_2 = batch['ID_1'], batch['ID_2']
    ID = ID_1 * ID_2

    # depth_alpha, rgb_alpha, depth_rgb_alpha = model(rgb, depth)
    depth_rgb_logits, rgb_logits, depth_logits, rgb_conf, depth_conf = model(rgb, depth)
    if args.df == 0:
        depth_rgb_logits = (rgb_logits + depth_logits) / 2

    depth_clf_loss = nn.CrossEntropyLoss()(depth_logits, tgt)
    rgb_clf_loss = nn.CrossEntropyLoss()(rgb_logits, tgt)
    depth_rgb_clf_loss = nn.CrossEntropyLoss()(depth_rgb_logits, tgt)
    clf_loss = depth_clf_loss * ID_2 + rgb_clf_loss * ID_1 + depth_rgb_clf_loss * ID

    depth_loss = nn.CrossEntropyLoss(reduction='none')(depth_logits, tgt)
    rgb_loss = nn.CrossEntropyLoss(reduction='none')(rgb_logits, tgt)

    depth_rank_loss = rank_loss(depth_conf, idx, depth_history)
    rgb_rank_loss = rank_loss(rgb_conf, idx, rgb_history)

    depth_history.correctness_update(idx, depth_loss, depth_conf.squeeze())
    rgb_history.correctness_update(idx, rgb_loss, rgb_conf.squeeze())

    crl_loss = depth_rank_loss + rgb_rank_loss
    loss = P.mean(clf_loss + args.lamb * crl_loss)

    return loss, depth_rgb_logits, rgb_logits, depth_logits, tgt


def model_forward_eval(_, model, args, batch):
    model.set_train(False)
    rgb, depth, tgt = batch['A'], batch['B'], batch['label']

    # depth_alpha, rgb_alpha, depth_rgb_alpha = model(rgb, depth)
    depth_rgb_logits, rgb_logits, depth_logits, rgb_conf, depth_conf = model(rgb, depth)
    if args.df == 0:
        depth_rgb_logits = (rgb_logits + depth_logits) / 2

    depth_clf_loss = nn.CrossEntropyLoss()(depth_logits, tgt)
    rgb_clf_loss = nn.CrossEntropyLoss()(rgb_logits, tgt)
    depth_rgb_clf_loss = nn.CrossEntropyLoss()(depth_rgb_logits, tgt)
    # clf_loss = depth_clf_loss + rgb_clf_loss + depth_rgb_clf_loss
    clf_loss = depth_clf_loss + rgb_clf_loss + depth_rgb_clf_loss

    loss = P.mean(clf_loss)

    model.set_train(True)
    return loss, depth_rgb_logits, rgb_logits, depth_logits, tgt


class MELFAWithLoss(nn.Cell):

    def __init__(self, model, args):
        super(MELFAWithLoss, self).__init__()
        self.model = model
        self.args = args

    def construct(self, rgb, depth, tgt):
        depth_rgb_logits, rgb_logits, depth_logits, rgb_conf, depth_conf = self.model(rgb, depth)
        if self.args.df == 0:
            depth_rgb_logits = (rgb_logits + depth_logits) / 2

        depth_clf_loss = nn.CrossEntropyLoss()(depth_logits, tgt)
        rgb_clf_loss = nn.CrossEntropyLoss()(rgb_logits, tgt)
        depth_rgb_clf_loss = nn.CrossEntropyLoss()(depth_rgb_logits, tgt)
        # clf_loss = depth_clf_loss + rgb_clf_loss + depth_rgb_clf_loss
        clf_loss = depth_clf_loss + rgb_clf_loss + depth_rgb_clf_loss

        loss = P.mean(clf_loss)
        return loss


class MELFAOneStepCell(nn.Cell):

    def __init__(self, network, optimizer, sens=1.0):
        super(MELFAOneStepCell, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.cast = P.Cast()

    def set_sens(self, value):
        self.sens = value

    def construct(self, rgb, depth, tgt):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(rgb, depth, tgt)
        grads = self.grad(self.network, weights)(rgb, depth, tgt,
                                                 self.cast(F.tuple_to_array((self.sens,)), mstype.float32))
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


class MELFAInferCell(nn.Cell):

    def __init__(self, network):
        super(MELFAInferCell, self).__init__()
        self.network = network

    def construct(self, *args, **kwargs):
        return self.network(*args, **kwargs)


def model_eval(i_epoch, data, model, args, store_preds=False):
    model.set_train(False)
    losses, depth_preds, rgb_preds, depthrgb_preds, tgts = [], [], [], [], []
    for batch in data:
        new_batch = {'A': batch[0], 'B': batch[1], 'label': batch[2]}
        batch = new_batch
        loss, depth_rgb_logits, rgb_logits, depth_logits, tgt = model_forward_eval(i_epoch, model, args, batch)
        losses.append(loss.item())

        depth_pred = depth_logits.argmax(axis=1).asnumpy()
        rgb_pred = rgb_logits.argmax(axis=1).asnumpy()
        depth_rgb_pred = depth_rgb_logits.argmax(axis=1).asnumpy()

        depth_preds.append(depth_pred)
        rgb_preds.append(rgb_pred)
        depthrgb_preds.append(depth_rgb_pred)
        tgt = tgt.asnumpy()
        tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    depth_preds = [l for sl in depth_preds for l in sl]
    rgb_preds = [l for sl in rgb_preds for l in sl]
    depthrgb_preds = [l for sl in depthrgb_preds for l in sl]
    metrics["depth_acc"] = accuracy_score(tgts, depth_preds)
    metrics["rgb_acc"] = accuracy_score(tgts, rgb_preds)
    metrics["depthrgb_acc"] = accuracy_score(tgts, depthrgb_preds)
    return metrics


class EvalCallback(Callback):

    def __init__(self, args, test_loader, logger, best_metric):
        self.args = args
        self.test_loader = test_loader
        self.logger = logger
        self.best_metric = best_metric
        self.n_no_improve = 0
        self.infer_cell = None

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_losses = cb_params.net_outputs.asnumpy()
        model = cb_params.network
        i_epoch = cb_params.cur_epoch_num
        print('😊', cur_losses)

        model.set_train(False)
        self.infer_cell = MELFAInferCell(model.network.model)
        metrics = model_eval(
            np.inf, self.test_loader, self.infer_cell, self.args, store_preds=True
        )
        self.logger.info("Train Loss: {:.4f}".format(np.mean(cur_losses)))
        log_metrics("val", metrics, self.logger)
        self.logger.info(
            "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
                "val", metrics["loss"], metrics["depth_acc"], metrics["rgb_acc"], metrics["depthrgb_acc"]
            )
        )
        tuning_metric = metrics["depthrgb_acc"]

        # early stopping
        is_improvement = tuning_metric > self.best_metric
        if is_improvement:
            self.best_metric = tuning_metric
            self.n_no_improve = 0
        else:
            self.n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "parameters_dict": model.parameters_dict(),
                "n_no_improve": self.n_no_improve,
                "best_metric": self.best_metric,
            },
            is_improvement,
            self.args.savedir,
        )

        if self.n_no_improve >= self.args.patience:
            self.logger.info("No improvement. Breaking out of loop.")
            run_context.request_stop()


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name, str(args.seed))
    os.makedirs(args.savedir, exist_ok=True)

    # transforms
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]
    train_transforms = list()
    train_transforms.append(vision.Resize((args.LOAD_SIZE, args.LOAD_SIZE)))
    train_transforms.append(vision.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)))
    train_transforms.append(vision.RandomHorizontalFlip())
    train_transforms.append(vision.ToTensor())
    train_transforms.append(vision.Normalize(mean=mean, std=std, is_hwc=False))
    val_transforms = list()
    val_transforms.append(vision.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    val_transforms.append(vision.ToTensor())
    val_transforms.append(vision.Normalize(mean=mean, std=std, is_hwc=False))

    # create datasets
    train_loader = ds.GeneratorDataset(
        source=oodConcDataset(args,
                              data_dir=os.path.join(args.data_path, 'train'),
                              transform=transforms.Compose(train_transforms),
                              rgb_threshold=args.sample,
                              depth_threshold=args.sample),
        column_names=['rgb', 'depth', 'tgt'],
        shuffle=True)
    train_loader = train_loader.batch(batch_size=args.batch_sz)
    test_loader = ds.GeneratorDataset(
        source=AlignedConcDataset(args,
                                  data_dir=os.path.join(args.data_path, 'test'),
                                  transform=transforms.Compose(val_transforms)),
        column_names=['rgb', 'depth', 'tgt'],
        shuffle=False)
    test_loader = test_loader.batch(batch_size=args.batch_sz)

    model = MELFA(args)
    net_with_loss = MELFAWithLoss(model, args)
    optimizer = get_optimizer(model, args)

    with open(os.path.join(args.savedir, "args.pt"), 'wb') as f:
        pickle.dump(args, f)
    logger = create_logger("%s/logfile.log" % args.savedir, args)

    callbacks = [EvalCallback(args, test_loader, logger, 0)]

    net_with_grads = MELFAOneStepCell(net_with_loss, optimizer)
    net_with_grads.set_train(True)
    net = Model(net_with_grads)
    net.train(args.max_epochs, train_loader, callbacks=callbacks)

    # load the best model, and test on best model
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.set_train(False)
    infer_cell = MELFAInferCell(model)
    test_metrics = model_eval(
        np.inf, test_loader, infer_cell, args, store_preds=True
    )
    logger.info(
        "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
            "Test", test_metrics["loss"], test_metrics["depth_acc"], test_metrics["rgb_acc"],
            test_metrics["depthrgb_acc"]
        )
    )
    log_metrics(f"Test", test_metrics, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    # FIXME: set device = GPU/Ascend
    # mindspore.set_context(device_target='Ascend', device_id=0)
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    cli_main()
