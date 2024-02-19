#!/usr/bin/env python3
#

import contextlib
import numpy as np
import random
import shutil
import os

import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter
from mindspore.train.serialization import save_checkpoint as save, load_checkpoint as load

import pickle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, filename)
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))


def load_checkpoint(model, path):
    with open(path, 'rb') as f:
        best_checkpoint = pickle.load(f)
    mindspore.load_param_into_net(model, best_checkpoint["parameters_dict"])


def log_metrics(set_name, metrics, logger):
    logger.info(
        "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}".format(
            set_name, metrics["loss"], metrics["depth_acc"], metrics["rgb_acc"]
        )
    )
