import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

# TODO: convert to Mindspore version
from models.image_ms import ImageEncoder


def update_parameters_name(cell, prefix=''):
    # Update the name of the 'weight' parameter
    for param in cell.trainable_params():
        param.name = prefix + param.name


class MELFA(nn.Cell):

    def __init__(self, args):
        super(MELFA, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        depth_rgb_last_size = args.img_hidden_sz * args.num_image_embeds * 2
        self.clf_depth = nn.CellList()
        self.clf_rgb = nn.CellList()
        self.tcp_depth = nn.CellList()
        self.tcp_rgb = nn.CellList()
        self.clf_depth_rgb = nn.CellList()


        for hidden in args.hidden:
            self.clf_rgb.append(nn.Dense(rgb_last_size, hidden))
            print(rgb_last_size)
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            rgb_last_size = hidden
        self.dense1 = nn.Dense(rgb_last_size, args.n_classes)
        self.clf_rgb.append(self.dense1)

        for hidden in args.hidden:
            self.clf_depth.append(nn.Dense(depth_last_size, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.dense2 = nn.Dense(depth_last_size, args.n_classes)
        self.clf_depth.append(self.dense2)

        for hidden in args.hidden:
            self.tcp_rgb.append(nn.Dense(rgb_last_size, hidden))
            print(rgb_last_size)
            self.tcp_rgb.append(nn.ReLU())
            self.tcp_rgb.append(nn.Dropout(args.dropout))
            rgb_last_size = hidden
        self.dense3 = nn.Dense(rgb_last_size, 1)
        self.tcp_rgb.append(self.dense3)

        for hidden in args.hidden:
            self.tcp_depth.append(nn.Dense(depth_last_size, hidden))
            self.tcp_depth.append(nn.ReLU())
            self.tcp_depth.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.dense4 = nn.Dense(depth_last_size, 1)
        self.tcp_depth.append(self.dense4)

        self.dense1.update_parameters_name('clf_rgb.', True)
        self.dense2.update_parameters_name('clf_depth.', True)
        self.dense3.update_parameters_name('tcp_rgb.', True)
        self.dense4.update_parameters_name('tcp_depth.', True)

    def construct(self, rgb, depth):
        print('#', end='')
        rgb, depth = rgb.squeeze(), depth.squeeze()
        depth = self.depthenc(depth)
        depth = P.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = P.flatten(rgb, start_dim=1)

        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)

        rgb_tcp_out = rgb
        for layer in self.tcp_rgb:
            rgb_tcp_out = layer(rgb_tcp_out)

        depth_tcp_out = depth
        for layer in self.tcp_depth:
            depth_tcp_out = layer(depth_tcp_out)

        # depth_rgb_out = (depth_out + rgb_out)/2
        # FIXME: mindspore.ops.log supports type float16, float32
        rgb_energy = P.log(P.sum(P.exp(rgb_out), dim=1))
        depth_energy = P.log(P.sum(P.exp(depth_out), dim=1))
        # rgb_energy_diff = torch.log((torch.sum(torch.exp(rgb_out), dim=1))/(torch.sum(torch.exp(depth_out), dim=1)))/15
        # depth_energy_diff = torch.log((torch.sum(torch.exp(depth_out), dim=1))/(torch.sum(torch.exp(rgb_out), dim=1)))/15

        # rgb_energy_diff = (torch.sum(torch.exp(rgb_out), dim=1)-torch.sum(torch.exp(depth_out), dim=1))/10
        # depth_energy_diff = (torch.sum(torch.exp(depth_out), dim=1)-torch.sum(torch.exp(rgb_out), dim=1))/10
        # rgb_conf = torch.log(torch.sum(torch.exp(rgb_tcp_out), dim=1))/10
        # depth_conf = torch.log(torch.sum(torch.exp(depth_tcp_out), dim=1))/10
        # rgb_conf = torch.nn.Sigmoid()(rgb_tcp_out)
        # depth_conf = torch.nn.Sigmoid()(depth_tcp_out)
        # rgb_conf = F.sigmoid(rgb_energy_diff)
        # depth_conf = F.sigmoid(depth_energy_diff)

        # print(rgb_conf.shape)
        # print(rgb_out.shape)
        # print(depth_conf.shape)
        # print(depth_out.shape)
        rgb_conf = rgb_energy / 10
        depth_conf = depth_energy / 10

        rgb_conf = P.reshape(rgb_conf, (-1, 1))
        depth_conf = P.reshape(depth_conf, (-1, 1))
        # rgb_energy = torch.reshape(rgb_energy, (-1,1))
        # depth_energy = torch.reshape(depth_energy, (-1,1))
        # print(rgb_energy_diff)
        # print(depth_energy_diff)
        # print(rgb_conf)
        # print(depth_conf)
        # exit()
        # print(torch.cat([depth_energy, rgb_energy], dim=1))
        ### LATE FUSION
        depth_rgb_out = (depth_out * depth_conf + rgb_out * rgb_conf)

        return depth_rgb_out, rgb_out, depth_out, depth_conf, rgb_conf

    def get_feature(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = P.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = P.flatten(rgb, start_dim=1)

        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        return rgb, depth, rgb_out, depth_out

