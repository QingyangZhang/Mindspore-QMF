#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pprint

import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

import mindcv


class ImageEncoder(nn.Cell):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        """
        model = models.__dict__['resnet18'](num_classes=365)
        # places model downloaded from http://places2.csail.mit.edu/
        checkpoint = torch.load(args.CONTENT_MODEL_PATH, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        print('content model pretrained using place')
        """
        # FIXME: use pretrained=True here
        model = mindcv.create_model('resnet18', pretrained=True)
        # state_dict = torch.load(args.CONTENT_MODEL_PATH)
        #state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        # model.load_state_dict(state_dict)
        modules = list(model.cells())[:-2]
        # for module in modules:
        #     print('$$$$', module)
        # self._all_modules = list(model.cells())
        self.model = nn.SequentialCell(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def construct(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        # print('for debug')
        # for i in range(len(self._all_modules)):
        #     submodules = self._all_modules[:i+1]
        #     tmp_model = nn.SequentialCell(*submodules)
        #     tmp_out = tmp_model(x)
        #     print('tmp_out.shape', tmp_out.shape)

        # print('x ', x.shape)
        out = self.model(x)
        # print('after model', out.shape)
        out = self.pool(out)
        # print('after pool', out.shape)
        out = P.flatten(out, start_dim=2)
        # print('after flatten', out.shape)
        out = out.transpose((0, 2, 1))
        return out  # BxNx2048
