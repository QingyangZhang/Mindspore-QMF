#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Amirreza Shaban

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: amirreza
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def init_weights(m):
 print(m)
 if type(m) == nn.Linear:
   print(m.weight)
 else:
   print('error')

class MMTM(nn.Module):
  def __init__(self, dim_visual, dim_skeleton, ratio):
    super(MMTM, self).__init__()
    dim = dim_visual + dim_skeleton
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Linear(dim, dim_out)

    self.fc_visual = nn.Linear(dim_out, dim_visual)
    self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # initialize
    with torch.no_grad():
      self.fc_squeeze.apply(init_weights)
      self.fc_visual.apply(init_weights)
      self.fc_skeleton.apply(init_weights)

  def forward(self, visual, skeleton):

    squeeze_array = [visual, skeleton]
    
    squeeze = torch.cat(squeeze_array, 1)
    squeeze = self.avgpool(squeeze)
    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)
    vis_out = self.fc_visual(excitation)
    sk_out = self.fc_skeleton(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(visual.shape) - len(vis_out.shape)
    vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

    dim_diff = len(skeleton.shape) - len(sk_out.shape)
    sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

    return visual * vis_out, skeleton * sk_out

class MMTNet(nn.Module):
  def __init__(self, args):
    super(MMTNet, self).__init__()
    self.rgb_model = None
    self.depth_model = None
    self.final_pred = None

    self.mmtm0 = MMTM(128, 128, 4)
    self.mmtm1 = MMTM(256, 256, 4)
    self.mmtm2 = MMTM(512, 512, 4)

    #set_rgb_depth_nets(models.resnet18(), models.resnet18())
    self.rgb_model = models.resnet18()
    self.depth_model = models.resnet18()
    state_dict = torch.load(args.CONTENT_MODEL_PATH)

    self.rgb_model.load_state_dict(state_dict)
    self.depth_model.load_state_dict(state_dict)
    
    self.rgb_classifier = nn.Linear(512, args.n_classes)
    self.depth_classifier = nn.Linear(512, args.n_classes)
    self.rgb_dropout = nn.Dropout(args.dropout)  # dropout
    self.depth_dropout = nn.Dropout(args.dropout)  # dropout

    self.final_pred = nn.Linear(1024, args.n_classes)

  def get_mmtm_params(self):
    parameters = [
                {'params': self.mmtm0.parameters()},
                {'params': self.mmtm1.parameters()},
                {'params': self.mmtm2.parameters()}
                         ]
    return parameters

  def forward(self, rgb, depth):
    #################################################################
    ################################################ DEPTH INIT BLOCK
    depth_resnet = self.depth_model
    B, W, H, C = depth.size()

    # 1st conv
    depth_out = depth_resnet.conv1(depth)
    depth_out = depth_resnet.bn1(depth_out)
    depth_out = depth_resnet.relu(depth_out)
    depth_out = depth_resnet.maxpool(depth_out)

    # 1st residual block
    depth_out = depth_resnet.layer1(depth_out)

    # 2nd residual block
    depth_out = depth_resnet.layer2(depth_out)
    depth_out_p0 = depth_out

    #################################################################
    ################################################ VISUAL INIT BLOCK
    rgb_resnet = self.rgb_model

    # Changing temporal and channel dim to fit the inflated resnet input requirements
    B, W, H, C = rgb.size()

    # 1st conv
    rgb_out = rgb_resnet.conv1(rgb)
    rgb_out = rgb_resnet.bn1(rgb_out)
    rgb_out = rgb_resnet.relu(rgb_out)
    rgb_out = rgb_resnet.maxpool(rgb_out)

    # 1st residual block
    rgb_out = rgb_resnet.layer1(rgb_out)

    # 2nd residual block
    rgb_out = rgb_resnet.layer2(rgb_out)
    #rgb_out_p0 = rgb_out

    #################################### FIRST MMTM
    #fm2, out5_max ==> fm2, out5_p0 (out5_p1)
    #print(rgb_out_p0.shape)
    #print(depth_out_p0.shape)
    #exit()
    rgb_out, depth_out = self.mmtm0(rgb_out, depth_out)
    ####################################
    # DEPTH
    # 3rd residual block
    depth_out = depth_resnet.layer3(depth_out)
    #depth_out_p1 = depth_out

    # RGB
    # 3rd residual block
    rgb_out = rgb_resnet.layer3(rgb_out)
    #rgb_out_p1 = rgb_out

    ###################################### SECOND MMTM
    #fm3, out7 ==> fm3, out7
    rgb_out, depth_out = self.mmtm1(rgb_out, depth_out)
    ######################################

    # RGB
    # 4th residual block
    rgb_out = rgb_resnet.layer4(rgb_out)

    # Depth
    # 4th residual block
    depth_out = depth_resnet.layer4(depth_out)

    ########################################## THIRD MMTM
    #final_fm, out8 => final_fm, out8
    rgb_out, depth_out = self.mmtm2(rgb_out, depth_out)
    ### #######################################
    # Final Classifier
    rgb_out = F.avg_pool2d(rgb_out, 7)
    depth_out = F.avg_pool2d(depth_out, 7)
    rgb_out = rgb_out.squeeze()
    depth_out = depth_out.squeeze()

    #rgb_out = self.rgb_dropout(rgb_out)
    #depth_out = self.depth_dropout(depth_out)

    pred = self.final_pred(torch.cat([depth_out, rgb_out], dim=-1))

    rgb_pred = self.rgb_classifier(rgb_out)
    depth_pred = self.depth_classifier(depth_out)

    ### LATE FUSION
    # pred = (depth_pred + rgb_pred)/2
    
    return pred, rgb_pred, depth_pred
