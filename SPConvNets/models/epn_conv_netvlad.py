"""
network architechture for place recognition (Oxford dataset)
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import time
from collections import OrderedDict
import json
import vgtk
import SPConvNets.utils as M
import vgtk.spconv.functional as L
import SPConvNets.models.pr_so3net_pn as frontend
import config as cfg

class EPNConvNetVLAD(nn.Module):
    def __init__(self, opt):
        super(EPNConvNetVLAD, self).__init__()
        self.opt = opt
        
        # epn param
        mlps=[[64,64], [128, 128]]
        out_mlps=[128, self.opt.model.output_num]
        strides=[2, 2]
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, self.opt.model.initial_radius_ratio, self.opt.model.sampling_ratio)
        
        # conv
        self.conv1 = torch.nn.Conv2d(1024, 1024, (1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(1024)

        # netvlad
        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, 128, 64
        Global Feature: B, 128
        '''
        # print('x input', x.shape)
        x, attn = self.epn(x)
        x_frontend = x
        # print('x after epn', x_frontend.shape) # B, N, D

        # conv
        x = x.transpose(1, 2)
        x = x.unsqueeze(3) # B, D, N, 1
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N, 1
        x = x.squeeze(3) # B, D, N
        x = x.transpose(1, 2) # B, N, D
        x_conv = x

        x = self.netvlad(x)

        return x, x_conv