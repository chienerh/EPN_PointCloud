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

class EPNNetVLAD(nn.Module):
    def __init__(self, opt):
        super(EPNNetVLAD, self).__init__()
        self.opt = opt
        
        # epn param
        mlps=[[64,64], [128, 128]]
        out_mlps=[128, self.opt.model.output_num]
        strides=[2, 2]
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, self.opt.model.initial_radius_ratio, self.opt.model.sampling_ratio)
        
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
        # print('x after epn', x_frontend.shape)

        x = self.netvlad(x)

        return x, x_frontend