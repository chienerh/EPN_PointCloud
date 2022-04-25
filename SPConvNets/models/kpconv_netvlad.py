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

class KPConvNetVLAD(nn.Module):
    def __init__(self, opt):
        super(KPConvNetVLAD, self).__init__()
        self.opt = opt
        
        # epn param
        mlps=[[64], [128]]
        out_mlps=[128, self.opt.model.output_num]
        strides=[2, 1]
        self.kpconv = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False, outblock='linear')
        print('kpconv network', self.kpconv)
        
        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D_input=3
        Local Feature: B, N', D_local
        Global Feature: B, D_output
        '''
        x, _ = self.kpconv(x)
        x_frontend = x

        x = self.netvlad(x)

        return x, x_frontend