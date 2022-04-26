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
        mlps=[[128]]
        out_mlps=[128, self.opt.model.output_num]
        strides=[1, 1]
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False, outblock='linear')
        print('EPN', self.epn)
        
        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D_input=3
        Local Feature: B, N', D_local
        Global Feature: B, D_output
        '''
        # print('EPNNetVLAD x1', x.shape)
        B, _, _ = x.shape
        x_frontend = torch.empty(size=(B, self.opt.num_selected_points, self.opt.model.output_num), device=x.device)
        for i in range(B):
            x_onlyone = x[i, :, :].unsqueeze(0)
            x_onlyone = self.downsample_pointcloud(x_onlyone)
            x_frontend[i], _ = self.epn(x_onlyone)
        # print('x after epn', x_frontend.shape)

        x_out = self.netvlad(x_frontend)

        return x_out, x_frontend

    def downsample_pointcloud(self, x_input):
        select_index = torch.randint(0, cfg.NUM_POINTS, (self.opt.num_selected_points,))

        # reduce size of point cloud
        x_downsampled = x_input[:, select_index, :].view(1, self.opt.num_selected_points, 3)
        
        return x_downsampled