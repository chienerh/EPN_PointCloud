"""
Code taken from https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/models/PointNetVlad.py
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
import SPConvNets.models.pr_so3net_pn as pr_so3net_pn
import SPConvNets.utils as M


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3), stride=2)
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1), stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1), stride=2)
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1), stride=2)
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        '''
        INPUT: (22, 1, 4096, 3) [Bx(1+P+N+1), 1, N, D]
        OUTPUT: (22, 1024, 2096, 1) if not max pool
        '''
        batchsize = x.size()[0]
        trans = self.stn(x) # 22, 3, 3
        x = torch.matmul(torch.squeeze(x), trans) # 22, 4096, 3
        x = x.view(batchsize, 1, -1, 3) # 22, 1, 4096, 3
        x = F.relu(self.bn1(self.conv1(x))) # 22, 64, 4096, 1
        x = F.relu(self.bn2(self.conv2(x))) # 22, 64, 4096, 1
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans


class PointNetEPN_NetVLAD(nn.Module):
    def __init__(self, opt):
        super(PointNetEPN_NetVLAD, self).__init__()
        self.opt = opt
        self.point_net = PointNetfeat(num_points=4096, global_feat=True,
                                      feature_transform=False, max_pool=False)
        mlps=[[64,64], [128,128]]
        out_mlps=[128, 1024]
        self.epn = pr_so3net_pn.build_model(self.opt, mlps=mlps, out_mlps=out_mlps)
        self.net_vlad = M.NetVLADLoupe(feature_size=1024, max_samples=2*self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        # print('x', x.shape)
        x_unsqueeze = x.unsqueeze(1)
        x_pointnet = self.point_net(x_unsqueeze) # Bx(1+P+N+1), LOCAL_DIM, N, 1
        # print('x_pointnet', x_pointnet.shape)
        x_pointnet = x_pointnet.transpose(1, 3).contiguous()
        x_pointnet = x_pointnet.view((-1, self.opt.num_selected_points, 1024))
        # print('x_pointnet', x_pointnet.shape)
        x_epn, _ = self.epn(x)
        # print('x_epn', x_epn.shape)
        x_frontend = torch.cat((x_pointnet, x_epn), 1) # Where to concatenate?
        # print('x_frontend', x_frontend.shape)
        x = self.net_vlad(x_frontend)
        return x, x_frontend


class PointNetVLAD_EPNNetVLAD(nn.Module):
    def __init__(self, opt):
        super(PointNetVLAD_EPNNetVLAD, self).__init__()
        self.opt = opt
        self.point_net = PointNetfeat(num_points=4096, global_feat=True,
                                      feature_transform=False, max_pool=False)
        self.net_vlad1 = M.NetVLADLoupe(feature_size=1024, max_samples=4096, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim//2, gating=True, add_batch_norm=True,
                                     is_training=True)
        mlps=[[64,64], [128,128]]
        out_mlps=[128, 1024]
        self.epn = pr_so3net_pn.build_model(self.opt, mlps=mlps, out_mlps=out_mlps)
        self.net_vlad2 = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim//2, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        # print('x input', x.shape)
        # PointNetVLAD
        x_unsqueeze = x.unsqueeze(1)
        x_pointnet = self.point_net(x_unsqueeze) # Bx(1+P+N+1), LOCAL_DIM, N, 1
        # print('x_pointnet', x_pointnet.shape)
        x_pointnet = x_pointnet.transpose(1, 3).contiguous()
        x_pointnet = x_pointnet.view((-1, 4096, 1024))
        # print('x_pointnet reshaped', x_pointnet.shape)
        x_pointnetvlad = self.net_vlad1(x_pointnet)
        # print('x_pointnetvlad', x_pointnetvlad.shape)
        # EPNNetVLAD
        x_epn, _ = self.epn(x)
        # print('x_epn', x_epn.shape)
        x_epnnetvlad = self.net_vlad2(x_epn)
        # print('x_epnnetvlad', x_epnnetvlad.shape)

        x_output = torch.cat((x_pointnetvlad, x_epnnetvlad), 1)
        x_frontend = torch.cat((x_pointnet, x_epn), 1)
        # print('x_output', x_output.shape)
        return x_output, x_frontend