"""
network architechture for place recognition (Oxford dataset) with attention
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
import SPConvNets.models.gcn as attention
from SPConvNets.models.pointnet_epn_netvlad import STN3d
import config as cfg

def split_train(x, model):
    if x.shape[0] >=4:
        query_pcd, pos_pcd, neg_pcd, otherneg_pcd = torch.split(
            x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)

        x_query, _ = model(query_pcd)
        x_pos, _ = model(pos_pcd)
        x_neg, _ = model(neg_pcd)
        x_otherneg, _ = model(otherneg_pcd)
        x_frontend = torch.cat((x_query, x_pos, x_neg, x_otherneg), 0)
    elif x.shape[0] == 3:
        query_pcd, pos_pcd, neg_pcd = torch.split(
            x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)

        x_query, _ = model(query_pcd)
        x_pos, _ = model(pos_pcd)
        x_neg, _ = model(neg_pcd)
        x_frontend = torch.cat((x_query, x_pos, x_neg), 0)
    elif x.shape[0] == 2:
        query_pcd, pos_pcd = torch.split(
            x, [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)

        x_query, _ = model(query_pcd)
        x_pos, _ = model(pos_pcd)
        x_frontend = torch.cat((x_query, x_pos), 0)
    elif x.shape[0] == 1:
        x_frontend, _ = model(x)
    else:
        print('x.shape[0]', x.shape[0])
    
    return x_frontend


class EPN_GCN_NetVLAD(nn.Module):
    def __init__(self, opt):
        super(EPN_GCN_NetVLAD, self).__init__()
        self.opt = opt
        # epn param
        mlps=[[64,64], [128, 128]]
        out_mlps=[128, self.opt.model.output_num]
        strides=[1, 1]
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides)

        self.gnn = attention.GCN(4, self.opt.model.output_num, 10, ['self','cross','self'])
        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, 128, self.opt.model.output_num
        Global Feature: B, self.opt.global_feature_dim
        '''
        select_index = np.arange(0, cfg.NUM_POINTS, cfg.NUM_POINTS//cfg.NUM_SELECTED_POINTS)
        # select_index = torch.randint(0, cfg.NUM_POINTS, (cfg.NUM_SELECTED_POINTS,))

        if x.shape[0] >=4:
            ####################################################
            # STEP 1: Frontend, learn invariant local features #
            ####################################################
            # input point cloud with shape [B, N, 3]
            query_pcd, pos_pcd, neg_pcd, otherneg_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:,select_index, :]
            pos_pcd = pos_pcd[:,select_index, :]
            neg_pcd = neg_pcd[:,select_index, :]
            otherneg_pcd = otherneg_pcd[:,select_index, :]
            x_downsize = torch.cat((query_pcd, pos_pcd, neg_pcd, otherneg_pcd), 0)
            # print('x after downsize', x_downsize.shape)

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_neg, _ = self.epn(neg_pcd)
            x_otherneg, _ = self.epn(otherneg_pcd)
            x_frontend = torch.cat((x_query, x_pos, x_neg, x_otherneg), 0)
            # local features with shape [B, C, N]
            # print('x after epn', x_frontend.shape)
        elif x.shape[0] == 3:
            ####################################################
            # STEP 1: Frontend, learn invariant local features #
            ####################################################
            # input point cloud with shape [B, N, 3]
            query_pcd, pos_pcd, neg_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:,select_index, :]
            pos_pcd = pos_pcd[:,select_index, :]
            neg_pcd = neg_pcd[:,select_index, :]
            x_downsize = torch.cat((query_pcd, pos_pcd, neg_pcd), 0)
            # print('x after downsize', x_downsize.shape)

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_neg, _ = self.epn(neg_pcd)
            x_frontend = torch.cat((x_query, x_pos, x_neg), 0)
            # local features with shape [B, C, N]
            # print('x after epn', x_frontend.shape)
        elif x.shape[0] == 2:
            ####################################################
            # STEP 1: Frontend, learn invariant local features #
            ####################################################
            # input point cloud with shape [B, N, 3]
            query_pcd, pos_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:,select_index, :]
            pos_pcd = pos_pcd[:,select_index, :]
            x_downsize = torch.cat((query_pcd, pos_pcd), 0)
            # print('x after downsize', x_downsize.shape)

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_frontend = torch.cat((x_query, x_pos), 0)
            # local features with shape [B, C, N]
            # print('x after epn', x_frontend.shape)
        elif x.shape[0] == 1:
            x_downsize = x[:,select_index, :]
            # print('x after downsize', x_downsize.shape)
            x_frontend, _ = self.epn(x_downsize)
            # print('x after epn', x_frontend.shape)
        else:
            print('x.shape[0]', x.shape[0])

        ######################################################
        # STEP 2: Attention, learn co-contextual information #
        ######################################################
        x_gcn = self.gnn(x_downsize, x_frontend)
        # print('x after gcn', x_gcn.shape)

        ###################################################################
        # STEP 3: NetVLAD, learn global descriptors for place recognition #
        ###################################################################
        x = self.netvlad(x_gcn)

        return x, x_frontend

class EPN_Atten_NetVLAD(nn.Module):
    def __init__(self, opt):
        super(EPN_Atten_NetVLAD, self).__init__()
        self.opt = opt
        # epn param
        mlps=[[64]]
        out_mlps=[64, self.opt.model.output_num]
        strides=[1,1]
        
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False, outblock='linear')

        self.atten = torch.nn.MultiheadAttention(self.opt.model.output_num, 4)

        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, 128, self.opt.model.output_num
        Global Feature: B, self.opt.global_feature_dim
        '''
        x_frontend = split_train(x, self.epn)

        x_atten, atten_weight = self.atten(x_frontend, x_frontend, x_frontend)

        x = self.netvlad(x_atten)

        return x, x_frontend

class Atten_EPN_NetVLAD(nn.Module):
    def __init__(self, opt):
        super(Atten_EPN_NetVLAD, self).__init__()
        self.opt = opt
        # epn param
        mlps=[[32], [32]]
        out_mlps=[32, self.opt.model.output_num]
        strides=[1, 1, 1]

        self.atten = torch.nn.MultiheadAttention(3, 3)
        
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False, outblock='linear')

        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, 128, self.opt.model.output_num
        Global Feature: B, self.opt.global_feature_dim
        '''
        x_atten, atten_weight = self.atten(x, x, x)

        x_frontend = split_train(x_atten, self.epn)

        x = self.netvlad(x_frontend)

        return x, x_frontend


class Atten_EPN_NetVLAD_select(nn.Module):
    def __init__(self, opt):
        super(Atten_EPN_NetVLAD_select, self).__init__()
        self.opt = opt
        # epn param
        mlps=[[32], [64]]
        out_mlps=[64, self.opt.model.output_num]
        strides=[1, 1, 1]

        self.atten = torch.nn.MultiheadAttention(3, 3, batch_first=True)
        
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False, outblock='linear')

        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, 128, self.opt.model.output_num
        Global Feature: B, self.opt.global_feature_dim
        '''        
        # Use attention to choose points and downsample
        # print('input x', x.shape)
        x_atten, atten_weight = self.atten(x, x, x)
        # print('x_weight', x_atten.shape)
        # print('x_atten', atten_weight.shape)
        atten_weight_sum = torch.sum(atten_weight, 2)
        # print('atten_weight', atten_weight.shape)

        # select_index = torch.randint(0, cfg.NUM_POINTS, (cfg.NUM_SELECTED_POINTS,))
        # print('input x', x.shape)

        if x.shape[0] >=4:
            query_pcd, pos_pcd, neg_pcd, otherneg_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            query_attn, pos_attn, neg_attn, otherneg_attn = torch.split(
                atten_weight_sum, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            # reduce size of point cloud
            query_pcd = query_pcd[:, torch.topk(query_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            pos_pcd = pos_pcd[:, torch.topk(pos_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            neg_pcd = neg_pcd[:, torch.topk(neg_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            otherneg_pcd = otherneg_pcd[:,torch.topk(otherneg_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_neg, _ = self.epn(neg_pcd)
            x_otherneg, _ = self.epn(otherneg_pcd)
            x_frontend = torch.cat((x_query, x_pos, x_neg, x_otherneg), 0)
        elif x.shape[0] == 3:
            query_pcd, pos_pcd, neg_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)
            query_attn, pos_attn, neg_attn = torch.split(
                atten_weight_sum, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:, torch.topk(query_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            pos_pcd = pos_pcd[:, torch.topk(pos_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            neg_pcd = neg_pcd[:, torch.topk(neg_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_neg, _ = self.epn(neg_pcd)
            x_frontend = torch.cat((x_query, x_pos, x_neg), 0)
        elif x.shape[0] == 2:
            query_pcd, pos_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)
            query_attn, pos_attn = torch.split(
                atten_weight_sum, [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:,torch.topk(query_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            pos_pcd = pos_pcd[:,torch.topk(pos_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_frontend = torch.cat((x_query, x_pos), 0)
        elif x.shape[0] == 1:
            query_pcd = x[:,torch.topk(atten_weight_sum, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]

            x_frontend, _ = self.epn(query_pcd)
        else:
            print('x.shape[0]', x.shape[0])

        # print('x_frontend', x_frontend.shape)

        x = self.netvlad(x_frontend)

        return x, atten_weight_sum

class EPN_CA_NetVLAD(nn.Module):
    def __init__(self, opt):
        super(EPN_CA_NetVLAD, self).__init__()
        self.opt = opt
        # epn param
        mlps=[[64]]
        out_mlps=[64, self.opt.model.output_num]
        strides=[1,1]
        
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False, outblock='linear')

        # self.atten = attention.CrossAttnetion(4, self.opt.model.output_num)
        self.atten = attention.CrossAttnetion_all(4, self.opt.model.output_num)
        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, 128, self.opt.model.output_num
        Global Feature: B, self.opt.global_feature_dim
        '''
        x_frontend = split_train(x, self.epn)

        x = self.netvlad(x_frontend)

        front_end = {'invariant':x_frontend, 'attention':x_atten}

        return x, x_frontend


class EPN_CA_NetVLAD_select(nn.Module):
    def __init__(self, opt):
        super(EPN_CA_NetVLAD_select, self).__init__()
        self.opt = opt

        # transformation
        self.trans = False
        self.stn = STN3d(num_points=cfg.NUM_POINTS, k=3, use_bn=False)

        # epn param
        mlps=[[64]]
        out_mlps=[64, self.opt.model.output_num]
        strides=[1,1]        
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False, outblock='linear')

        self.atten = attention.CrossAttnetion_all(4, self.opt.model.output_num)
        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, N', self.opt.model.output_num
        Global Feature: B, self.opt.global_feature_dim
        '''
        # transformation
        if self.trans:
            x = x.unsqueeze(1) # (B, 1, N, D)
            trans = self.stn(x) # B, 3, 3
            x = torch.matmul(torch.squeeze(x), trans) # B, N, 3

        select_index = torch.randint(0, cfg.NUM_POINTS, (cfg.NUM_SELECTED_POINTS,))
        # print('input x', x.shape)

        if x.shape[0] >=4:
            query_pcd, pos_pcd, neg_pcd, otherneg_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            # reduce size of point cloud
            query_pcd = query_pcd[:,select_index, :]
            pos_pcd = pos_pcd[:,select_index, :]
            neg_pcd = neg_pcd[:,select_index, :]
            otherneg_pcd = otherneg_pcd[:,select_index, :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_neg, _ = self.epn(neg_pcd)
            x_otherneg, _ = self.epn(otherneg_pcd)
            x_frontend = torch.cat((x_query, x_pos, x_neg, x_otherneg), 0)
        elif x.shape[0] == 3:
            query_pcd, pos_pcd, neg_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:,select_index, :]
            pos_pcd = pos_pcd[:,select_index, :]
            neg_pcd = neg_pcd[:,select_index, :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_neg, _ = self.epn(neg_pcd)
            x_frontend = torch.cat((x_query, x_pos, x_neg), 0)
        elif x.shape[0] == 2:
            query_pcd, pos_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:,select_index, :]
            pos_pcd = pos_pcd[:,select_index, :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_frontend = torch.cat((x_query, x_pos), 0)
        elif x.shape[0] == 1:
            query_pcd = x[:,select_index, :]

            x_frontend, _ = self.epn(query_pcd)
        else:
            print('x.shape[0]', x.shape[0])

        x_gcn = self.atten(x_frontend)

        x = self.netvlad(x_gcn)

        front_end = {'invariant':x_frontend, 'attention':x_gcn}

        return x, x_gcn


class CA_EPN_NetVLAD_select(nn.Module):
    def __init__(self, opt):
        super(CA_EPN_NetVLAD_select, self).__init__()
        self.opt = opt
        
        # self.atten = attention.CrossAttnetion(3, 3)
        # self.atten = attention.CrossAttnetion_all(3, 3)
        self.atten = attention.CrossAttnetion_all_weights(3, 3)        

        # epn param
        mlps=[[128]]
        out_mlps=[128, self.opt.model.output_num]
        strides=[1,1]
        self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False, outblock='linear')
        # self.epn = frontend.build_model(self.opt, mlps, out_mlps, strides, downsample=False)

        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, NUM_POINTS, D
        Local Feature: B, NUM_SELECTED_POINTS, self.opt.model.output_num
        Global Feature: B, self.opt.global_feature_dim
        '''
        # Use attention to choose points and downsample
        x_gcn = self.atten(x)
        x_gcn = torch.sum(x_gcn, 2)
        # print('x_gcn', x_gcn.shape)

        # select_index = torch.randint(0, cfg.NUM_POINTS, (cfg.NUM_SELECTED_POINTS,))
        # print('input x', x.shape)

        if x.shape[0] >=4:
            query_pcd, pos_pcd, neg_pcd, otherneg_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            query_attn, pos_attn, neg_attn, otherneg_attn = torch.split(
                x_gcn, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            # reduce size of point cloud
            query_pcd = query_pcd[:, torch.topk(query_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            pos_pcd = pos_pcd[:, torch.topk(pos_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            neg_pcd = neg_pcd[:, torch.topk(neg_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            otherneg_pcd = otherneg_pcd[:,torch.topk(otherneg_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_neg, _ = self.epn(neg_pcd)
            x_otherneg, _ = self.epn(otherneg_pcd)
            x_frontend = torch.cat((x_query, x_pos, x_neg, x_otherneg), 0)
        elif x.shape[0] == 3:
            query_pcd, pos_pcd, neg_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)
            query_attn, pos_attn, neg_attn = torch.split(
                x_gcn, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:, torch.topk(query_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            pos_pcd = pos_pcd[:, torch.topk(pos_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            neg_pcd = neg_pcd[:, torch.topk(neg_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_neg, _ = self.epn(neg_pcd)
            x_frontend = torch.cat((x_query, x_pos, x_neg), 0)
        elif x.shape[0] == 2:
            query_pcd, pos_pcd = torch.split(
                x, [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)
            query_attn, pos_attn = torch.split(
                x_gcn, [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)

            # reduce size of point cloud
            query_pcd = query_pcd[:,torch.topk(query_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]
            pos_pcd = pos_pcd[:,torch.topk(pos_attn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]

            x_query, _ = self.epn(query_pcd)
            x_pos, _ = self.epn(pos_pcd)
            x_frontend = torch.cat((x_query, x_pos), 0)
        elif x.shape[0] == 1:
            query_pcd = x[:,torch.topk(x_gcn, cfg.NUM_SELECTED_POINTS, 1)[1].squeeze(), :]

            x_frontend, _ = self.epn(query_pcd)
        else:
            print('x.shape[0]', x.shape[0])

        # print('x_frontend', x_frontend.shape)

        x = self.netvlad(x_frontend)

        front_end = {'invariant':x_frontend, 'attention':x_gcn}

        return x, x_gcn


class EPN_Transformer_NetVLAD(nn.Module):
    def __init__(self, opt):
        super(EPN_Transformer_NetVLAD, self).__init__()
        self.opt = opt
        # epn param
        mlps=[[64,64], [128, 128]]
        out_mlps=[128, self.opt.model.output_num]        
        self.epn = frontend.build_model(self.opt, mlps, out_mlps)
        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.opt.model.output_num, nhead=4, \
                                                    dim_feedforward=1024, activation='relu', batch_first=False, dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.conv_after_cat = nn.Conv2d(2048, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        # netvlad
        self.netvlad = M.NetVLADLoupe(feature_size=self.opt.model.output_num, max_samples=self.opt.num_selected_points, cluster_size=64,
                                     output_dim=self.opt.global_feature_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, 128, self.opt.model.output_num
        Global Feature: B, self.opt.global_feature_dim
        '''
        x_frontend = split_train(x, self.epn)

        # print('x_frontend', x_frontend.shape)
        x_atten = self.transformer_encoder(x_frontend)
        x_encoder = torch.cat((x_frontend, x_atten), dim=2)
        # print('x_encoder', x_encoder.shape)

        x_encoder = x_encoder.transpose(1, 2)
        x_encoder = x_encoder.unsqueeze(3)
        # print('x_encoder', x_encoder.shape)

        x_encoder = self.relu(self.conv_after_cat(x_encoder))
        x_encoder = F.normalize(x_encoder, dim=2)        
        x_encoder = torch.squeeze(x_encoder, 3)
        x_encoder = x_encoder.transpose(1, 2)
        # print('x_encoder', x_encoder.shape)

        x_output = self.netvlad(x_encoder)
        # print('x_output', x_output.shape)

        return x_output, x_frontend