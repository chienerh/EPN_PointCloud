"""
Test output feature from the network
"""
import numpy as np
import socket
import importlib
import os
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from SPConvNets.options import opt as opt_oxford
from importlib import import_module
import config as cfg


def visualize_attention():
    def load_pcd_file(filename):
        pc=np.load(filename)
        return pc

    def load_pc_file(filename):
        #returns Nx3 matrix
        pc=np.fromfile(filename, dtype=np.float64)
        pc=np.reshape(pc,(pc.shape[0]//3,3))
        return pc

    def load_model(opt):
        # build model
        if opt.model.model == 'atten_epn_netvlad_select':
            from SPConvNets.models.epn_gcn_netvlad import Atten_EPN_NetVLAD_select
            model = Atten_EPN_NetVLAD_select(opt_oxford)
        else:
            print('Model not available')
        
        # load pretrained weight
        if opt.resume_path.split('.')[1] == 'pth':
            saved_state_dict = torch.load(opt.resume_path)
        elif opt.resume_path.split('.')[1] == 'ckpt':
            checkpoint = torch.load(opt.resume_path)
            saved_state_dict = checkpoint['state_dict']    
        model.load_state_dict(saved_state_dict)
        model = nn.DataParallel(model)

        return model
    
    def get_global_descriptor(model, network_input, opt):
        network_input = network_input.reshape((1, network_input.shape[0], network_input.shape[1]))
        network_input = torch.Tensor(network_input).float().cuda()

        # get output features from the model
        model = model.eval()
        network_output, frontend_output = model(network_input)
        
        frontend_output = frontend_output.detach().cpu().numpy() #[:, :, 0].reshape((1024,))
        print('frontend_output', frontend_output.shape)

        # frontend_output = frontend_output[:,0,:]
        frontend_output = frontend_output.reshape((-1,))

        # tensor to numpy
        network_output = network_output.detach().cpu().numpy()[0, :]
        network_output = network_output.astype(np.double)
        frontend_output = frontend_output.astype(np.double)
        
        print('network_output', network_output.shape)
        return network_output, frontend_output

    opt_oxford.batch_size = 1
    opt_oxford.no_augmentation = True # TODO
    opt_oxford.model.model = 'atten_epn_netvlad_select'
    opt_oxford.device = torch.device('cuda')

    # IO
    opt_oxford.model.input_num = cfg.NUM_POINTS #4096
    opt_oxford.model.output_num = cfg.LOCAL_FEATURE_DIM #1024
    opt_oxford.global_feature_dim = cfg.FEATURE_OUTPUT_DIM #256

    # place recognition
    opt_oxford.num_points = opt_oxford.model.input_num
    opt_oxford.num_selected_points = cfg.NUM_SELECTED_POINTS
    # opt_oxford.num_selected_points = cfg.NUM_POINTS//(2*int(2*cfg.NUM_POINTS/1024))
    opt_oxford.pos_per_query = 1
    opt_oxford.neg_per_query = 1

    # pretrained weight
    opt_oxford.resume_path = 'pretrained_model/atten_epn_netvlad_select_seq567_64_1024_ds1024.ckpt'
    
    # input file
    input_folder = 'results/test_network_output/'
    input_filename = os.path.join(input_folder, '0_anchor.npy')
    input_pointcloud = load_pcd_file(os.path.join(input_filename))
    print('input_pointcloud', input_pointcloud.shape)
    print('input_pointcloud', input_pointcloud[0,:])

    # rotate and translate
    rotated_pointcloud = load_pcd_file('results/test_network_output/0_rotated.npy')
    translated_pointcloud = load_pcd_file('results/test_network_output/0_translated.npy')
    rotated_translated_pointcloud = load_pcd_file('results/test_network_output/0_rotated_translated.npy')
    translated_rotated_pointcloud = load_pcd_file('results/test_network_output/0_translated_rotated.npy')
    
    # anchor, positive, negative
    positive_pointcloud = load_pcd_file('results/test_network_output/0_positive.npy')
    negative_pointcloud = load_pcd_file('results/test_network_output/0_negative.npy')

    model = load_model(opt_oxford)
    
    with torch.no_grad():          
        # generate descriptors from point clouds
        _, output_atten = get_global_descriptor(model, input_pointcloud, opt_oxford)
        # _, rotated_atten = get_global_descriptor(model, rotated_pointcloud, opt_oxford)
        # _, translated_atten = get_global_descriptor(model, translated_pointcloud, opt_oxford)
        # _, rotated_translated_atten = get_global_descriptor(model, rotated_translated_pointcloud, opt_oxford)
        # _, translated_rotated_atten = get_global_descriptor(model, translated_rotated_pointcloud, opt_oxford)
        # # positives and negatives
        # _, positive_frontend = get_global_descriptor(model, positive_pointcloud, opt_oxford)
        # _, negative_frontend = get_global_descriptor(model, negative_pointcloud, opt_oxford)
    downsample_ind = np.argpartition(output_atten, -cfg.NUM_SELECTED_POINTS)[-cfg.NUM_SELECTED_POINTS:]
    downsampled_pcd = input_pointcloud[downsample_ind, :]
    cut_off_atten_thres = output_atten[downsample_ind[-1]]
    print('cut off attention weight: ', cut_off_atten_thres)
    view_angle = 45

    # # visualize input point clouds

    # fig = plt.figure()
    # ax = plt.axes(projection ="3d")
    # ax.scatter(input_pointcloud[:, 0], input_pointcloud[:, 1], input_pointcloud[:, 2], marker='.', c='C0', s=5, label='input point cloud')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.view_init(20, view_angle)
    # ax.set_title('input point cloud')
    # plt.savefig(os.path.join(input_folder, '0_intput_point_cloud.png'))

    
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    scatter_plot = ax.scatter(input_pointcloud[:, 0], input_pointcloud[:, 1], input_pointcloud[:, 2], marker='.', c=output_atten, s=5, label='attention weight')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(20, view_angle)
    ax.set_title('attention weight')
    plt.colorbar(scatter_plot)
    plt.savefig(os.path.join(input_folder, '0_self_attention_weight_ds1024.png'))


    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    scatter_plot = ax.scatter(input_pointcloud[:, 0], input_pointcloud[:, 1], input_pointcloud[:, 2], marker='.', c=output_atten<cut_off_atten_thres, s=5, label='Downsample or not')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(20, view_angle)
    ax.set_title('downsample hightlight (1 for dropping, 0 for keeping)')
    plt.colorbar(scatter_plot)
    plt.savefig(os.path.join(input_folder, '0_downsample_highlight_1024.png'))
    
    # fig = plt.figure()
    # ax = plt.axes(projection ="3d")
    # ax.scatter(downsampled_pcd[:, 0], downsampled_pcd[:, 1], downsampled_pcd[:, 2], marker='.', c='C0', s=5, label='downsampled point cloud')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.view_init(20, view_angle)
    # ax.set_title('downsampled point cloud')
    # plt.savefig(os.path.join(input_folder, '0_downsampled_point_cloud.png'))


if __name__ == "__main__":
    visualize_attention()

