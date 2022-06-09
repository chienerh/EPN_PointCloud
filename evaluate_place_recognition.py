"""
Code taken from https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/evaluate.py
"""
from SPConvNets.options import opt as opt_oxford
from importlib import import_module
import torch.nn as nn
import numpy as np
import pickle
import torch
import os
import sys
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import config as cfg


def get_sets_dict(filename):
	#[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
	with open(filename, 'rb') as handle:
		trajectories = pickle.load(handle)
		print("Trajectories Loaded.")
		return trajectories

def load_pc_file(filename):
	#returns Nx3 matrix
	pc=np.fromfile(filename, dtype=np.float64)

	if(pc.shape[0]!= cfg.NUM_POINTS*3):
		print("Error in pointcloud shape")
		return np.array([])

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	return pc

def load_pc_files(filenames, opt):
	pcs=[]
	for filename in filenames:
		#print(filename)
		pc=load_pc_file(os.path.join(opt.dataset_path,filename))
		if(pc.shape[0]!=cfg.NUM_POINTS):
			continue
		pcs.append(pc)
	pcs=np.array(pcs)
	return pcs

def evaluate():
    opt_oxford.batch_size = 1
    opt_oxford.no_augmentation = True # TODO
    opt_oxford.model.model = 'epn_netvlad' #'atten_epn_netvlad_select' # 'epn_ca_netvlad_select', 'epn_ca_netvlad', 'epn_netvlad' , 'pointnetepn_netvlad', 'pointnetvlad_epnnetvlad'
    opt_oxford.device = torch.device('cuda')

    # IO
    opt_oxford.model.input_num = cfg.NUM_POINTS #4096
    opt_oxford.model.output_num = cfg.LOCAL_FEATURE_DIM #1024
    opt_oxford.global_feature_dim = cfg.FEATURE_OUTPUT_DIM #256

    # place recognition
    opt_oxford.num_points = opt_oxford.model.input_num
    opt_oxford.num_selected_points = cfg.NUM_SELECTED_POINTS
    # opt_oxford.num_selected_points = cfg.NUM_POINTS//(2*int(2*cfg.NUM_POINTS/1024))
    opt_oxford.pos_per_query = cfg.TRAIN_POSITIVES_PER_QUERY
    opt_oxford.neg_per_query = cfg.TRAIN_NEGATIVES_PER_QUERY

    opt_oxford.model.search_radius = 0.35
    opt_oxford.model.initial_radius_ratio = 0.2
    opt_oxford.model.sampling_ratio = 0.8 #0.8

    # pretrained weight
    # opt_oxford.resume_path = 'pretrained_model/epn_transformer_conv_netvlad_seq567.ckpt'
    # opt_oxford.result_folder = 'results/pr_evaluation_epn_transformer_conv_netvlad_seq567'
    opt_oxford.resume_path = 'pretrained_model/epn_netvlad_seq567_random_ds1600.ckpt'
    opt_oxford.result_folder = 'results/pr_evaluation_epn_netvlad_seq567_random_ds1600_evalall'

    """evaluation"""
    '''rotation vs. translation vs. partial overlap'''
    '''eval on training set'''
    # opt_oxford.database_file = '/home/cel/data/benchmark_datasets/oxford_train_evaluation_database_part.pickle'
    # opt_oxford.query_file = '/home/cel/data/benchmark_datasets/oxford_train_evaluation_query_part.pickle'
    # opt_oxford.result_folder = 'results/pr_evaluation_epn_ca_netvlad_select_parttrain'
    '''eval on testing set'''
    # opt_oxford.database_file = '/home/cel/data/benchmark_datasets/oxford_evaluation_database_tran.pickle'
    # opt_oxford.query_file = '/home/cel/data/benchmark_datasets/oxford_evaluation_query_tran.pickle'
    # opt_oxford.result_folder = 'results/pr_evaluation_epn_conv_netvlad_tran'

    '''seq 5-7'''
    # opt_oxford.database_file = '/home/cel/data/benchmark_datasets/oxford_evaluation_database_seq567.pickle'
    # opt_oxford.query_file = '/home/cel/data/benchmark_datasets/oxford_evaluation_query_seq567.pickle'
    # opt_oxford.pointnetvlad_result_folder = 'results/pr_evaluation_pointnetvlad_seq567'

    '''whole dataset'''
    opt_oxford.database_file = '/home/cel/data/benchmark_datasets/oxford_evaluation_database.pickle'
    opt_oxford.query_file = '/home/cel/data/benchmark_datasets/oxford_evaluation_query.pickle'
    opt_oxford.pointnetvlad_result_folder = 'results/pr_evaluation_pointnetvlad_seq567_evalall'
    opt_oxford.scancontext_result_folder = 'results/pr_evaluation_scan_context_oxford_evalall'
    opt_oxford.m2dp_result_folder = 'results/pr_evaluation_m2dp_evalall'

    '''overlap dataset'''
    # opt_oxford.database_file = '/home/cel/data/benchmark_datasets/oxford_19overlap_evaluation_database_seq56.pickle'
    # opt_oxford.query_file = '/home/cel/data/benchmark_datasets/oxford_19overlap_evaluation_query_seq56.pickle'


    opt_oxford.output_file = opt_oxford.result_folder+'/results.txt'


    # print('opt_oxford', opt_oxford)

    # build model
    if opt_oxford.model.model == 'epn_netvlad':
        from SPConvNets.models.epn_netvlad import EPNNetVLAD
        model = EPNNetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'pointnetvlad_epnnetvlad':
        from SPConvNets.models.pointnet_epn_netvlad import PointNetVLAD_EPNNetVLAD
        model = PointNetVLAD_EPNNetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'pointnetepn_netvlad':
        from SPConvNets.models.pointnet_epn_netvlad import PointNetEPN_NetVLAD
        model = PointNetEPN_NetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'epn_gcn_netvlad':
        from SPConvNets.models.epn_gcn_netvlad import EPN_GCN_NetVLAD
        model = EPN_GCN_NetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'epn_ca_netvlad':
        from SPConvNets.models.epn_gcn_netvlad import EPN_CA_NetVLAD
        model = EPN_CA_NetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'epn_ca_netvlad_select':
        from SPConvNets.models.epn_gcn_netvlad import EPN_CA_NetVLAD_select
        model = EPN_CA_NetVLAD_select(opt_oxford)
    elif opt_oxford.model.model == 'epn_transformer_netvlad':
        from SPConvNets.models.epn_gcn_netvlad import EPN_Transformer_NetVLAD
        model = EPN_Transformer_NetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'epn_conv_netvlad':
        from SPConvNets.models.epn_conv_netvlad import EPNConvNetVLAD
        model = EPNConvNetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'ca_epn_netvlad_select':
        from SPConvNets.models.epn_gcn_netvlad import CA_EPN_NetVLAD_select
        model = CA_EPN_NetVLAD_select(opt_oxford)
    elif opt_oxford.model.model == 'kpconv_netvlad':
        from SPConvNets.models.kpconv_netvlad import KPConvNetVLAD
        model = KPConvNetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'epn_atten_netvlad':
        from SPConvNets.models.epn_gcn_netvlad import EPN_Atten_NetVLAD
        model = EPN_Atten_NetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'atten_epn_netvlad':
        from SPConvNets.models.epn_gcn_netvlad import Atten_EPN_NetVLAD
        model = Atten_EPN_NetVLAD(opt_oxford)
    elif opt_oxford.model.model == 'atten_epn_netvlad_select':
        from SPConvNets.models.epn_gcn_netvlad import Atten_EPN_NetVLAD_select
        model = Atten_EPN_NetVLAD_select(opt_oxford)
        
    # load pretrained file
    if opt_oxford.resume_path.split('.')[1] == 'pth':
        saved_state_dict = torch.load(opt_oxford.resume_path)
    elif opt_oxford.resume_path.split('.')[1] == 'ckpt':
        checkpoint = torch.load(opt_oxford.resume_path)
        saved_state_dict = checkpoint['state_dict']

    model.load_state_dict(saved_state_dict)
    model = nn.DataParallel(model)

    print('average one percent recall', evaluate_model(model, opt_oxford))


def evaluate_model(model, opt):
    DATABASE_SETS = get_sets_dict(opt.database_file)
    QUERY_SETS = get_sets_dict(opt.query_file)

    if not os.path.exists(opt.result_folder):
        os.mkdir(opt.result_folder)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    try:
        DATABASE_VECTORS = np.load(os.path.join(opt.result_folder,'database_vectors.npy'), allow_pickle=True)
        QUERY_VECTORS = np.load(os.path.join(opt.result_folder, 'query_vectors.npy'), allow_pickle=True)
    except:
        # generate descriptors from input point clouds
        print('Generating descriptors from database sets')
        for i in tqdm(range(len(DATABASE_SETS))):
            DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i], opt))

        print('Generating descriptors from query sets')
        for j in tqdm(range(len(QUERY_SETS))):
            QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j], opt))

        np.save(os.path.join(opt.result_folder,'database_vectors.npy'), np.array(DATABASE_VECTORS))
        np.save(os.path.join(opt.result_folder, 'query_vectors.npy'), np.array(QUERY_VECTORS))

    print('Calculating average recall')
    for m in tqdm(range(len(DATABASE_SETS))):
        for n in range(len(QUERY_SETS)):
            if (m == n):
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    print()
    ave_recall = 0
    if count > 0:
        ave_recall = recall / count
    # print(ave_recall)

    # print(similarity)
    average_similarity = np.mean(similarity)
    # print(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    # print(ave_one_percent_recall)

    with open(opt.output_file, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))

    plot_average_recall_curve(ave_recall, ave_one_percent_recall, opt)

    # precision-recall curve
    get_precision_recall_curve(QUERY_SETS, QUERY_VECTORS, DATABASE_VECTORS, opt, ave_one_percent_recall)
    get_f1_recall_curve(opt)

    return ave_one_percent_recall


def plot_average_recall_curve(ave_recall, ave_one_percent_recall, opt):
    index = np.arange(1, 26)
    plt.figure()
    if opt_oxford.model.model == 'kpconv_netvlad':
        plt.plot(index, ave_recall, label='KPConv-NetVLAD')
    elif opt_oxford.model.model == 'epn_ca_netvlad_select' or opt_oxford.model.model == 'epn_ca_netvlad':
        plt.plot(index, ave_recall, label='EPN-CA-NetVLAD')
    else:
        plt.plot(index, ave_recall, label='EPN-NetVLAD, average recall=%.2f' % (ave_one_percent_recall))

    try:
        ave_one_percent_recall_pointnetvlad = None
        with open(os.path.join(opt.pointnetvlad_result_folder, 'results.txt'), "r") as pointnetvlad_result_file:
            ave_one_percent_recall_pointnetvlad = float(pointnetvlad_result_file.readlines()[-1])
        ave_recall_pointnetvlad = ''
        with open(os.path.join(opt.pointnetvlad_result_folder, 'results.txt'), "r") as pointnetvlad_result_file:
            ave_recall_pointnetvlad_temp = pointnetvlad_result_file.readlines()[1:6]
            for i in range(len(ave_recall_pointnetvlad_temp)):
                ave_recall_pointnetvlad_temp[i] = ave_recall_pointnetvlad_temp[i].replace('[', '')
                ave_recall_pointnetvlad_temp[i] = ave_recall_pointnetvlad_temp[i].replace(']', '')
                ave_recall_pointnetvlad_temp[i] = ave_recall_pointnetvlad_temp[i].replace('\n', '')
                ave_recall_pointnetvlad = ave_recall_pointnetvlad + ave_recall_pointnetvlad_temp[i]
            ave_recall_pointnetvlad = np.array(ave_recall_pointnetvlad.split())
            ave_recall_pointnetvlad = np.asarray(ave_recall_pointnetvlad, dtype = float)
        plt.plot(index, ave_recall_pointnetvlad, 'k--', label='PointNetVLAD, average recall=%.2f' % (ave_one_percent_recall_pointnetvlad))
    except:
        print('no pointnetvlad')

    try:
        ave_one_percent_recall_scancontext = None
        with open(os.path.join(opt.scancontext_result_folder, 'results.txt'), "r") as scancontext_result_folder:
            ave_one_percent_recall_scancontext = float(scancontext_result_folder.readlines()[-1])
            ave_recall_scancontext = ''
        with open(os.path.join(opt.scancontext_result_folder, 'results.txt'), "r") as scancontext_result_file:
            ave_recall_scancontext_temp = scancontext_result_file.readlines()[1:6]
            for i in range(len(ave_recall_scancontext_temp)):
                ave_recall_scancontext_temp[i] = ave_recall_scancontext_temp[i].replace('[', '')
                ave_recall_scancontext_temp[i] = ave_recall_scancontext_temp[i].replace(']', '')
                ave_recall_scancontext_temp[i] = ave_recall_scancontext_temp[i].replace('\n', '')
                ave_recall_scancontext = ave_recall_scancontext + ave_recall_scancontext_temp[i]
            ave_recall_scancontext = np.array(ave_recall_scancontext.split())
            ave_recall_scancontext = np.asarray(ave_recall_scancontext, dtype = float)
        plt.plot(index, ave_recall_scancontext, 'm-.', label='Scan Context, average recall=%.2f' % (ave_one_percent_recall_scancontext))
    except:
        print('no scan context')

    try:
        ave_one_percent_recall_m2dp = None
        with open(os.path.join(opt.m2dp_result_folder, 'results.txt'), "r") as m2dp_result_folder:
            ave_one_percent_recall_m2dp = float(m2dp_result_folder.readlines()[-1])
            ave_recall_m2dp = ''
        with open(os.path.join(opt.m2dp_result_folder, 'results.txt'), "r") as m2dp_result_file:
            ave_recall_m2dp_temp = m2dp_result_file.readlines()[1:6]
            for i in range(len(ave_recall_m2dp_temp)):
                ave_recall_m2dp_temp[i] = ave_recall_m2dp_temp[i].replace('[', '')
                ave_recall_m2dp_temp[i] = ave_recall_m2dp_temp[i].replace(']', '')
                ave_recall_m2dp_temp[i] = ave_recall_m2dp_temp[i].replace('\n', '')
                ave_recall_m2dp = ave_recall_m2dp + ave_recall_m2dp_temp[i]
            ave_recall_m2dp = np.array(ave_recall_m2dp.split())
            ave_recall_m2dp = np.asarray(ave_recall_m2dp, dtype = float)
        plt.plot(index, ave_recall_m2dp, 'g:', label='M2DP, average recall=%.2f' % (ave_one_percent_recall_m2dp))
    except:
        print('no M2DP')
    
    plt.title("Average recall @N Curve")
    plt.xlabel('in top N')
    plt.ylabel('Average recall @N')
    plt.xlim(-1,26)
    plt.ylim(0,1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.result_folder, "average_recall_curve.png"))
    print('Average recall curve is saved at:', os.path.join(opt.result_folder, "average_recall_curve.png"))


def get_precision_recall_curve(QUERY_SETS, QUERY_VECTORS, DATABASE_VECTORS, opt, ave_one_percent_recall):
    try:
        precision = np.load(os.path.join(opt.result_folder, 'precision.npy'))
        recall = np.load(os.path.join(opt.result_folder, 'recall.npy'))
    except:
        y_true = []
        y_predicted = []

        for q in range(len(QUERY_SETS)):
            for d in range(len(QUERY_SETS)):
                if (q==d):
                    continue

                database_nbrs = KDTree(DATABASE_VECTORS[d])

                for i in range(len(QUERY_SETS[q])):
                    true_neighbors = QUERY_SETS[q][i][d]
                    if(len(true_neighbors)==0):
                        continue
                    distances, indices = database_nbrs.query(np.array([QUERY_VECTORS[q][i]]))
                    current_y_true = 0
                    current_y_predicted = 0
                    for j in range(len(indices[0])):
                        if indices[0][j] in true_neighbors:
                            # predicted neighbor is correct
                            current_y_true = 1
                        current_y_predicted_temp = np.dot(QUERY_VECTORS[q][i], DATABASE_VECTORS[d][indices[0][j]]) / \
                                                        (np.linalg.norm(QUERY_VECTORS[q][i]) * np.linalg.norm(DATABASE_VECTORS[d][indices[0][j]]))
                        # take prediction similarity that is the highest amoung neighbors
                        if current_y_predicted_temp > current_y_predicted:
                            current_y_predicted = current_y_predicted_temp
                    # loop or not
                    y_true.append(current_y_true)

                    # similarity
                    y_predicted.append(current_y_predicted)
        
        np.set_printoptions(threshold=sys.maxsize)
        # print('y_true', y_true)
        # print('y_predicted', y_predicted)

        precision, recall, thresholds = precision_recall_curve(y_true, y_predicted)

        # print('precision', precision)
        # print('recall', recall)
        # print('thresholds', thresholds)
        np.set_printoptions(threshold=1000)

        np.save(os.path.join(opt.result_folder, 'precision.npy'), np.array(precision))
        np.save(os.path.join(opt.result_folder, 'recall.npy'), np.array(recall))

    # Plot Precision-recall curve
    plt.figure()
    if opt.model.model=='pointnetvlad_epnnetvlad':
        plt.plot(recall, precision, label='PointNetVLAD + EPNNetVLAD')    
    elif opt_oxford.model.model == 'pointnetepn_netvlad':
        plt.plot(recall, precision, label='PointNetEPN-NetVLAD')     
    elif opt_oxford.model.model == 'epn_gcn_netvlad':
        if opt.model.kpconv:
            plt.plot(recall, precision, label='KPConv-GCN-NetVLAD')
        else:
            plt.plot(recall, precision, label='EPN-GCN-NetVLAD')    
    elif opt_oxford.model.model == 'epn_ca_netvlad' or opt_oxford.model.model == 'epn_ca_netvlad_select':
        if opt.model.kpconv:
            plt.plot(recall, precision, label='KPConv-CA-NetVLAD')
        else:
            plt.plot(recall, precision, label='EPN-CA-NetVLAD') 
    elif opt_oxford.model.model == 'epn_transformer_netvlad':
        plt.plot(recall, precision, label='EPN-Transformer-NetVLAD')   
    else:
        if opt.model.kpconv:
            plt.plot(recall, precision, label='KPConv-NetVLAD')
        else:
            plt.plot(recall, precision, label='EPN-NetVLAD')
            # pass
    
    # plot baselines
    try:
        if len(opt.baseline_result_folder) > 0:
            for baseline_folder in opt.baseline_result_folder:
                ave_one_percent_recall_baseline = None
                with open(os.path.join(baseline_folder, 'results.txt'), "r") as baseline_result_file:
                    ave_one_percent_recall_baseline = float(baseline_result_file.readlines()[-1])
                precision_baseline = np.load(os.path.join(baseline_folder, 'precision.npy'))
                recall_baseline = np.load(os.path.join(baseline_folder, 'recall.npy'))
                if baseline_folder[-2] == '0':
                    plt.plot(recall_baseline, precision_baseline, label='EPN-NetVLAD, radius=0.'+baseline_folder[-1]+'0, average recall=%.2f' % (ave_one_percent_recall_baseline))
                elif baseline_folder[-3] == '0':
                    plt.plot(recall_baseline, precision_baseline, label='EPN-NetVLAD, radius=0.'+baseline_folder[-2:]+', average recall=%.2f' % (ave_one_percent_recall_baseline))
    
    except:
        print('no baseline')
    
    try:
        precision_pointnetvlad = np.load(os.path.join(opt.pointnetvlad_result_folder, 'precision.npy'))
        recall_pointnetvlad = np.load(os.path.join(opt.pointnetvlad_result_folder, 'recall.npy'))
        plt.plot(recall_pointnetvlad, precision_pointnetvlad, 'k--', label='PointNetVLAD')
    except:
        print('no pointnetvlad')
    try:
        precision_scancontext = np.load(os.path.join(opt.scancontext_result_folder, 'precision.npy'))
        recall_scancontext = np.load(os.path.join(opt.scancontext_result_folder, 'recall.npy'))
        plt.plot(recall_scancontext, precision_scancontext, 'm-.', label='Scan Context')
    except:
        print('no scan context')
    try:
        precision_m2dp = np.load(os.path.join(opt.m2dp_result_folder, 'precision.npy'))
        recall_m2dp = np.load(os.path.join(opt.m2dp_result_folder, 'recall.npy'))
        plt.plot(recall_m2dp, precision_m2dp, 'g:', label='M2DP')
    except:
        print('no M2DP')
        
    
    plt.title("Precision-recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.result_folder, "precision_recall_oxford.png"))
    print('Precision-recall curve is saved at:', os.path.join(opt.result_folder, "precision_recall_oxford.png"))


def get_f1_recall_curve(opt):
    precision = np.load(os.path.join(opt.result_folder, 'precision.npy'))
    recall = np.load(os.path.join(opt.result_folder, 'recall.npy'))
    f1 = 2 * precision * recall / (precision + recall)
    np.save(os.path.join(opt.result_folder, 'f1.npy'), np.array(f1))

    # Plot F1-recall curve
    plt.figure()
    if opt_oxford.model.model == 'epn_ca_netvlad' or opt_oxford.model.model == 'epn_ca_netvlad_select':
        plt.plot(recall, f1, label='EPN-CA-NetVLAD')
    else:
        plt.plot(recall, f1, label='EPN-NetVLAD')

    try:
        f1_pointnetvlad = np.load(os.path.join(opt.pointnetvlad_result_folder, 'f1.npy'))
        recall_pointnetvlad = np.load(os.path.join(opt.pointnetvlad_result_folder, 'recall.npy'))
        plt.plot(recall_pointnetvlad, f1_pointnetvlad, 'k--', label='PointNetVLAD')
    except:
        print('no pointnetvlad')

    try:
        f1_scancontext = np.load(os.path.join(opt.scancontext_result_folder, 'f1.npy'))
        recall_scancontext = np.load(os.path.join(opt.scancontext_result_folder, 'recall.npy'))
        plt.plot(recall_scancontext, f1_scancontext, 'm-.', label='Scan Context')
    except:
        print('no scan context')

    try:
        f1_m2dp = np.load(os.path.join(opt.m2dp_result_folder, 'f1.npy'))
        recall_m2dp = np.load(os.path.join(opt.m2dp_result_folder, 'recall.npy'))
        plt.plot(recall_m2dp, f1_m2dp, 'g:', label='M2DP')
    except:
        print('no M2DP')

    plt.title("F1-recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('F1 Score')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.result_folder, "f1_recall_oxford.png"))
    print('F1-recall curve is saved at:', os.path.join(opt.result_folder, "f1_recall_oxford.png"))


def get_latent_vectors(model, dict_to_process, opt):

    model.eval()
    is_training = False
    eval_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = opt.batch_size * \
        (1 + opt.pos_per_query + opt.neg_per_query)
    q_output = []
    for q_index in range(len(eval_file_idxs)//batch_num):
        file_indices = eval_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names, opt)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            # feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(opt.device)
            # print('evaluation get_latent_vectors', feed_tensor.shape)
            out, _ = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(eval_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = eval_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names, opt)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            # feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(opt.device)
            # print('evaluation get_latent_vectors edge cases', feed_tensor.shape)
            o1, _ = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    # model.train()
    # print(q_output.shape)
    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    # print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # print('QUERY_SETS[n][i]', QUERY_SETS[n][i])
        true_neighbors = QUERY_SETS[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))
    recall = (np.cumsum(recall)/float(num_evaluated))
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


if __name__ == "__main__":
    evaluate()