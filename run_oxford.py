"""
Adapted from https://github.com/FlyingGiraffe/vnn/blob/master/train_cls.py
and https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/train_pointnetvlad.py
"""
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
import torch.nn as nn
from sklearn.neighbors import KDTree, NearestNeighbors
from importlib import import_module
import config as cfg
from SPConvNets.options import opt as opt_oxford
from SPConvNets.utils.loading_pointclouds import *
import SPConvNets.utils.pointnetvlad_loss as PNV_loss

# optional, keep track of training
import wandb


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


'''PARAMETERS'''
opt_oxford.batch_size = cfg.BATCH_NUM_QUERIES
opt_oxford.model.model = cfg.MODEL
opt_oxford.model.flag = 'max'
print('kpconv', opt_oxford.model.kpconv)
# param tuning
opt_oxford.model.search_radius = 0.35 #0.4
opt_oxford.model.initial_radius_ratio = 0.2 #0.2
opt_oxford.model.sampling_ratio = 0.6 #0.8

# EPN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt_oxford.device = device

# NetVLAD parameters
opt_oxford.model.output_num = cfg.LOCAL_FEATURE_DIM # output of EPN
# if cfg.MODEL == 'epn_gcn_netvlad':
#     opt_oxford.num_selected_points = cfg.NUM_SELECTED_POINTS
# else:
opt_oxford.num_selected_points = cfg.NUM_POINTS//(2*int(2*cfg.NUM_POINTS/1024)) #128
opt_oxford.global_feature_dim = cfg.FEATURE_OUTPUT_DIM # output of NetVLAD


# set up log dir and result dir
if not os.path.exists(os.path.join(cfg.LOG_DIR, cfg.EXP_NAME)):
    os.mkdir(os.path.join(cfg.LOG_DIR, cfg.EXP_NAME))
LOG_FOUT = open(os.path.join(cfg.LOG_DIR, cfg.EXP_NAME, 'log_train.txt'), 'w')
LOG_FOUT.write(str(cfg) + '\n')
LOG_FOUT.write(str(opt_oxford) + '\n')


# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)

# bn parameters
cfg.BN_INIT_DECAY = 0.5
cfg.BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(cfg.DECAY_STEP)
cfg.BN_DECAY_CLIP = 0.99

# global parameters
HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []

TOTAL_ITERATIONS = 0


def get_bn_decay(batch):
    bn_momentum = cfg.BN_INIT_DECAY * \
        (cfg.BN_DECAY_DECAY_RATE **
         (batch * cfg.BATCH_NUM_QUERIES // BN_DECAY_DECAY_STEP))
    return min(cfg.BN_DECAY_CLIP, 1 - bn_momentum)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# learning rate halfed every 5 epoch
def get_learning_rate(epoch):
    learning_rate = cfg.BASE_LEARNING_RATE * ((0.9) ** (epoch // 5))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def main():
    global HARD_NEGATIVES, TOTAL_ITERATIONS
    bn_decay = get_bn_decay(0)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

    '''LOGGING'''
    # optional, wandb
    wandb.init(
        project=cfg.EXP_NAME, 
        config={
        "learning_rate": cfg.BASE_LEARNING_RATE,
        "architecture": cfg.MODEL,
        "dataset": cfg.DATASET,
        "loss function": cfg.LOSS_FUNCTION,
        "number of selected points": opt_oxford.num_selected_points,
        "local feature dimension": cfg.LOCAL_FEATURE_DIM,
        "global feature dimension": cfg.FEATURE_OUTPUT_DIM,
        })

    '''DATA LOADING'''
    log_string('Load dataset ...')
        
    '''MODEL LOADING'''
    if cfg.MODEL == 'epn_netvlad':
        from SPConvNets.models.epn_netvlad import EPNNetVLAD
        model = EPNNetVLAD(opt_oxford)
    elif cfg.MODEL == 'pointnetvlad_epnnetvlad':
        from SPConvNets.models.pointnet_epn_netvlad import PointNetVLAD_EPNNetVLAD
        model = PointNetVLAD_EPNNetVLAD(opt_oxford)
    elif cfg.MODEL == 'pointnetepn_netvlad':
        from SPConvNets.models.pointnet_epn_netvlad import PointNetEPN_NetVLAD
        model = PointNetEPN_NetVLAD(opt_oxford)
    elif cfg.MODEL == 'epn_gcn_netvlad':
        from SPConvNets.models.epn_gcn_netvlad import EPN_GCN_NetVLAD
        model = EPN_GCN_NetVLAD(opt_oxford)
    elif cfg.MODEL == 'epn_ca_netvlad':
        from SPConvNets.models.epn_gcn_netvlad import EPN_CA_NetVLAD
        model = EPN_CA_NetVLAD(opt_oxford)
    elif cfg.MODEL == 'epn_ca_netvlad_select':
        from SPConvNets.models.epn_gcn_netvlad import EPN_CA_NetVLAD_select
        model = EPN_CA_NetVLAD_select(opt_oxford)
    elif cfg.MODEL == 'epn_transformer_netvlad':
        from SPConvNets.models.epn_gcn_netvlad import EPN_Transformer_NetVLAD
        model = EPN_Transformer_NetVLAD(opt_oxford)
        
    model = model.to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    '''OPTIMIZER SETUP'''
    learning_rate = get_learning_rate(0)
    if cfg.OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=cfg.MOMENTUM)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        log_string('No optimizer, exiting the code')
        exit(0)
    
    '''PREVIOUS MODEL'''
    if cfg.RESUME:
        resume_filename = os.path.join(cfg.LOG_DIR, cfg.EXP_NAME, 'model.ckpt')
        log_string('Resuming from: '+resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
        
        starting_epoch = checkpoint['epoch']
        TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        log_string('No existing model, starting training from scratch...')
        starting_epoch = 0

    '''LOSS FUNCTION SETUP'''
    if cfg.LOSS_FUNCTION == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    elif cfg.LOSS_FUNCTION == 'triplet':
        loss_function = PNV_loss.triplet_loss_wrapper


    model = nn.DataParallel(model)

    LOG_FOUT.write(cfg.cfg_str())
    LOG_FOUT.write('\n')
    LOG_FOUT.flush()

    '''TRANING'''
    log_string('Start training...')
    for epoch in tqdm(range(starting_epoch, cfg.MAX_EPOCH)):
        train_one_epoch(model, optimizer, loss_function, epoch)
        
    # optional, wandb
    wandb.finish()


        
def train_one_epoch(model, optimizer, loss_function, epoch):      
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    is_training = True
    sampled_neg = 4000
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    num_to_take = 10

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    mean_training_loss = 0
    training_count = 0
    mean_validation_loss = 0
    validation_count = 0

    for i in tqdm(range(len(train_file_idxs)//cfg.BATCH_NUM_QUERIES)):
        batch_keys = train_file_idxs[i *
                                     cfg.BATCH_NUM_QUERIES:(i+1)*cfg.BATCH_NUM_QUERIES]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        for j in range(cfg.BATCH_NUM_QUERIES):
            if (len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < cfg.TRAIN_POSITIVES_PER_QUERY):
                faulty_tuple = True
                break

            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True))
                
            elif (len(HARD_NEGATIVES.keys()) == 0):
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                hard_negs = list(set().union(
                    HARD_NEGATIVES[batch_keys[j]], hard_negs))
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
            if (q_tuples[j][3].shape[0] != cfg.NUM_POINTS):
                no_other_neg = True
                break

        if(faulty_tuple):
            log_string('\n---- Iteration ' + str(i) + '/'+ str(len(train_file_idxs)//cfg.BATCH_NUM_QUERIES) + ' | FAULTY TUPLE -----\n')
            continue

        if(no_other_neg):
            log_string('\n---- Iteration ' + str(i) + '/'+ str(len(train_file_idxs)//cfg.BATCH_NUM_QUERIES) + ' | NO OTHER NEG -----\n')
            continue

        queries = []
        positives = []
        negatives = []
        other_neg = []
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries, dtype=np.float32)
        queries = np.expand_dims(queries, axis=1)
        other_neg = np.array(other_neg, dtype=np.float32)
        other_neg = np.expand_dims(other_neg, axis=1)
        positives = np.array(positives, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)
        if (len(queries.shape) != 4):
            log_string('\n---- Iteration ' + str(i) + '/'+ str(len(train_file_idxs)//cfg.BATCH_NUM_QUERIES) + ' | FAULTY QUERY -----\n')
            continue

        ''' SERIOUSLY TRAINING'''
        model.train()
        optimizer.zero_grad()
        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg)
        loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, cfg.MARGIN_1, cfg.MARGIN_2, use_min=cfg.TRIPLET_USE_BEST_POSITIVES, lazy=cfg.LOSS_LAZY, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)
        loss.backward()
        optimizer.step()
        mean_training_loss+=loss
        training_count+=1

        # log_string('batch loss: %f' % loss)
        TOTAL_ITERATIONS += cfg.BATCH_NUM_QUERIES
        
        # check norm of graadient
        total_norm = 0.
        # for p in model.parameters():
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5


        '''VALIDATION'''
        if (i%200==7):
            test_file_idxs = np.arange(0, len(TEST_QUERIES.keys()))
            np.random.shuffle(test_file_idxs)

            eval_loss=0
            eval_batches=5
            eval_batches_counted=0
            for eval_batch in range(eval_batches):
                eval_keys= test_file_idxs[eval_batch*cfg.BATCH_NUM_QUERIES:(eval_batch+1)*cfg.BATCH_NUM_QUERIES]
                eval_tuples=[]

                faulty_eval_tuple=False
                no_other_neg= False
                for e_tup in range(cfg.BATCH_NUM_QUERIES):
                    if(len(TEST_QUERIES[eval_keys[e_tup]]["positives"])<cfg.TRAIN_POSITIVES_PER_QUERY):
                        faulty_eval_tuple=True
                        break
                    eval_tuples.append(get_query_tuple(TEST_QUERIES[eval_keys[e_tup]],cfg.TRAIN_POSITIVES_PER_QUERY,cfg.TRAIN_NEGATIVES_PER_QUERY, TEST_QUERIES, hard_neg=[], other_neg=True)) 

                    if(eval_tuples[e_tup][3].shape[0]!=cfg.NUM_POINTS):
                        no_other_neg= True
                        break

                if(faulty_eval_tuple):
                    log_string('\n----' + str(i) + ' | FAULTY EVAL TUPLE' + '-----')
                    continue

                if(no_other_neg):
                    log_string('\n----' + str(i) + ' | NO OTHER NEG EVAL' + '-----')
                    continue  

                eval_batches_counted+=1
                eval_queries=[]
                eval_positives=[]
                eval_negatives=[]
                eval_other_neg=[]

                for tup in range(len(eval_tuples)):
                    eval_queries.append(eval_tuples[tup][0])
                    eval_positives.append(eval_tuples[tup][1])
                    eval_negatives.append(eval_tuples[tup][2])
                    eval_other_neg.append(eval_tuples[tup][3])

                eval_queries= np.array(eval_queries)
                eval_queries= np.expand_dims(eval_queries,axis=1)                
                eval_other_neg= np.array(eval_other_neg)
                eval_other_neg= np.expand_dims(eval_other_neg,axis=1)
                eval_positives= np.array(eval_positives)
                eval_negatives= np.array(eval_negatives)

                '''SERIOUSLY VALIDATING'''
                model.eval()
                optimizer.zero_grad()

                output_queries, output_positives, output_negatives, output_other_neg = run_model(
                    model, eval_queries, eval_positives, eval_negatives, eval_other_neg, require_grad=False)
                e_loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, cfg.MARGIN_1, cfg.MARGIN_2, use_min=cfg.TRIPLET_USE_BEST_POSITIVES, lazy=cfg.LOSS_LAZY, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)
                optimizer.step()
                eval_loss+=e_loss
            average_eval_loss= float(eval_loss)/eval_batches_counted

            mean_validation_loss+=average_eval_loss
            validation_count+=1

        
        ''' UPDATE CACHE'''
        if (epoch > 5 and i % (1400 // cfg.BATCH_NUM_QUERIES) == 29):
            TRAINING_LATENT_VECTORS = get_latent_vectors(
                model, TRAINING_QUERIES)
            log_string('\nUpdated cached feature vectors\n')

        if (i % (6000 // cfg.BATCH_NUM_QUERIES) == 101):
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            save_name = os.path.join(cfg.LOG_DIR, cfg.EXP_NAME, cfg.MODEL_FILENAME)
            torch.save({
                'epoch': epoch,
                'iter': TOTAL_ITERATIONS,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
                save_name)
            log_string("Model Saved As " + save_name)

    # loss for each batch
    if training_count > 0:
        mean_training_loss = mean_training_loss / training_count
    if validation_count > 0:
        mean_validation_loss = mean_validation_loss / validation_count
    log_string('training loss: %f' % mean_training_loss)
    log_string('validation loss: %f' % mean_validation_loss)

    # optional wandb
    wandb.log({"Training loss": mean_training_loss, \
                "Validation loss": mean_validation_loss, \
                "Norm of Gradient": total_norm, \
                "Learning Rate": optimizer.param_groups[0]['lr']})


def get_feature_representation(filename, model):
    model.eval()
    queries = load_pc_files([filename])
    # queries = np.expand_dims(queries, axis=1)
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output, _ = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output


def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])

    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=num_to_take)
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs


def get_latent_vectors(model, dict_to_process):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.BATCH_NUM_QUERIES * \
        (1 + cfg.TRAIN_POSITIVES_PER_QUERY + cfg.TRAIN_NEGATIVES_PER_QUERY + 1)
    q_output = []

    model.eval()

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        feed_tensor = torch.from_numpy(queries).float()
        # feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out, _ = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        # queries = np.expand_dims(queries, axis=1)

        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1, _ = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    # print(q_output.shape)
    return q_output


def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, cfg.NUM_POINTS, 3))
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device) # 22, 4096, 3
    # print('feed_tensor', feed_tensor.shape)
    if require_grad:
        output, _ = model(feed_tensor)
    else:
        with torch.no_grad():
            output, _ = model(feed_tensor)
    # print('output', output.shape)
    output = output.view(cfg.BATCH_NUM_QUERIES, -1, cfg.FEATURE_OUTPUT_DIM)
    # print('output reshape', output.shape)
    o1, o2, o3, o4 = torch.split(
        output, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=1)

    return o1, o2, o3, o4

def run_model_split(model, queries, positives, negatives, other_neg, require_grad=True):
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat((queries_tensor, other_neg_tensor), 1)

    output_tensor = []
    for input_tensor in [feed_tensor, positives_tensor, negatives_tensor]:
        input_tensor = input_tensor.view((-1, cfg.NUM_POINTS, 3))
        input_tensor.requires_grad_(require_grad)
        input_tensor = input_tensor.to(device) # 22, 4096, 3
        # print('input_tensor', input_tensor.shape)
        if require_grad:
            output, _ = model(input_tensor)
        else:
            with torch.no_grad():
                output, _ = model(input_tensor)
        
        # print('output', output.shape)
        output = output.view(cfg.BATCH_NUM_QUERIES, -1, cfg.FEATURE_OUTPUT_DIM)
        # print('output reshape', output.shape)
        output_tensor.append(output)

    o1, o4 = torch.split(output_tensor[0], [1, 1], dim=1)
    o2 = output_tensor[1]
    o3 = output_tensor[2]

    return o1, o2, o3, o4


if __name__ == '__main__':
    main()
