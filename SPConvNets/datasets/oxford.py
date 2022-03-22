import numpy as np
import trimesh
import os
import glob
import scipy.io as sio
import torch
import torch.utils.data as data
import vgtk.pc as pctk
import vgtk.point3d as p3dtk
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np, label_relative_rotation_np
from scipy.spatial.transform import Rotation as sciR
import random
import pickle        

class Dataloader_Oxford(data.Dataset):
    def __init__(self, opt, mode=None):
        super(Dataloader_Oxford, self).__init__()
        self.opt = opt

        # 'train' or 'eval'
        self.mode = opt.mode if mode is None else mode

        # Load data dictionaries from the pickle files
        if self.mode == 'train':
            self.pickle_file = self.opt.train_file
        elif self.mode == 'eval':
            self.pickle_file = self.opt.val_file

        self.raw_queries = self.get_queries_dict(self.pickle_file)

        # only keep training queries that have enough positive data
        # TODO: only keep queries that have other neg
        self.queries = {}
        self.queries_key = []
        for i in range(len(self.raw_queries.keys())):
            if (len(self.raw_queries[i]["positives"]) >= self.opt.pos_per_query):
                self.queries[i] = self.raw_queries[i]
                self.queries_key.append(i)

        print("[Dataloader] : Training dataset size:", len(self.queries.keys()))

        if self.opt.no_augmentation:
            print("[Dataloader]: USING ALIGNED OXFORD LOADER!")
        else:
            print("[Dataloader]: USING ROTATED OXFORD LOADER!")

    def load_pc_file(self, filename):
        #returns Nx3 matrix
        pc=np.fromfile(os.path.join(self.opt.dataset_path,filename), dtype=np.float64)

        if(pc.shape[0]!= self.opt.num_points*3):
            print("Error in pointcloud shape")
            return np.array([])

        pc=np.reshape(pc,(pc.shape[0]//3,3))
        return pc

    def load_pc_files(self, filenames):
        pcs=[]
        for filename in filenames:
            #print(filename)
            pc=self.load_pc_file(filename)
            if(pc.shape[0]!=self.opt.num_points):
                continue
            pcs.append(pc)
        pcs=np.array(pcs)
        return pcs

    def get_queries_dict(self, filename):
        #key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
        with open(filename, 'rb') as handle:
            queries = pickle.load(handle)
            print("Queries Loaded.")
            return queries


    def get_query_tuple(self, dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
        #get query tuple for dictionary entry
        #return list [query,positives,negatives]

        query=self.load_pc_file(dict_value["query"]) #Nx3

        random.shuffle(dict_value["positives"])
        pos_files=[]

        for i in range(num_pos):
            pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
        positives=self.load_pc_files(pos_files)

        neg_files=[]
        neg_indices=[]
        if(len(hard_neg)==0):
            random.shuffle(dict_value["negatives"])	
            for i in range(num_neg):
                neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
                neg_indices.append(dict_value["negatives"][i])

        else:
            random.shuffle(dict_value["negatives"])
            for i in hard_neg:
                neg_files.append(QUERY_DICT[i]["query"])
                neg_indices.append(i)
            j=0
            while(len(neg_files)<num_neg):

                if not dict_value["negatives"][j] in hard_neg:
                    neg_files.append(QUERY_DICT[dict_value["negatives"][j]]["query"])
                    neg_indices.append(dict_value["negatives"][j])
                j+=1
        
        negatives=self.load_pc_files(neg_files)

        if(other_neg==False):
            return [query,positives,negatives]
        #For Quadruplet Loss
        else:
            #get neighbors of negatives and query
            neighbors=[]
            for pos in dict_value["positives"]:
                neighbors.append(pos)
            for neg in neg_indices:
                for pos in QUERY_DICT[neg]["positives"]:
                    neighbors.append(pos)
            possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
            random.shuffle(possible_negs)

            if(len(possible_negs)==0):
                return [query, positives, negatives, np.array([])]		

            neg2= self.load_pc_file(QUERY_DICT[possible_negs[0]]["query"])

            return [query,positives,negatives,neg2]


    def __len__(self):
        return len(self.queries.keys())

    def __getitem__(self, index):
        current_key = self.queries_key[index]
        anchor_pcd, positive_pcd, negative_pcd, other_neg_pcd = self.get_query_tuple(self.queries[current_key], \
                                                                                self.opt.pos_per_query, \
                                                                                self.opt.neg_per_query, \
                                                                                self.raw_queries, \
                                                                                other_neg=True)
        
        # reshape to have same sizes
        anchor_pcd = anchor_pcd.reshape(1, anchor_pcd.shape[0], anchor_pcd.shape[1])
        other_neg_pcd = other_neg_pcd.reshape(1, other_neg_pcd.shape[0], other_neg_pcd.shape[1])        

        # TODO: rotate points if requires augmentation
        # for if not self.opt.no_augmentation:
            # pctk.rotate_point_cloud(pc)
        
        return {'anchor': anchor_pcd,
                'positive': positive_pcd,
                'negative': negative_pcd,
                'other_neg': other_neg_pcd,}