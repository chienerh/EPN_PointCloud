"""
Code Taken from https://github.com/overlappredator/OverlapPredator/blob/770c3063399f08b3836935212ab4c84d355b4704/models/gcn.py
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import torch.utils.checkpoint as checkpoint
import config as cfg

def square_distance(src, dst, normalised = False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if(normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist

def get_graph_feature(coords, feats, k=10):
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    dist = square_distance(coords.transpose(1,2), coords.transpose(1,2))

    idx = dist.topk(k=k+1, dim=-1, largest=False, sorted=True)[1]  #[B, N, K+1], here we ignore the smallest element as it's the query itself  
    idx = idx[:,:,1:]  #[B, N, K]

    idx = idx.unsqueeze(1).repeat(1,C,1,1) #[B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1,index=idx) #[B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1,1,1,k)

    feats_cat = torch.cat((feats, neighbor_feats-feats),dim=1)

    return feats_cat



class SelfAttention(nn.Module):
    def __init__(self,feature_dim,k=10):
        super(SelfAttention, self).__init__() 
        self.conv1 = nn.Conv2d(feature_dim*2, feature_dim, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)
        
        self.conv2 = nn.Conv2d(feature_dim*2, feature_dim * 2, kernel_size=1, bias=False)
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        self.k = k

    def forward(self, coords, features):
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input: 
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  #[B, C, N, 1]

        x1 = get_graph_feature(coords, x0.squeeze(-1), self.k)
        x1 = F.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
        x1 = x1.max(dim=-1,keepdim=True)[0]

        x2 = get_graph_feature(coords, x1.squeeze(-1), self.k)
        x2 = F.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = torch.cat((x0,x1,x2),dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        return x3


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class GCN(nn.Module):
    """
        Alternate between self-attention and cross-attention
        Input:
            coords:     [B, N, 3] -> [B, 3, N]
            feats:      [B, N, C] -> [B, C, N]
        Output:
            feats:      [B, C, N]
        """
    def __init__(self, num_head: int, feature_dim: int, k: int, layer_names: list):
        super().__init__()
        self.layers=[]
        for atten_type in layer_names:
            if atten_type == 'cross':
                self.layers.append(AttentionalPropagation(feature_dim,num_head))
            elif atten_type == 'self':
                self.layers.append(SelfAttention(feature_dim, k))
        self.layers = nn.ModuleList(self.layers)
        self.names = layer_names

    def forward(self, coords, descs):
        if coords.shape[0] >=4:
            # input point cloud with shape [B, 3, N]
            query_pcd, pos_pcd, neg_pcd, otherneg_pcd = torch.split(
                coords.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            # local features with shape [B, D, N]
            x_query, x_pos, x_neg, x_otherneg = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            for layer, name in zip(self.layers, self.names):
                if name == 'cross':
                    x_query = x_query + layer(x_query, x_pos)
                    x_pos = x_pos + layer(x_pos, x_query)
                    x_neg = x_neg + layer(x_neg, x_query)
                    x_otherneg = x_otherneg + layer(x_otherneg, x_query)
                elif name == 'self':
                    x_query = layer(query_pcd, x_query)
                    x_pos = layer(pos_pcd, x_pos)
                    x_neg = layer(neg_pcd, x_neg)
                    x_otherneg = layer(otherneg_pcd, x_otherneg)        
            
            # output conditioned feature with shape [B, C, N]
            x_gcn = torch.cat((x_query, x_pos, x_neg, x_otherneg), 0)
        elif coords.shape[0] == 3:
            # input point cloud with shape [B, 3, N]
            query_pcd, pos_pcd, neg_pcd = torch.split(
                coords.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)
            # local features with shape [B, D, N]
            x_query, x_pos, x_neg = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)
            
            for layer, name in zip(self.layers, self.names):
                if name == 'cross':
                    x_query = x_query + layer(x_query, x_pos)
                    x_pos = x_pos + layer(x_pos, x_query)
                    x_neg = x_neg + layer(x_neg, x_query)
                elif name == 'self':
                    x_query = layer(query_pcd, x_query)
                    x_pos = layer(pos_pcd, x_pos)
                    x_neg = layer(neg_pcd, x_neg)      
            
            # output conditioned feature with shape [B, C, N]
            x_gcn = torch.cat((x_query, x_pos, x_neg), 0)
        elif coords.shape[0] == 2:
            # input point cloud with shape [B, 3, N]
            query_pcd, pos_pcd = torch.split(
                coords.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)
            # local features with shape [B, D, N]
            x_query, x_pos = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)
            
            for layer, name in zip(self.layers, self.names):
                if name == 'cross':
                    x_query = x_query + layer(x_query, x_pos)
                    x_pos = x_pos + layer(x_pos, x_query)
                elif name == 'self':
                    x_query = layer(query_pcd, x_query)
                    x_pos = layer(pos_pcd, x_pos)   
            
            # output conditioned feature with shape [B, C, N]
            x_gcn = torch.cat((x_query, x_pos), 0)
        elif coords.shape[0] == 1:
            # input point cloud with shape [B, 3, N]
            query_pcd = coords.transpose(1,2)
            # local features with shape [B, D, N]
            x_query = descs.transpose(1,2)

            for layer, name in zip(self.layers, self.names):
                if name == 'cross':
                    x_query = x_query + layer(x_query, x_query)
                elif name == 'self':
                    x_query = layer(query_pcd, x_query)
            
            # output conditioned feature with shape [B, C, N]
            x_gcn = x_query
        else:
            print('not able to run network')
            x_gcn = descs.transpose(1,2)

        x_gcn = x_gcn.transpose(1,2)
        return x_gcn


class CrossAttnetion_all(nn.Module):
    """
        Only cross-attention
        Input:
            feats:      [B, N, C] -> [B, C, N]
        Output:
            feats:      [B, C, N]
        """
    def __init__(self, num_head: int, feature_dim: int):
        super().__init__()
        self.ca = AttentionalPropagation(feature_dim, num_head)

    def forward(self, descs):
        if descs.shape[0] >=4:
            # during training, the input for each iteration is a pair of query, pos, neg, and otherneg
            x_query, x_pos, x_neg, x_otherneg = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            x_query = x_query + self.ca(x_query, x_pos) + self.ca(x_query, x_neg) + self.ca(x_query, x_otherneg)
            x_pos = x_pos + self.ca(x_pos, x_query)
            x_neg = x_neg + self.ca(x_neg, x_query)
            x_otherneg = x_otherneg + self.ca(x_otherneg, x_query)    
            
            x_gcn = torch.cat((x_query, x_pos, x_neg, x_otherneg), 0) # [B, C, N]
        elif descs.shape[0] == 3:
            # print('no otherneg')
            x_query, x_pos, x_neg = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)
            
            x_query = x_query + self.ca(x_query, x_pos) + self.ca(x_query, x_neg)
            x_pos = x_pos + self.ca(x_pos, x_query)
            x_neg = x_neg + self.ca(x_neg, x_query) 
            
            x_gcn = torch.cat((x_query, x_pos, x_neg), 0)
        elif descs.shape[0] == 2:
            # print('only pos')
            x_query, x_pos = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)
            
            x_query = x_query + self.ca(x_query, x_pos)
            x_pos = x_pos + self.ca(x_pos, x_query)
            
            x_gcn = torch.cat((x_query, x_pos), 0)
        elif descs.shape[0] == 1:
            # print('only query')
            x_query = descs.transpose(1,2)

            x_query = x_query + self.ca(x_query, x_query)
            
            x_gcn = x_query
        else:
            print('not able to run network')
            x_gcn = descs.transpose(1,2)

        x_gcn = x_gcn.transpose(1,2)
        return x_gcn


class CrossAttnetion_all_weights(nn.Module):
    """
        Only cross-attention
        Input:
            feats:      [B, N, C] -> [B, C, N]
        Output:
            feats:      [B, C, N]
        """
    def __init__(self, num_head: int, feature_dim: int):
        super().__init__()
        self.ca = AttentionalPropagation(feature_dim, num_head)

    def forward(self, descs):
        if descs.shape[0] >=4:
            # during training, the input for each iteration is a pair of query, pos, neg, and otherneg
            x_query, x_pos, x_neg, x_otherneg = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            o1 = self.ca(x_query, x_pos) + self.ca(x_query, x_neg) + self.ca(x_query, x_otherneg)
            o2 = self.ca(x_pos, x_query)
            o3 = self.ca(x_neg, x_query)
            o4 = self.ca(x_otherneg, x_query)    
            
            x_gcn = torch.cat((o1, o2, o3, o4), 0) # [B, C, N]
        elif descs.shape[0] == 3:
            # print('no otherneg')
            x_query, x_pos, x_neg = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)
                        
            o1 = self.ca(x_query, x_pos) + self.ca(x_query, x_neg)
            o2 = self.ca(x_pos, x_query)
            o3 = self.ca(x_neg, x_query)
            
            x_gcn = torch.cat((o1, o2, o3), 0)
        elif descs.shape[0] == 2:
            # print('only pos')
            x_query, x_pos = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)
            
            o1 = self.ca(x_query, x_pos)
            o2 = self.ca(x_pos, x_query)
            
            x_gcn = torch.cat((o1, o2), 0)
        elif descs.shape[0] == 1:
            # print('only query')
            x_query = descs.transpose(1,2)

            o1 = self.ca(x_query, x_query)
            
            x_gcn = o1
        else:
            print('not able to run network')
            x_gcn = descs.transpose(1,2)

        x_gcn = x_gcn.transpose(1,2)
        return x_gcn



class CrossAttnetion(nn.Module):
    """
        Only cross-attention
        Input:
            feats:      [B, N, C] -> [B, C, N]
        Output:
            feats:      [B, C, N]
        """
    def __init__(self, num_head: int, feature_dim: int):
        super().__init__()
        self.ca = AttentionalPropagation(feature_dim, num_head)

    def forward(self, descs):
        if descs.shape[0] >=4:
            # during training, the input for each iteration is a pair of query, pos, neg, and otherneg
            x_query, x_pos, x_neg, x_otherneg = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=0)
            
            x_query = x_query + self.ca(x_query, x_pos)
            x_pos = x_pos + self.ca(x_pos, x_query)
            x_neg = x_neg + self.ca(x_neg, x_query)
            x_otherneg = x_otherneg + self.ca(x_otherneg, x_query)    
            
            # output conditioned feature with shape [B, C, N]
            x_gcn = torch.cat((x_query, x_pos, x_neg, x_otherneg), 0)
        elif descs.shape[0] == 3:
            # local features with shape [B, D, N]
            x_query, x_pos, x_neg = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY], dim=0)
            
            x_query = x_query + self.ca(x_query, x_pos)
            x_pos = x_pos + self.ca(x_pos, x_query)
            x_neg = x_neg + self.ca(x_neg, x_query) 
            
            # output conditioned feature with shape [B, C, N]
            x_gcn = torch.cat((x_query, x_pos, x_neg), 0)
        elif descs.shape[0] == 2:
            # local features with shape [B, D, N]
            x_query, x_pos = torch.split(
                descs.transpose(1,2), [1, cfg.TRAIN_POSITIVES_PER_QUERY], dim=0)
            
            x_query = x_query + self.ca(x_query, x_pos)
            x_pos = x_pos + self.ca(x_pos, x_query)
            
            # output conditioned feature with shape [B, C, N]
            x_gcn = torch.cat((x_query, x_pos), 0)
        elif descs.shape[0] == 1:
            # local features with shape [B, D, N]
            x_query = descs.transpose(1,2)

            x_query = x_query + self.ca(x_query, x_query)
            
            # output conditioned feature with shape [B, C, N]
            x_gcn = x_query
        else:
            print('not able to run network')
            x_gcn = descs.transpose(1,2)

        x_gcn = x_gcn.transpose(1,2)
        return x_gcn