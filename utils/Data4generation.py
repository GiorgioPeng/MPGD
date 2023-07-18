import os
import pickle
import torch
import numpy as np
import networkx as nx
import scipy.sparse as spsprs
from utils.preprocessing import normalize_attributes
import math

class DataReader4generation:
    def __init__(self, data_name, data_dir):
        # Reading the data...
        # with np.load(data_dir+data_name) as loader:
            # loader = dict(loader)
        feat = np.load('./data_generation/features_multi.npy')
        labels = np.load('./data_generation/labels_multi.npy').astype(np.int)
        adj = np.load('./data_generation/adj_multi.npy')

        # wholey = np.argmax(labels)
        num_classes = labels.max() + 1
        # print(num_classes)
        n = feat.shape[0]
        d = feat.shape[1]
        c = num_classes
        print('#instance x #feature ~ #class = %d x %d ~ %d' % (n, d, c))

        number_train = math.ceil(0.05*n)
        number_val = math.ceil(0.05*n)
        # number_test = math.ceil(0.9*wholex.shape[0])
        per_number_train = math.ceil(number_train/num_classes)-1
        
        indices = []
        for i in range(num_classes):
            temp = torch.tensor(labels)
            index = (temp == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        train_index = torch.cat([i[:per_number_train] for i in indices], dim=0)
        others = torch.cat([i[per_number_train:] for i in indices], dim=0)
        others = others[torch.randperm(others.size(0))]
        val_index = others[:number_val]
        test_index = others[number_val:]
        
        # train_index = torch.cat([i[:20] for i in indices], dim=0)
        # val_index = torch.cat([i[20:50] for i in indices], dim=0)
        # tst_index = torch.cat([i[50:] for i in indices], dim=0)

        trn_idx = np.array(train_index, dtype=np.int64)
        val_idx = np.array(val_index, dtype=np.int64)
        tst_idx = np.array(test_index, dtype=np.int64)
        print('The number of train set: ', trn_idx.shape)
        print('The number of val set: ', val_idx.shape)
        print('The number of test set: ', tst_idx.shape)
        
        graph = nx.from_numpy_array(adj)
        
        # num_same_label = 0
        # num_edges = nx.number_of_edges(graph)
        # edges = nx.edges(graph)
        # for edge in edges:
        #     source_label = labels[edge[0]]
        #     target_label = labels[edge[1]]
        #     if source_label == target_label:
        #         num_same_label += 1
        # homophily_ratio = num_same_label / num_edges
        # print("Homophily ratio: {:.3f}".format(homophily_ratio))
        # exit(0)
        
        feat = normalize_attributes(feat)
        # print(feat.shape)
        norm_laplacian = torch.tensor(nx.normalized_laplacian_matrix(graph).todense())
        # Storing the data...
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.graph, self.feat, self.targ = graph, feat, labels
        self.norm_laplacian = norm_laplacian
        
    def get_split(self):
        # *val_idx* contains unlabeled samples for semi-supervised training.
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ
    
    def get_norm_laplacian(self):
        return self.norm_laplacian