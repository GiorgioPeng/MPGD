import os
import pickle
import torch
import numpy as np
import networkx as nx
import scipy.sparse as spsprs
from utils.preprocessing import normalize_attributes
import math

class DataReader4npz:
    def __init__(self, data_name, data_dir):
        if data_name[:-4] in ('squirrel', 'crocodile', 'chameleon', 'squirrel_filtered', 'chameleon_filtered'):
            dataset = np.load(data_dir+data_name)
            if data_name[:-4] in ('squirrel_filtered', 'chameleon_filtered'):
                features = dataset['node_features']
                labels = dataset['node_labels']
                edges = dataset['edges']
            else:
                labels = dataset['label'] # numpy array
                edges = dataset['edges'] # numpy array
                features = dataset['features'] # numpy array
            data = np.ones(edges.shape[0])
            adj = spsprs.csr_matrix((data, (edges[:,0], edges[:,1])), shape=(features.shape[0], features.shape[0]))
            
            n = features.shape[0]
            y = labels.max()+1
            if data_name[:-4] in ('squirrel_filtered', 'chameleon_filtered'):
                ntrain_per_class = int(n*0.5/y)
                number_val = int(n*0.25)
                number_tst = int(n*0.25)
            else:
                ntrain_per_class = int(n*0.1/y)
                number_val = int(n*0.1)
                number_tst = int(n*0.8)
            indices = []
            for i in range(y):
                temp = torch.tensor(labels)
                index = (temp == i).nonzero().view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)
            train_index = torch.cat([i[:ntrain_per_class] for i in indices], dim=0)
            others = torch.cat([i[ntrain_per_class:] for i in indices], dim=0)
            others = others[torch.randperm(others.size(0))]
            val_index = others[:number_val]
            test_index = others[number_val:]
            
            # idx_split_args = {'ntrain_per_class': 20, 'nstopping': 1000, 'nknown': 1500, 'seed': seed}
            # idx_train, idx_val, idx_test = gen_splits(labels, idx_split_args, test=True)
            # features = torch.tensor(features)
            feat = normalize_attributes(features)
            
            trn_idx = np.array(train_index, dtype=np.int64)
            val_idx = np.array(val_index, dtype=np.int64)
            tst_idx = np.array(test_index, dtype=np.int64)
            print('The number of train set: ', trn_idx.shape)
            print('The number of val set: ', val_idx.shape)
            print('The number of test set: ', tst_idx.shape)
            
            graph = nx.from_scipy_sparse_matrix(adj)
            
            # # calculate the homophily ratio
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
            
            # print(feat.shape)
            norm_laplacian = torch.tensor(nx.normalized_laplacian_matrix(graph).todense())
            # Storing the data...
            self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
            self.graph, self.feat, self.targ = graph, feat, labels
            self.norm_laplacian = norm_laplacian 
        elif data_name[:-4] in ('coauthor_cs', 'ms_academic'):
            dataset = np.load(data_dir+data_name)
            adj_data = dataset['adj_data']
            adj_indices = dataset['adj_indices']
            adj_indptr = dataset['adj_indptr']
            adj_shape = dataset['adj_shape']
            attr_data = dataset['attr_data']
            attr_indices = dataset['attr_indices']
            attr_indptr = dataset['attr_indptr']
            attr_shape = dataset['attr_shape']
            labels = dataset['labels']
            adj = spsprs.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
            features = spsprs.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)
            features = normalize_attributes(features)
            feat = torch.FloatTensor(np.array(features.todense()))
            
            n = feat.shape[0]
            y = labels.max()+1
            ntrain_per_class = int(n*0.1/y)
            number_val = int(n*0.1)
            number_tst = int(n*0.8)
            indices = []
            for i in range(y):
                temp = torch.tensor(labels)
                index = (temp == i).nonzero().view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)
            train_index = torch.cat([i[:ntrain_per_class] for i in indices], dim=0)
            others = torch.cat([i[ntrain_per_class:] for i in indices], dim=0)
            others = others[torch.randperm(others.size(0))]
            val_index = others[:number_val]
            test_index = others[number_val:]
            
            # idx_split_args = {'ntrain_per_class': 20, 'nstopping': 1000, 'nknown': 1500, 'seed': seed}
            # idx_train, idx_val, idx_test = gen_splits(labels, idx_split_args, test=True)
            # features = torch.tensor(features)
            
            trn_idx = np.array(train_index, dtype=np.int64)
            val_idx = np.array(val_index, dtype=np.int64)
            tst_idx = np.array(test_index, dtype=np.int64)
            print('The number of train set: ', trn_idx.shape)
            print('The number of val set: ', val_idx.shape)
            print('The number of test set: ', tst_idx.shape)
            
            graph = nx.from_scipy_sparse_matrix(adj)
            # print(feat.shape)
            norm_laplacian = torch.tensor(nx.normalized_laplacian_matrix(graph).todense())
            # Storing the data...
            self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
            self.graph, self.feat, self.targ = graph, feat, labels
            self.norm_laplacian = norm_laplacian
        else:
            # Reading the data...
            with np.load(data_dir+data_name) as loader:
                loader = dict(loader)
                adj_matrix = spsprs.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                        shape=loader['adj_shape'])

                if 'attr_data' in loader:
                    # Attributes are stored as a sparse CSR matrix
                    feat = spsprs.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                                shape=loader['attr_shape'])
                elif 'attr_matrix' in loader:
                    # Attributes are stored as a (dense) np.ndarray
                    feat = loader['attr_matrix']
                else:
                    feat = None

                if 'labels_data' in loader:
                    # Labels are stored as a CSR matrix
                    labels = spsprs.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                        shape=loader['labels_shape'])
                elif 'labels' in loader:
                    # Labels are stored as a numpy array
                    labels = loader['labels']
                else:
                    labels = None

            # wholey = np.argmax(labels)
            num_classes = labels.max() + 1
            
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
            
            graph = nx.from_scipy_sparse_matrix(adj_matrix)
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