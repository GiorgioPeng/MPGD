import os
import pickle
import torch
import numpy as np
import networkx as nx
import scipy.sparse as spsprs
import math
from .preprocessing import normalize_attributes

class RandomDataReader:
    def __init__(self, data_name, data_dir):
        # Reading the data...
        tmp = []
        prefix = os.path.join(data_dir, 'ind.%s.' % data_name)
        for suffix in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
            with open(prefix + suffix, 'rb') as fin:
                tmp.append(pickle.load(fin, encoding='latin1'))
        x, y, tx, ty, allx, ally, graph = tmp
        with open(prefix + 'test.index') as fin:
            tst_idx = [int(i) for i in fin.read().split()] 
        assert np.sum(x != allx[:x.shape[0], :]) == 0
        assert np.sum(y != ally[:y.shape[0], :]) == 0

        tst_idx_range = np.sort(tst_idx)
        if data_name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(tst_idx), max(tst_idx) + 1)
            tx_extended = spsprs.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[tst_idx_range - min(tst_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]), dtype=np.int32)
            ty_extended[tst_idx_range - min(tst_idx_range), :] = ty
            ty = ty_extended
         
        wholex = spsprs.vstack((allx, tx)).tolil()
        wholex[tst_idx, :] = wholex[tst_idx_range, :]
         
        wholey = np.vstack((ally, ty))
        wholey[tst_idx, :] = wholey[tst_idx_range, :]
        wholey_ = np.argmax(wholey, 1)
        num_classes = wholey_.max()+1 
        number_train = math.ceil(0.05*wholex.shape[0])
        number_val = math.ceil(0.05*wholex.shape[0]) 
        per_number_train = math.ceil(number_train/num_classes)-1
        
        indices = []
        for i in range(num_classes):
            temp = torch.tensor(wholey_)
            index = (temp == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))] 
            indices.append(index)
        train_index = torch.cat([i[:per_number_train] for i in indices], dim=0)
        others = torch.cat([i[per_number_train:] for i in indices], dim=0)
         
        others = others[torch.randperm(others.size(0))] 
        
        val_index = others[:number_val]
        test_index = others[number_val:] 

        trn_idx = np.array(train_index, dtype=np.int64)
        val_idx = np.array(val_index, dtype=np.int64)
        tst_idx = np.array(test_index, dtype=np.int64)
        print('The number of train set: ', trn_idx.shape)
        print('The number of val set: ', val_idx.shape)
        print('The number of test set: ', tst_idx.shape) 
        assert len(tst_idx) > 0
        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0

        # Building the graph...
        graph = nx.from_dict_of_lists(graph)
        assert min(graph.nodes()) == 0
        n = graph.number_of_nodes()
        assert max(graph.nodes()) + 1 == n
        n = max(n, np.max(tst_idx) + 1)
        for u in range(n):
            graph.add_node(u)
        assert graph.number_of_nodes() == n
        assert not graph.is_directed()
 
        norm_laplacian = torch.tensor(nx.normalized_laplacian_matrix(graph).todense())
        # Building the feature matrix and the label matrix...
        d, c = x.shape[1], y.shape[1]
        feat_ridx, feat_cidx, feat_data = [], [], [] 
        wholex_coo = wholex.tocoo()
        for i, j, v in zip(wholex_coo.row, wholex_coo.col, wholex_coo.data):
            feat_ridx.append(i)
            feat_cidx.append(j)
            feat_data.append(v)

        feat = spsprs.csr_matrix((feat_data, (feat_ridx, feat_cidx)), (n, d))
        feat = normalize_attributes(feat)
        targ = np.zeros((n, c), dtype=np.int64)
        
        targ[trn_idx, :] = wholey[trn_idx, :]
        targ[val_idx, :] = wholey[val_idx, :]
        targ[tst_idx, :] = wholey[tst_idx, :]
        targ = dict((i, j) for i, j in zip(*np.where(targ)))
        targ = np.array([targ.get(i, -1) for i in range(n)], dtype=np.int64)
        print('#instance x #feature ~ #class = %d x %d ~ %d, with Random Split' % (n, d, c))

        # Storing the data...
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.graph, self.feat, self.targ = graph, feat, targ 
        self.norm_laplacian = norm_laplacian

    def get_split(self): 
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ

    def get_norm_laplacian(self):
        return self.norm_laplacian