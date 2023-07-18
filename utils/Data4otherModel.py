import os
import pickle
import numpy as np
import networkx as nx
import torch
import scipy.sparse as spsprs
from utils.preprocessing import normalize_attributes
from collections import Counter
# 用于将数据保存起来，给其他模型使用
class DataReader:
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

        # Spliting the data...
        trn_idx = np.array(range(x.shape[0]), dtype=np.int64)
        val_idx = np.array(range(x.shape[0], allx.shape[0]), dtype=np.int64)
        tst_idx = np.array(tst_idx, dtype=np.int64)
        assert len(trn_idx) == x.shape[0]
        assert len(trn_idx) + len(val_idx) == allx.shape[0]
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

        # Building the feature matrix and the label matrix...
        d, c = x.shape[1], y.shape[1]
        feat_ridx, feat_cidx, feat_data = [], [], []
        allx_coo = allx.tocoo()
        for i, j, v in zip(allx_coo.row, allx_coo.col, allx_coo.data):
            feat_ridx.append(i)
            feat_cidx.append(j)
            feat_data.append(v)
        tx_coo = tx.tocoo()
        for i, j, v in zip(tx_coo.row, tx_coo.col, tx_coo.data):
            feat_ridx.append(tst_idx[i])
            feat_cidx.append(j)
            feat_data.append(v)
        # if data_name.startswith('nell.0'):
        #     isolated = np.sort(np.setdiff1d(range(allx.shape[0], n), tst_idx))
        #     for i, r in enumerate(isolated):
        #         feat_ridx.append(r)
        #         feat_cidx.append(d + i)
        #         feat_data.append(1)
        #     d += len(isolated)
        feat = spsprs.csr_matrix((feat_data, (feat_ridx, feat_cidx)), (n, d))
        targ = np.zeros((n, c), dtype=np.int64)
        targ[trn_idx, :] = y
        targ[val_idx, :] = ally[val_idx, :]
        targ[tst_idx, :] = ty
        targ = dict((i, j) for i, j in zip(*np.where(targ)))
        targ = np.array([targ.get(i, -1) for i in range(n)], dtype=np.int64)
        print('#instance x #feature ~ #class = %d x %d ~ %d' % (n, d, c))
        # feat = normalize_attributes(feat)
        # Storing the data...
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        print(len(trn_idx),len(val_idx),len(tst_idx))
        self.graph, self.feat, self.targ = graph, feat, targ
        # print("graph:", graph, type(graph))
        # print('feat:', feat, type(feat))
        # print('targ:', targ, type(targ), targ.shape, Counter(targ))
        # exit(0)
        # 开始保存
        ## 用于EdgeDisentangle_SSL模型
        ## feat: 2708 * 1433, torch.Tensor
        ## adj: 2708 * 2708 torch.Tensor, layout: torch.sparse_coo
        adj_EdgeDisentangle_SSL = np.array(nx.adjacency_matrix(self.graph).todense())
        b_EdgeDisentangle_SSL = adj_EdgeDisentangle_SSL.sum(1)
        b_EdgeDisentangle_SSL = b_EdgeDisentangle_SSL.reshape(adj_EdgeDisentangle_SSL.shape[0],1)
        adj_EdgeDisentangle_SSL = torch.tensor(adj_EdgeDisentangle_SSL/b_EdgeDisentangle_SSL).to_sparse()
        with open('adj_EdgeDisentangle_SSL_fixed_split.t','wb') as f:
            torch.save(adj_EdgeDisentangle_SSL,f)
        feat_EdgeDisentangle_SSL = torch.tensor(self.feat.toarray())
        with open('feat_EdgeDisentangle_SSL_fixed_split.t','wb') as f:
            torch.save(feat_EdgeDisentangle_SSL,f)
        exit(0)

    def get_split(self):
        # *val_idx* contains unlabeled samples for semi-supervised training.
        # 将划分存储起来
        # np.save('trn_idx.npy', self.trn_idx)
        # np.save('val_idx.npy', self.val_idx)
        # np.save('tst_idx.npy', self.tst_idx)
        # exit(0)
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ
