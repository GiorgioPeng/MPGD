import torch
import torch.nn.functional as fn
import torch.optim as optim
import torch.autograd
import numpy as np
from utils.NeibSampler import NeibSampler
from Model.Model import CapsuleNet
import time
import scipy.stats

def thsprs_from_spsprs(x):
    try:
        x = x.tocoo().astype(np.float32)
        idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
        val = torch.from_numpy(x.data)
        return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))
    except:
        return torch.FloatTensor(x)
    


class EvalHelper: 
    def __init__(self, dataset, hyperpm, isOGB=False, name='none'):
        use_cuda = torch.cuda.is_available() and not hyperpm.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        self.dev=dev
        graph, feat, targ = dataset.get_graph_feat_targ()
        laplacian = dataset.get_norm_laplacian()
        trn_idx, val_idx, tst_idx = dataset.get_split()
        targ = torch.from_numpy(targ).to(dev)
        feat = thsprs_from_spsprs(feat).to(dev)
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        nfeat, nclass = feat.size(1), int(targ.max() + 1)
        self.laplacian = laplacian.float().to(dev)
        self.graph, self.feat, self.targ = graph, feat, targ
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.neib_sampler = NeibSampler(graph, hyperpm.nbsz).to(dev)  

        model = CapsuleNet(nfeat, nclass, hyperpm).to(dev) 
        optmz = optim.Adam(model.parameters(),
                           lr=hyperpm.lr, weight_decay=hyperpm.reg)
        
        
        self.model, self.optmz = model, optmz
        self.dev = dev
        self.l1 = hyperpm.l1
        self.numberOfCapsule = hyperpm.ncaps
        self.dimension = hyperpm.nhidden
        self.vis_att = None
        self.vis_prob = None
        self.disentangled_embedding = None 
        self.lap = hyperpm.lap
        self.attCoe = hyperpm.att
        self.nclass = nclass

    def run_epoch(self, end='\n'):
        self.model.train()
        self.optmz.zero_grad()
    
        neighbors = self.neib_sampler.sample()
        prob, att, attentionResult, vis_x, self.disentangled_embedding = self.model(self.feat, neighbors, self.laplacian)
        lap_loss = torch.trace(torch.matmul(torch.matmul(att.T, self.laplacian),att))/att.shape[0] 
        l1_loss = torch.sum(torch.abs(att))/att.shape[0]
        loss = fn.nll_loss(prob[self.trn_idx][torch.where(self.targ[self.trn_idx]!=-1)],
            self.targ[self.trn_idx][torch.where(self.targ[self.trn_idx]!=-1)]) + \
            self.attCoe*fn.nll_loss(attentionResult[self.trn_idx][torch.where(self.targ[self.trn_idx]!=-1)],
                self.targ[self.trn_idx][torch.where(self.targ[self.trn_idx]!=-1)])\
                +self.lap*lap_loss + self.l1*l1_loss
        self.vis_att = att.clone()
        self.vis_prob = vis_x
        loss.backward()
        self.optmz.step()
        time1 = time.time()
        print('trn-loss: %.4f' % loss.item(), end=end)
        return loss.item()

    def print_trn_acc(self):
        print('trn-', end='')
        trn_acc = self._print_acc(self.trn_idx, end=' val-')
        val_acc = self._print_acc(self.val_idx)
        return trn_acc, val_acc

    def print_tst_acc(self):
        print('tst-', end='')
        tst_acc = self._print_acc(self.tst_idx)
        return tst_acc

    def _print_acc(self, eval_idx, end='\n'):
        self.model.eval()
    
        prob, att, attentionResult, vis_x, self.disentangled_embedding = self.model(self.feat, self.neib_sampler.nb_all, self.laplacian)
        
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
            
        print('acc: %.4f' % acc, end=end)
        
        return acc

    def get_vis_att(self):
        return self.vis_att
    
    def get_vis_prob(self):
        return self.vis_prob

    def get_nclass(self):
        return self.nclass

    def get_labels(self):
        return self.targ

    def get_disentangled_embedding(self):
        return self.disentangled_embedding