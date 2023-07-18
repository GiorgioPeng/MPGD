import math
from statistics import variance
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn
import time

class AttentionLayer(nn.Module):
    def __init__(self, representationDimension, numberOfCapsule):
        super(AttentionLayer, self).__init__()
        self.d = representationDimension
        self.k = numberOfCapsule
        self.delta_d = representationDimension // numberOfCapsule

        query = np.zeros((self.delta_d, self.delta_d), dtype=np.float32)
        query = nn.Parameter(torch.from_numpy(query))
        key = np.zeros((self.delta_d, self.delta_d), dtype=np.float32)
        key = nn.Parameter(torch.from_numpy(key))
        self.query, self.key = query, key
        self.reset_parameters()
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)
        self.disentangled_result = None
    
    def getDisentangledEmbedding(self):
        return self.disentangled_result

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.query.size(1))
        self.query.data.uniform_(-stdv, stdv)
        self.key.data.uniform_(-stdv, stdv)    

    def calcCovs(self, centre, sample, w_q, k):
        dev = centre.device
        query = torch.matmul(sample, w_q)
        k = k.unsqueeze(1)
        p_neighbor_matrix = torch.matmul(query, k.transpose(-1, -2)) 
        p_matrix = p_neighbor_matrix.sum(1)  
        p = torch.zeros_like(p_matrix).to(dev)  
        for i in range(p_matrix.shape[0]):
            p[i] = torch.diag(p_matrix[i])
        
        centre = centre.unsqueeze(1)
        sub_graph = torch.cat([sample, centre], dim=1) 
        sub_graph = sub_graph.permute(0, 2, 1, 3)
        sub_graph = fn.normalize(sub_graph, dim=2)  
        middle = torch.median(sub_graph, dim=2).values  
        centre = centre.squeeze(1)
        cov_matrix_diag = torch.bmm(p, (centre - middle)**2)  
        return cov_matrix_diag

    def calcProbabilisticWeight(self, x, centre, cov_matrix, device): 
        d = centre.size(-1)
        diagonal = torch.eye(cov_matrix.shape[-1]).to(device)
        diff = x - centre  
        diff = diff.unsqueeze(2)  
        log_result = torch.log(cov_matrix)
        log_result = torch.where(torch.isnan(log_result), torch.full_like(log_result, 1e-6), log_result)  
        cov_diag_matrix_dets = torch.exp(torch.sum(torch.log(log_result), dim=2))  
        cov_matrix = cov_matrix.unsqueeze(2)  
        cov_diag_matrix = cov_matrix * diagonal 
 
        left = left = 1/(np.sqrt((2*np.pi)**d/2)) * 1/torch.sqrt(cov_diag_matrix_dets + 1e-6)  
        right = torch.exp(-1/2 * torch.matmul(torch.matmul(diff, cov_diag_matrix), torch.transpose(diff, -2, -1)))  
        right = right.squeeze() 
        probability = left * right
        
        probability = fn.normalize(probability, dim=1)  
        probability = torch.where(torch.isnan(probability), torch.full_like(probability, 1e-6), probability)

        return probability

    def forward(self, h, neighbors, max_iter, iterat):
        dev = h.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = h.size(0), neighbors.size(0) // h.size(0)
        d, k, delta_d = self.d, self.k, self.d // self.k
        neighbor_h = h[neighbors].view(n, m, k, delta_d)  

        h = fn.normalize(h.view(n, k, delta_d), dim=2).view(n, d) 
        h = h.view(n, k, delta_d)  
        self.disentangled_result = h.clone().detach()

        q = torch.matmul(h, self.query)
        k = torch.matmul(h, self.key)
        logits = torch.matmul(q, k.transpose(-1, -2))
        attention = torch.diagonal(logits, 0, 1, 2)
        attention = attention.unsqueeze(-1)
        attention = torch.softmax(attention, 1) 
        
        with torch.no_grad():
            cov_matrix = self.calcCovs(h, neighbor_h, self.query, k)
        x = attention * h
        iteraX = torch.sum(x, dim=1, keepdim=True)
        p = torch.zeros((n, self.k, self.delta_d)).to(dev)
        for i in range(iterat):
            with torch.no_grad():
                p = self.calcProbabilisticWeight(iteraX, h, cov_matrix, dev)
            p = p.unsqueeze(-1)
            iteraX = torch.sum(attention * p * h, dim=1, keepdim=True)/\
                     (torch.sum(attention * p, dim=1, keepdim=True) + 1e-9)
        x = iteraX.squeeze(1)
        attention = attention.squeeze(-1)
        x = fn.normalize(x, dim=1) 
        return x, attention
