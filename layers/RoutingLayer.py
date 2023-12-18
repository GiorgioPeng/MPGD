import torch
import torch.nn as nn
import torch.nn.functional as fn
from sklearn.manifold import TSNE
import math

class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features)).to('cuda')
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features)).to("cuda")
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = fn.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MLPLayer(nn.Module):
    def __init__(self, dim, num_caps, dropout=0):
        super(MLPLayer, self).__init__()
        self.d, self.k = dim, num_caps  
        self.delta_d = int(dim/num_caps)
        mlp_list = nn.ModuleList()
        for i in range(num_caps):
            mlp_list.append(Linear(self.delta_d, self.delta_d, dropout, bias=True))
        self.mlp_list = mlp_list
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps  
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter): 
        dev = x.device
        
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0)  
        d, k, delta_d = self.d, self.k, self.delta_d   
        x = fn.normalize(x.view(n, k, delta_d), dim=2)
        u = torch.zeros_like(x).to(dev)
        for i in range(k):
            u[:,i,:] = self.mlp_list[i](torch.squeeze(x[:,i,:]))
        return u.view(n, d)
    
    
class MeanLayer(nn.Module):
    def __init__(self, dim, num_caps, dropout=0):
        super(MeanLayer, self).__init__()
        self.d, self.k = dim, num_caps  
        self.delta_d = int(dim/num_caps)
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps   
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter): 
        dev = x.device
        
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0) 
        d, k, delta_d = self.d, self.k, self.delta_d  
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)   
        z = z.mean(1)
        return z.view(n, d)


class GCNLayer(nn.Module):
    def __init__(self, dim, num_caps, dropout=0):
        super(GCNLayer, self).__init__()
        self.d, self.k = dim, num_caps  
        self.delta_d = int(dim/num_caps)
        gcn_list = nn.ModuleList()
        for i in range(num_caps):
            gcn_list.append(Linear(self.delta_d, self.delta_d, dropout, bias=True))
        self.gcn_list = gcn_list
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps 
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter): 
        dev = x.device
        
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0)  
        d, k, delta_d = self.d, self.k, self.delta_d   
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)  
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)   
        z = z.sum(1) 
        u = torch.zeros_like(z).to(dev)
        for i in range(k):
            u[:,i,:] = self.gcn_list[i](torch.squeeze(z[:,i,:]))
        return u.view(n, d)

class GATLayer(nn.Module):
    def __init__(self, dim, num_caps, dropout=0, alpha=0.2):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = dim
        self.out_features = dim
        self.alpha = alpha
        self.k = num_caps
        self.W = nn.Parameter(torch.empty(size=(dim, dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x, lap):
        dev = x.device
        n = x.shape[0]
        d, k, delta_d = self.in_features, self.k, self.in_features // self.k  
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)   

        Wh = torch.mm(x, self.W)  
        e = self._prepare_attentional_mechanism_input(Wh)

        h_prime = torch.matmul(lap, Wh)

        return x
    
    def _prepare_attentional_mechanism_input(self, Wh): 
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]) 
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
    
    
class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps):
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps  
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter): 
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0)  
        d, k, delta_d = self.d, self.k, self.d // self.k  
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)   
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)   
        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2)
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d)
