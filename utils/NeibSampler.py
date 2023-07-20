import torch
import numpy as np
import math


class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False): 
        n = graph.number_of_nodes()   
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:   
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)  
            nb = nb_all[:, 1:]   
        else:   
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v)) 
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))   
                nb[v] = torch.LongTensor(nb_v)   
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self): 
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all  
        nb_size = nb.size(1)   
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)  
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all
