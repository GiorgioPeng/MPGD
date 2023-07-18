import torch
import torch.nn as nn
import torch.nn.functional as fn
from layers.SparseInputLinear import SparseInputLinear
from layers.AttentionLayer import AttentionLayer
from layers.AttentionPredictionLayer import AttentionPredictionLayer
from layers.PredictionLayer import PredictionLayer 
from layers.RoutingLayer import RoutingLayer
from layers.RoutingLayer import GCNLayer
from layers.RoutingLayer import GATLayer
from layers.RoutingLayer import MeanLayer
from layers.RoutingLayer import MLPLayer


class CapsuleNet(nn.Module):   
    def __init__(self, featureDimension, nclass, hyperpm):
        super(CapsuleNet, self).__init__()
        numberOfCapsule, representationDimension = hyperpm.ncaps, hyperpm.nhidden * hyperpm.ncaps   
        self.pca = SparseInputLinear(featureDimension, representationDimension)  
        convolutionList = []  
        for i in range(hyperpm.nlayer):   
            if hyperpm.agg == 'MLP':
                conv = MLPLayer(representationDimension, numberOfCapsule)
            elif hyperpm.agg == 'GCN':
                conv = GCNLayer(representationDimension, numberOfCapsule)
            elif hyperpm.agg == 'GAT':
                conv = GATLayer(representationDimension, numberOfCapsule)
            elif hyperpm.agg == 'MEAN':
                conv = MeanLayer(representationDimension, numberOfCapsule)
            elif hyperpm.agg == 'NR':                
                conv = RoutingLayer(representationDimension, numberOfCapsule)
            else:
                assert hyperpm.agg == 'NR'
            self.add_module('conv_%d' % i, conv)
            convolutionList.append(conv)
        self.agg = hyperpm.agg
        self.convolutionList = convolutionList
        self.PredictionLayer = PredictionLayer(hyperpm.nhidden, nclass)
        self.dropout = hyperpm.dropout
        self.routit = hyperpm.routit  
        self.iterat = hyperpm.iterat
        self.AttentionLayer = AttentionLayer(representationDimension,  numberOfCapsule)
        self.AttentionPredictionLayer = AttentionPredictionLayer(numberOfCapsule, nclass, self.dropout)
        self.numberOfCapsule = numberOfCapsule
        self.dimension = hyperpm.nhidden

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb, lap=None):
        nb = nb.view(-1)  
        x = fn.relu(self.pca(x))  
        for conv in self.convolutionList:
            if self.agg == 'GAT':
                x = self._dropout(fn.relu(conv(x, lap)))
            else:
                x = self._dropout(fn.relu(conv(x, nb, self.routit)))
        x, att = self.AttentionLayer(x, nb, self.routit, self.iterat)
        att = torch.squeeze(att)
        attentionResult = self.AttentionPredictionLayer(att)
        vis_x = x
        x = self.PredictionLayer(x)
        return fn.log_softmax(x, dim=1), att, fn.log_softmax(attentionResult, dim=1), vis_x, self.AttentionLayer.getDisentangledEmbedding()
