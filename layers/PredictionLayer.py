import torch.nn as nn


class PredictionLayer(nn.Module):
    def __init__(self, dimension, numberOfClass):
        super(PredictionLayer, self).__init__()
        self.mlp = nn.Linear(dimension, numberOfClass)
    
    def forward(self, x):
        return self.mlp(x)