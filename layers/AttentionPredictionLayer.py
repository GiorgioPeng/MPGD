import torch.nn as nn


class AttentionPredictionLayer(nn.Module):
    def __init__(self, numberOfCapsule, numberOfClass, dropout):
        super(AttentionPredictionLayer, self).__init__()
        self.inputDimension = numberOfCapsule
        self.outputDimension = numberOfClass
        self.pre = nn.Linear(self.inputDimension, self.outputDimension, dropout)

    def forward(self, attention):
        probs = self.pre.forward(attention)

        return probs

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
                self.inputDimension, self.outputDimension
        )
