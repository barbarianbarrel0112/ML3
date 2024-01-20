import torch
import torchvision
from torch import nn


class DesNet(nn.Module):
    def __init__(self, num_class):
        super(DesNet, self).__init__()
        self.pretrain = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights)
        self.linear1 = nn.Linear(1000, num_class)

    def forward(self, x):
        x = self.pretrain(x)
        output = self.linear1(x)
        return output

