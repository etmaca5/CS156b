import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from chexnet_read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from pneumonia_data import PneumoniaDataSet
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import mean_squared_error





# actual model is here
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x