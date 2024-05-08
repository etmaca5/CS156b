import os
import numpy as np
import pandas as pd
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

pathology = 'Pleural Effusion'

# TODO: change for running locally
labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv'
labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv'
images_path = '/groups/CS156b/data'

def get_data():
    df_train = pd.read_csv(labels_path_train)[:-1]
    df_test = pd.read_csv(labels_path_test)
    df_train = df_train[df_train['Frontal/Lateral'] == 'Frontal']
    df_test = df_test[df_test['Frontal/Lateral'] == 'Frontal']

    df_train = df_train.dropna(subset=['Pleural Effusion'])
    df_test = df_test.dropna(subset=['Pleural Effusion'])





# turn labels into data







