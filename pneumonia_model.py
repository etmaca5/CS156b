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

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted_probs = outputs.squeeze().cpu().numpy()  # Squeeze out unnecessary dimensions if any
            predictions.extend(predicted_probs)
            ground_truths.extend(labels.cpu().numpy())

    # Calculate AUROC
    auroc = roc_auc_score(ground_truths, predictions)
    print(f"AUROC: {auroc:.4f}")
    mse = mean_squared_error(ground_truths, predictions)
    print(f"MSE: {mse:.4f}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121().to(device)

    # Example to setup test_loader
    # You must define DATA_DIR and IMAGE_LIST_FILE or adjust as per your setup
    DATA_DIR = '/Users/etiennecasanova/Desktop/Caltech_classes/CS156b/'
    IMAGE_LIST_FILE = 'labels/labels.csv'
    test_dataset = PneumoniaDataSet(data_dir='', image_list_file=IMAGE_LIST_FILE, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)

    evaluate_model(model, test_loader, device)
