import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Configuration and paths
pathology = 'Pleural Effusion'
using_hpc = 1
images_path = '/groups/CS156b/data' if using_hpc else ''
labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv' if using_hpc else 'labels/test_ids.csv'
output_dir = '/groups/CS156b/2024/butters' if using_hpc else 'predictions'
ensemble_output_name = 'ensemble_predictions.csv'

# Load the test dataset
class TestDataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        self.data_frame = pd.read_csv(image_list_file)
        self.data_frame['Path'] = self.data_frame['Path'].apply(lambda x: os.path.join(data_dir, x))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.data_frame.iloc[index]['Path']
        image = Image.open(image_path).convert('RGB')
        label = self.data_frame.iloc[index]['Id']
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data_frame)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

X_test = TestDataset(images_path, labels_path_test, transform)
test_dataloader = DataLoader(X_test, batch_size=64, shuffle=False)

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = models.resnet152(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 1)
resnet_model.load_state_dict(torch.load(os.path.join(output_dir, 'resnet152_model.pth')))
resnet_model.to(device)
resnet_model.eval()

densenet_model = models.densenet121(pretrained=False)
num_features = densenet_model.classifier.in_features
densenet_model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1)
)
densenet_model.load_state_dict(torch.load(os.path.join(output_dir, 'densenet121_model.pth')))
densenet_model.to(device)
densenet_model.eval()

cnn_model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3, 3)),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(p=0.1),

    nn.Conv2d(32, 64, kernel_size=(3, 3)),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(p=0.1),

    nn.Conv2d(64, 128, kernel_size=(3, 3)),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(p=0.1),

    nn.Flatten(),
    nn.Linear(128 * 26 * 26, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)
cnn_model.load_state_dict(torch.load(os.path.join(output_dir, 'cnn_model.pth')))
cnn_model.to(device)
cnn_model.eval()

# Ensemble function
def ensemble_predictions(resnet_model, densenet_model, cnn_model, dataloader, weights):
    resnet_weight, densenet_weight, cnn_weight = weights
    predictions = []
    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)
            resnet_outputs = resnet_model(images).squeeze()
            densenet_outputs = densenet_model(images).squeeze()
            cnn_outputs = cnn_model(images).squeeze()

            ensemble_outputs = (resnet_weight * resnet_outputs +
                                densenet_weight * densenet_outputs +
                                cnn_weight * cnn_outputs)

            for id, output in zip(ids, ensemble_outputs):
                predictions.append((id.item(), output.item()))

    return predictions

# Get ensemble predictions
weights = (0.4, 0.4, 0.2)
predictions = ensemble_predictions(resnet_model, densenet_model, cnn_model, test_dataloader, weights)

# Create a DataFrame with the predictions
df_output = pd.DataFrame(predictions, columns=['Id', pathology])

# Save the predictions DataFrame to a CSV file
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file_path = os.path.join(output_dir, ensemble_output_name)
df_output.to_csv(output_file_path, index=False)

print(f"Ensemble predictions saved to {output_file_path}")
