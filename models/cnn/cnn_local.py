import os
import pandas as pd
from PIL import Image
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Configuration
labels_of_interest = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
                      'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
pathology = 'Cardiomegaly'
images_path = '/Users/jamiekwon/CS156b/CS156b/'  # Update this path
labels_path_train = '/Users/jamiekwon/CS156b/CS156b/labels/labels.csv'  # Update this path
n_epochs = 2
output_name = 'cnn_cardiomegaly.csv'

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, train=True):
        # self.data_frame = pd.read_csv(image_list_file)[:-1]
        # if train:
        #     self.data_frame = self.data_frame[:178]  # First 178 for training
        # else:
        #     self.data_frame = self.data_frame[-20:]  # Last 20 for testing
        
        # self.data_frame = self.data_frame[self.data_frame['Path'].str.endswith('frontal.jpg')]
        # self.data_frame = self.data_frame.dropna(subset=[pathology])
        # self.data_frame['Path'] = self.data_frame['Path'].apply(lambda x: os.path.join(data_dir, x))
        # self.transform = transform
        
        self.data_frame = pd.read_csv(image_list_file)
        self.data_frame = self.data_frame[self.data_frame['Frontal/Lateral'] == 'Frontal']
        self.data_frame = self.data_frame.dropna(subset=[pathology])
        
        if train:
            self.data_frame = self.data_frame[:-20]  # Use all but last 20 for training
        else:
            self.data_frame = self.data_frame[-20:]  # Use last 20 for testing
        
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(images_path, self.data_frame.iloc[index]['Path'])
        image = Image.open(img_path).convert('RGB')
        label = (self.data_frame.iloc[index][pathology] + 1) / 2  # Adjust label from [-1, 1] to [0, 1]
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor([label])

    def __len__(self):
        return len(self.data_frame)

# Model definition
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3, 3)),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.1),
    nn.Conv2d(32, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.1),
    nn.Conv2d(64, 128, 3),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.1),
    nn.Flatten(),
    nn.Linear(128 * 26 * 26, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Transformations
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

# Data loaders
train_dataset = ImageDataset(images_path, labels_path_train, transform=transform, train=True)
test_dataset = ImageDataset(images_path, labels_path_train, transform=transform, train=False)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training loop
for epoch in range(n_epochs):
    model.train()
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

predictions = []
with torch.no_grad():
    model.eval()
    for images, ids in test_dataloader:
        images = images.to(device)
        outputs = model(images)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)  # Fix for batch size of 1
        outputs = outputs * 2.0 - 1.0  # Ensure consistent dimension handling

        for output, id_val in zip(outputs, ids):
            predictions.append([id_val.item(), output.item()])


df_output = pd.DataFrame(predictions, columns=['Id', pathology])
print(df_output.head())

output_dir = 'predictions/attemps'  # Specify the directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file_path = os.path.join(output_dir, output_name)
df_output.to_csv(output_file_path, index=False)
print(f"DataFrame saved to {output_file_path}")
