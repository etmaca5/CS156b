import os
import numpy as np
import pandas as pd
from PIL import Image
import re
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.utils.data
# import torch.utils.data.distributed


pathology = 'Pleural Effusion'
# will determine how we run it
using_hpc = 1
use_subset = True
subset_fraction = 0.05
n_epochs = 2
output_name = 'cnn_j_2e.csv'



if using_hpc == 1:
    labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv'
    labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv'
    images_path = '/groups/CS156b/data'
else: 
    labels_path_train = 'labels/labels.csv'
    labels_path_test = 'labels/test_ids.csv'
    images_path = ''


# otherwise 




class ImageDataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        self.data_frame = pd.read_csv(image_list_file)[:-1]
        
        if using_hpc == 0:
            self.data_frame['pid'] = self.data_frame['Path'].apply(self.extract_pid)
            self.data_frame = self.data_frame[(self.data_frame['pid'] >= 1) & (self.data_frame['pid'] <= 198)]
        self.data_frame = self.data_frame[self.data_frame['Path'].str.endswith('frontal.jpg')]
        self.data_frame = self.data_frame.dropna(subset=[pathology])
        if use_subset:
            self.data_frame = self.data_frame.sample(frac=subset_fraction)
        self.data_frame['Path'] = self.data_frame['Path'].apply(lambda x: os.path.join(data_dir, x))
        self.transform = transform

    def extract_pid(self, path):
        try:
            pid = re.search(r'pid(\d+)', path)
            if pid: return int(pid.group(1)) 
            else:
                print("No PID found in path:", path)
                return None  
        except Exception as e:
            print("Error processing path:", path, "; Error:", str(e))
            raise

    def __getitem__(self, index):
        image_path = self.data_frame.iloc[index]['Path']
        image = Image.open(image_path).convert('RGB')
        label = self.data_frame.iloc[index][pathology]
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor([label])  # Ensure label is still a tensor

    def __len__(self):
        return len(self.data_frame)
    
# test set does not include labels
class TestDataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        self.data_frame = pd.read_csv(image_list_file)
        if using_hpc == 0:
            self.data_frame['pid'] = self.data_frame['Path'].apply(self.extract_pid)
            self.data_frame = self.data_frame[(self.data_frame['pid'] >= 1) & (self.data_frame['pid'] <= 198)]
        self.data_frame['Path'] = self.data_frame['Path'].apply(lambda x: os.path.join(data_dir, x))
        self.transform = transform

    def extract_pid(self, path):
        try:
            pid = re.search(r'pid(\d+)', path)
            if pid: return int(pid.group(1)) 
            else:
                print("No PID found in path:", path)
                return None  
        except Exception as e:
            print("Error processing path:", path, "; Error:", str(e))
            raise

    def __getitem__(self, index):
        image_path = self.data_frame.iloc[index]['Path']
        image = Image.open(image_path).convert('RGB')
        label = self.data_frame.iloc[index]['Id']
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data_frame)
    



transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

X_train = ImageDataset(images_path,labels_path_train, transform)
X_test = TestDataset(images_path,labels_path_test, transform)


train_size = int(0.9 * len(X_train))
val_size = len(X_train) - train_size
X_train, X_val = torch.utils.data.random_split(X_train, [train_size, val_size])

batch = 64

train_dataloader = DataLoader(X_train, batch_size=batch, shuffle=True)
val_dataloader = DataLoader(X_val, batch_size=batch, shuffle=True)
test_dataloader = DataLoader(X_test, batch_size=batch, shuffle=False)

print("completed preprocessing")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = nn.Sequential(
#     nn.Conv2d(3, 32, kernel_size=(3, 3)),
#     nn.BatchNorm2d(num_features=32),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Dropout(p=0.1),

#     nn.Conv2d(32, 64, kernel_size=(3, 3)),
#     nn.BatchNorm2d(num_features=64),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Dropout(p=0.1),

#     nn.Conv2d(64, 128, kernel_size=(3, 3)),
#     nn.BatchNorm2d(num_features=128),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Dropout(p=0.1),

#     nn.Flatten(),
#     nn.Linear(128 * 26 * 26, 128),
#     nn.ReLU(),
#     nn.Linear(128, 1),
# )

def initialize_model():
    # Load a pre-trained DenseNet
    model = models.densenet121(pretrained=True)
    # Freeze all the layers in the feature extraction part
    for param in model.features.parameters():
        param.requires_grad = False

    # Number of features in the bottleneck layer
    num_ftrs = model.classifier.in_features
    # Replace the classifier with a new classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        # nn.Tanh()
    )
    return model

model = initialize_model()
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.MSELoss()
learning_rate= 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

training_loss_history = np.zeros(n_epochs)
validation_loss_history = np.zeros(n_epochs)

# for epoch in range(n_epochs):
#     model.train()
#     for images, labels in train_dataloader:
#         images = images.to(device)
#         labels = labels.to(device).view(-1, 1)  # Ensure labels are the correct shape
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         # Additional logging or accuracy computation here

for epoch in range(n_epochs):
    model.train()
    for images, labels in train_dataloader:
        images = images.to(device)
        # Adjust labels from [0, 1] to [-1, 1]
        labels = labels.to(device).view(-1, 1) * 2 - 1  # Scale labels to -1 and 1
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            # Additional logging or accuracy computation here


labels_of_interest = [ 'Id', pathology]

# predictions = []
# with torch.no_grad():
#     model.eval()
#     for images, ids in test_dataloader:
#         images = images.to(device)
#         outputs = model(images).squeeze()  # Adjust assuming the output is a single probability
#         predicted_labels = (outputs > 0.5).long()  # Thresholding at 0.5
#         predictions.extend(zip(ids.cpu().numpy(), predicted_labels.cpu().numpy()))


predictions = []
with torch.no_grad():
    model.eval()
    for images, ids in test_dataloader:
        images = images.to(device)
        outputs = model(images).squeeze()
        predictions.extend(zip(ids.cpu().numpy(), outputs.cpu().numpy()))


df_output = pd.DataFrame(predictions, columns=['Id', pathology])
df_output.head()




if (using_hpc):
    output_dir = '/groups/CS156b/2024/butters'
else:
    output_dir = 'predictions'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))

output_file_path = os.path.join(output_dir, output_name)

df_output.to_csv(output_file_path, index=False)

print(f"DataFrame saved to {output_file_path}")
