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
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.utils.data
# import torch.utils.data.distributed

labels_of_interest = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
                      'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
pathology = 'Cardiomegaly'
# will determine how we run it
using_hpc = 1
use_subset = True
subset_fraction = 0.01
n_epochs = 2
output_name = 'cnn_cardiomegaly.csv'



if using_hpc == 1:
    labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv'
    labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv'
    images_path = '/groups/CS156b/data'
else: 
    labels_path_train = 'labels/labels.csv'
    labels_path_test = 'labels/test_ids.csv'
    images_path = ''


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
        # 0-2 labeling
        label = self.data_frame.iloc[index][pathology] / 2.0 + 0.5
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
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
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
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

criterion = nn.MSELoss()
learning_rate= 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

training_loss_history = np.zeros(n_epochs)
validation_loss_history = np.zeros(n_epochs)

for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}/{n_epochs}:')
    train_total = 0
    train_correct = 0
    model.train()
    for i, data in enumerate(train_dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        training_loss_history[epoch] += loss.item()
    training_loss_history[epoch] /= len(train_dataloader)
    print(f'Training Loss: {training_loss_history[epoch]:0.4f}')
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(val_dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            validation_loss_history[epoch] += loss.item()
        validation_loss_history[epoch] /= len(val_dataloader)
    print(f'Validation loss: {validation_loss_history[epoch]:0.4f}')



labels_of_interest = [ 'Id', pathology]



predictions = []
with torch.no_grad():
    model.eval()
    for images, ids in test_dataloader:
        images = images.to(device)
        outputs = model(images)

        if outputs.dim() > 1:
            outputs = outputs.squeeze()

        outputs = outputs * 2.0 - 1.0  # Scale and shift to range [-1, 1]

        for output, id_val in zip(outputs, ids):
            predictions.append([id_val.item(), output.item()])

df_output = pd.DataFrame(predictions, columns=['Id', pathology])
print(df_output.head())



if (using_hpc):
    output_dir = '/groups/CS156b/2024/butters'
else:
    output_dir = 'predictions'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file_path = os.path.join(output_dir, output_name)

df_output.to_csv(output_file_path, index=False)

print(f"DataFrame saved to {output_file_path}")
