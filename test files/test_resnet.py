import os
import numpy as np
import pandas as pd
from PIL import Image
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Configuration and paths
pathology = 'Pleural Effusion'
using_hpc = 1
model_path = '/groups/CS156b/2024/butters/resnet152_model.pth'
solution_labels_path = '/groups/CS156b/data/student_labels/solution_ids.csv'
images_path = '/groups/CS156b/data'
output_dir = '/groups/CS156b/2024/butters'

class SolutionDataset(Dataset):
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

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

# Load the solution dataset
solution_dataset = SolutionDataset(images_path, solution_labels_path, transform)
solution_dataloader = DataLoader(solution_dataset, batch_size=64, shuffle=False)

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet152()
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Collect predictions for the solution dataset
predictions = []
with torch.no_grad():
    for images, ids in solution_dataloader:
        images = images.to(device)
        outputs = model(images).squeeze()
        for id, output in zip(ids, outputs):
            predictions.append((id.item(), output.item()))

# Create a DataFrame with the predictions
df_output = pd.DataFrame(predictions, columns=['Id', pathology])

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the predictions DataFrame to a CSV file
output_file_path = os.path.join(output_dir, 'predictions_solution_resnet152.csv')
df_output.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
