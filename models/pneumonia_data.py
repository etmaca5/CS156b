import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import re

class PneumoniaDataSet(Dataset):

    def __init__(self, data_dir, image_list_file, transform=None):
        self.data_frame = pd.read_csv(image_list_file)
        self.data_frame = self.data_frame[self.data_frame['Frontal/Lateral'] == 'Frontal']
        self.data_frame = self.data_frame.dropna(subset=['Pneumonia'])
        self.data_frame['pid'] = self.data_frame['Path'].apply(self.extract_pid)
        # Filter data to include only specific patient IDs
        self.data_frame = self.data_frame[(self.data_frame['pid'] >= 1) & (self.data_frame['pid'] <= 198)]

        self.data_frame['Path'] = self.data_frame['Path'].apply(lambda x: os.path.join(data_dir, x))
        self.transform = transform

    def extract_pid(self, path):
        try:
            # Assuming the PID format is something like 'pid00123'
            pid = re.search(r'pid(\d+)', path)
            if pid:
                return int(pid.group(1))  # Extracts and converts the number part
            else:
                print("No PID found in path:", path)
                return None  # Return None or some default value or raise an exception
        except Exception as e:
            print("Error processing path:", path, "; Error:", str(e))
            raise


    def __getitem__(self, index):
        image_path = self.data_frame.iloc[index]['Path']
        # print("Attempting to open:", image_path)
        image = Image.open(image_path).convert('RGB')
        # Focus on pneumonia only (assuming 'Pneumonia' is a column in your CSV)
        pneumonia_label = self.data_frame.iloc[index]['Pneumonia']
        label = 0 if pd.isna(pneumonia_label) else int(pneumonia_label > 0)

        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor([label])  # Ensure label is still a tensor

    def __len__(self):
        return len(self.data_frame)
