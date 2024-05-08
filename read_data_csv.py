import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ChestXrayDataSet(torch.utils.data.Dataset):
    """ChestX-ray dataset."""
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            image_list_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        image_names = []
        labels = []

        # Read the CSV file
        data_frame = pd.read_csv(image_list_file)
        data_frame = data_frame[data_frame['Frontal/Lateral'] == 'Frontal']  # Filter to keep only frontal images

        for _, row in data_frame.iterrows():
            img_path = os.path.join(data_dir, row['Path'])
            image_names.append(img_path)

            # Assume that -1 is a missing label, 1 is positive, 0 is negative
            # Adjust according to your needs
            label = []
            for condition in ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']:
                if pd.isna(row[condition]):
                    label.append(0)  # Consider NA as negative or 0 (adjust as needed)
                else:
                    label.append(int(row[condition] > 0))  # Positive if value > 0

            labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)
