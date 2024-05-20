import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from pneumonia_data import PneumoniaDataSet
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import roc_auc_score, mean_squared_error

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

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted_probs = outputs.squeeze().cpu().numpy()
            predictions.extend(predicted_probs)
            ground_truths.extend(labels.cpu().numpy())
    auroc = roc_auc_score(ground_truths, predictions)
    mse = mean_squared_error(ground_truths, predictions)
    print(f"AUROC: {auroc:.4f}")
    print(f"MSE: {mse:.4f}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Setup Datasets

    DATA_DIR = 'Desktop/CS156b'
    IMAGE_LIST_FILE = 'labels/labels.csv'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = PneumoniaDataSet(data_dir=DATA_DIR, image_list_file=IMAGE_LIST_FILE, pid_range=(1, 180), transform=transform)
    test_dataset = PneumoniaDataSet(data_dir=DATA_DIR, image_list_file=IMAGE_LIST_FILE, pid_range=(181, 198), transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Train and Evaluate
    train_model(model, train_loader, criterion, optimizer, device)
    evaluate_model(model, test_loader, device)
