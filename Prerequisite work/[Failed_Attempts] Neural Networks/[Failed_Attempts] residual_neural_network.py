import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# Define ResNet18 architecture with initialization
class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)


    def forward(self, x):
        x = self.resnet18(x)
        return x

# Define custom loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define directories
input_folder = 'musdb18/spectrograms/train/lq_drums'
output_folder = 'musdb18/spectrograms/train/drums'

# Create dataset
class SpectrogramDataset(Dataset):
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_files = os.listdir(input_folder)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        output_path = os.path.join(self.output_folder, self.input_files[idx])
        input_spec = np.load(input_path).real
        output_spec = np.load(output_path).real
        input_spec = np.stack((input_spec, input_spec, input_spec), axis=0)  # Duplicate input to form [3, w, h]
        output_spec = np.stack((output_spec, output_spec, output_spec), axis=0)  # Duplicate output to form [3, w, h]
        return torch.tensor(input_spec, dtype=torch.float32), torch.tensor(output_spec, dtype=torch.float32)

# Create dataset
dataset = SpectrogramDataset(input_folder, output_folder)

# Initialize ResNet18 model
model = ResNet18(pretrained=True)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Define other training parameters
num_epochs = 10
batch_size = 32

# Create data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    scheduler.step()
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), 'resnet18_drum_enhancer.pth')

