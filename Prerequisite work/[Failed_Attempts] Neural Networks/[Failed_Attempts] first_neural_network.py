import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import torch.nn.functional as F

def SDR(estimates, references, eps=1e-7, min_sdr=-30):
    """
    Signal-to-Distortion Ratio (SDR) metric calculation.
    """
    numerator = torch.sum(references * estimates, dim=(-3, -2, -1))
    denominator = torch.sum(estimates ** 2, dim=(-3, -2, -1)) + eps
    alpha = numerator / denominator
    
    # Handle cases where denominator is close to zero
    sdr = 10 * torch.log10(torch.clamp(alpha, min=eps))
    
    # Replace NaN values with a predefined minimum value
    sdr = torch.where(torch.isnan(sdr), torch.tensor(min_sdr, dtype=sdr.dtype, device=sdr.device), sdr)
    
    return torch.mean(sdr)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # concatenating along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, gate_channels, in_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.gate_channels = gate_channels

        self.theta = nn.Conv2d(in_channels, gate_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.phi = nn.Conv2d(in_channels, gate_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(gate_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        phi = F.interpolate(phi, size=theta.size()[2:], mode='bilinear', align_corners=True)
        psi = self.sigmoid(theta + phi)
        out = psi.expand_as(x) * x
        return self.conv(out)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.att1 = AttentionGate(256, 256, 256)
        self.up2 = Up(512, 128, bilinear)
        self.att2 = AttentionGate(128, 128, 128)
        self.up3 = Up(256, 64, bilinear)
        self.att3 = AttentionGate(64, 64, 64)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.att1(x)
        x = self.up2(x, x3)
        x = self.att2(x)
        x = self.up3(x, x2)
        x = self.att3(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Define a custom dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        for chunk_file in os.listdir(os.path.join(root_dir)):
            chunk_name, ext = os.path.splitext(chunk_file)
            if ext:  # Check if the file has an extension (not a directory)
                input_path = os.path.join(root_dir, chunk_file)
                output_paths = {
                    stem_type: os.path.join(root_dir, stem_type, chunk_file) for stem_type in ["drums", "bass", "other", "vocals"]
                }
                self.samples.append((input_path, output_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, output_paths = self.samples[idx]
        input_spec = np.load(input_path).real  # Ensure real values
        output_specs = {stem_type: np.load(output_paths[stem_type]).real for stem_type in output_paths}  # Ensure real values
        input_tensor = torch.tensor(input_spec, dtype=torch.float32).unsqueeze(0)
        output_tensors = {stem_type: torch.tensor(output_specs[stem_type], dtype=torch.float32) for stem_type in output_specs}
        return input_tensor, output_tensors

# Prepare dataset and dataloaders
train_dataset = SpectrogramDataset('musdb18/spectrograms/train')
test_dataset = SpectrogramDataset('musdb18/spectrograms/test')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)

# Initialize model, loss function, optimizer, and learning rate scheduler
model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Apply weight decay for regularization
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)  # Reduce LR on plateau

# Gradient clipping
clip_value = 1.0
for p in model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

# Training loop
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to device

# Check if there is a saved model, if not, create a new one
saved_model_path = 'saved_model.pt'
if os.path.exists(saved_model_path):
    print("Loading existing model!\n")
    model.load_state_dict(torch.load(saved_model_path))
else:
    torch.save(model.state_dict(), saved_model_path)

# Training loop with weighted SDR and MSE loss
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(train_dataloader)
    accumulated_batches = 201  # Accumulate gradients over 4 batches
    loss_history = []  # List to store losses of last 4 batches
    for i, batch in enumerate(train_dataloader):
        if i == total_batches - 1:  # Skip the last batch iteration
            break
        optimizer.zero_grad()
        inputs, outputs_dict = batch
        inputs = inputs.to(device)
        # Move each output to device separately
        outputs = {stem_type: output_tensor.to(device) for stem_type, output_tensor in outputs_dict.items()}
        out_drums, out_bass, out_other, out_vocals = model(inputs)

        # Compute SDR loss along with MSE loss
        sdr_loss = SDR(out_drums, outputs["drums"]) + SDR(out_bass, outputs["bass"]) + SDR(out_other, outputs["other"]) + SDR(out_vocals, outputs["vocals"])
        mse_loss = criterion(out_drums, outputs["drums"]) + criterion(out_bass, outputs["bass"]) + criterion(out_other, outputs["other"]) + criterion(out_vocals, outputs["vocals"])

        # Weighted combination of SDR and MSE loss
        total_loss =0.8 * mse_loss - 0.2 * sdr_loss

        total_loss /= accumulated_batches  # Divide the loss by the number of accumulated batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{total_batches-1}], Loss: {total_loss.item():.7f}")
        total_loss.backward()
        loss_history.append(total_loss.item())
        
        if (i + 1) % accumulated_batches == 0 or i == total_batches - 1:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Total Loss of Last Batches: {sum(loss_history):.7f}")  # Print total loss of last batches
            loss_history = []  # Clear the loss history after printing

        running_loss += total_loss.item()
        

        # Save the model at each batch iteration
        torch.save(model.state_dict(), saved_model_path)

    epoch_loss = running_loss / (total_batches - 1) * accumulated_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.7f}")

    # Update learning rate scheduler
    scheduler.step(epoch_loss)


# Testing loop
model.eval()
test_loss = 0.0
total_batches = len(test_dataloader)
with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        if i == total_batches - 1:  # Skip the last batch iteration
            break
        inputs, outputs_dict = batch
        inputs = inputs.to(device)
        # Move each output to device separately
        outputs = {stem_type: output_tensor.squeeze(1).to(device) for stem_type, output_tensor in outputs_dict.items()}  # Remove the extra dimension
        out_drums, out_bass, out_other, out_vocals = model(inputs)
        loss = criterion(out_drums, outputs["drums"]) + criterion(out_bass, outputs["bass"]) + criterion(out_other, outputs["other"]) + criterion(out_vocals, outputs["vocals"])
        test_loss += loss.item()
        print(f"Testing Batch [{i+1}/{total_batches-1}], Loss: {loss.item():.4f}")
    avg_test_loss = test_loss / (total_batches - 1)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
