import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from scipy.io import wavfile

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

# Function to reconstruct audio from spectrogram using librosa
def spectrogram_to_audio(spec):
    audio = librosa.istft(spec, hop_length=3584)
    return audio

# Prepare dataset and dataloaders
train_dataset = SpectrogramDataset('musdb18/spectrograms/train')

# Initialize an empty list to store drum masks
drum_masks = []

# Calculate drum masks for all songs in the dataset
for idx in range(len(train_dataset)):
    _, output_tensors = train_dataset[idx]
    drum_spec = output_tensors['other'].numpy().real
    original_spec = np.load(train_dataset.samples[idx][0]).real
    drum_ratio = drum_spec / (original_spec + 1e-8)
    # Calculate the threshold dynamically based on the 95th percentile of drum ratios
    threshold = np.percentile(drum_ratio, 95)
    print(threshold)
    drum_mask = (drum_ratio > (threshold * 0.1)).astype(np.float32)
    if(threshold != 0.0):
        drum_masks.append(drum_mask)

# Aggregate drum masks using median and standard deviation
drum_masks = np.array(drum_masks)
median_mask = np.median(drum_masks, axis=0)
std_mask = np.std(drum_masks, axis=0)

# Create a generalized drum mask using both median and standard deviation
general_drum_mask = median_mask

# Save the general drum mask
np.save('general_drum_mask_1.npy', general_drum_mask)

# Test the general drum mask on one instance
# Load an example song
example_song_idx = 0
input_path, output_paths = train_dataset.samples[example_song_idx]
original_spec = np.load(input_path).real

# Mask the original spectrogram to extract drum frequencies
masked_spec = original_spec * general_drum_mask

# Reconstruct audio from masked spectrogram
reconstructed_audio = spectrogram_to_audio(masked_spec)

# Normalize the audio to a reasonable range
reconstructed_audio /= np.max(np.abs(reconstructed_audio))

# Save reconstructed audio to a temporary file
temp_wav_file = "general_masked_audio_1.wav"
wavfile.write(temp_wav_file, 44100, reconstructed_audio.astype(np.float32))

