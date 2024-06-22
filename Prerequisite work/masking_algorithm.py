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

# Prepare dataset and dataloaders
train_dataset = SpectrogramDataset('musdb18/spectrograms/train')

# Load the drum spectrogram
drum_path = train_dataset.samples[0][1]['drums']  # Get the path of the drum spectrogram
print("Drum Spectrogram Path:", drum_path)


# Load the original spectrogram of the song
song_path = train_dataset.samples[0][0]  # Get the input path of the first complete chunk
print("Original Spectrogram Path:", song_path)

# Load the original and drum spectrograms
original_spec = np.load(song_path).real
drum_spec = np.load(drum_path).real

# Calculate the ratio of drum spectrogram to original spectrogram
drum_ratio = drum_spec / (original_spec + 1e-8)  # Adding a small epsilon to avoid division by zero

# Apply a threshold to identify regions predominantly occupied by drums
threshold = 0.5  # You can adjust this threshold as needed
drum_mask = (drum_ratio > threshold).astype(np.float32)

# Mask the original spectrogram to extract drum frequencies
masked_spec = original_spec * drum_mask 

# Reconstruct the given drum spectrogram using the computed frequencies and original spectrogram
extraction = masked_spec / (drum_ratio + 1e-8)  # Reversing the ratio to reconstruct the drum spectrogram

# Save the masked spectrogram
output_path = 'masked_spectrogram.npy'
np.save(output_path, extraction)

# Function to reconstruct audio from spectrogram using librosa
def spectrogram_to_audio(spec):
    audio = librosa.istft(spec, hop_length=3584)
    return audio

# Reconstruct audio from masked spectrogram
reconstructed_audio = spectrogram_to_audio(masked_spec)

# Normalize the audio to a reasonable range
reconstructed_audio /= np.max(np.abs(reconstructed_audio))

# Save reconstructed audio to a temporary file
temp_wav_file = "masked_audio_test.wav"
wavfile.write(temp_wav_file, 44100, reconstructed_audio.astype(np.float32))
