import os
import numpy as np
import librosa
from scipy.io import wavfile

# Function to reconstruct audio from spectrogram using librosa
def spectrogram_to_audio(spec):
    audio = librosa.istft(spec, hop_length=3584)
    return audio

# Load the general drum mask
general_drum_mask = np.load('general_drum_mask.npy')

# Path to the newly created folder for the masked spectrograms
spectrogram_folder = 'musdb18/spectrograms/train/lq_drums'

# Get a list of all spectrograms in the folder
spectrogram_files = [f for f in os.listdir(spectrogram_folder) if os.path.isfile(os.path.join(spectrogram_folder, f))]

# Select a spectrogram at random
selected_spectrogram = np.random.choice(spectrogram_files)

# Load the selected spectrogram
selected_spec = np.load(os.path.join(spectrogram_folder, selected_spectrogram))

# Reconstruct audio from masked spectrogram
reconstructed_audio = spectrogram_to_audio(selected_spec)

# Normalize the audio to a reasonable range
reconstructed_audio /= np.max(np.abs(reconstructed_audio))

# Save the reconstructed audio to a music file in the root directory
output_path = os.path.join(os.getcwd(), 'random_masked_audio.wav')
wavfile.write(output_path, 44100, reconstructed_audio.astype(np.float32))

print(f"Random masked audio saved to: {output_path}")

