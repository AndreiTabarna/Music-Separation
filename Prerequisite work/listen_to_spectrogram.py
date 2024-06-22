import os
import random
import numpy as np
import librosa
from scipy.io import wavfile

# Function to reconstruct audio from spectrogram using librosa
def spectrogram_to_audio(spec):
    audio = librosa.istft(spec, hop_length=3584)
    return audio

# Function to load spectrograms
def load_spectrograms(spec_dir, track_name):
    spec_files = [f for f in os.listdir(spec_dir) if f.startswith(track_name)]
    specs = [np.load(os.path.join(spec_dir, f)) for f in spec_files]
    return specs

# Path to spectrograms
spec_dir = "musdb18/spectrograms"

# Choose a random track from the test subset
track_names = os.listdir(os.path.join(spec_dir, "train/bass"))
track_name = random.choice(track_names)

# Load mixture spectrograms
mixture_spec_dir = os.path.join(spec_dir, "train/bass")
mixture_specs = load_spectrograms(mixture_spec_dir, track_name)

# Choose a random chunk to reconstruct and listen to
chunk_idx = random.randint(0, len(mixture_specs) - 1)

# Reconstruct audio from spectrogram using librosa
reconstructed_audio = spectrogram_to_audio(mixture_specs[chunk_idx])

# Normalize the audio to a reasonable range
reconstructed_audio /= np.max(np.abs(reconstructed_audio))

# Save reconstructed audio to a temporary file
temp_wav_file = "temp.wav"
wavfile.write(temp_wav_file, 44100, reconstructed_audio.astype(np.float32))

# Play the reconstructed audio
# Add code here to play the audio using your preferred method (e.g., a media player library)

