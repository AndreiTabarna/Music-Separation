import os
import musdb
import numpy as np
import librosa

# Function to split audio into chunks
def split_into_chunks(audio, chunk_duration, sample_rate):
    chunk_length = int(chunk_duration * sample_rate)
    num_chunks = len(audio) // chunk_length
    chunks = [audio[i * chunk_length:(i + 1) * chunk_length] for i in range(num_chunks)]
    return chunks

# Function to convert audio to spectrogram with enhanced parameters using librosa
def audio_to_spectrogram(audio):
    spec = librosa.stft(audio, n_fft=8192, hop_length=3584)
    return spec

# Function to save spectrograms
def save_spectrograms(specs, save_dir, track_name):
    for i, spec in enumerate(specs):
        np.save(os.path.join(save_dir, f"{track_name}_{i}.npy"), spec)

# Path to musdb dataset
musdb_root = "musdb18"

# Initialize musdb
mus = musdb.DB(root=musdb_root)

# Define parameters
chunk_duration = 7  # seconds
sample_rate = 44100  # Hz

# Iterate over tracks
for track in mus:
    # Split mixture into chunks
    mixture_chunks = split_into_chunks(track.audio.T[0], chunk_duration, sample_rate)
    
    # Convert each chunk to spectrogram with enhanced parameters using librosa
    mixture_specs = [audio_to_spectrogram(chunk) for chunk in mixture_chunks]
    
    # Save mixture spectrograms
    save_dir = os.path.join(musdb_root, "spectrograms", track.subset)
    os.makedirs(save_dir, exist_ok=True)
    save_spectrograms(mixture_specs, save_dir, track.name)
    
    # Split stems into chunks
    for stem_idx, stem_name in enumerate(['drums', 'bass', 'other', 'vocals']):
        if stem_name == 'drums':
            stem_chunks = split_into_chunks(track.targets['drums'].audio.T[0], chunk_duration, sample_rate)
        elif stem_name == 'bass':
            stem_chunks = split_into_chunks(track.targets['bass'].audio.T[0], chunk_duration, sample_rate)
        elif stem_name == 'other':
            stem_chunks = split_into_chunks(track.targets['other'].audio.T[0], chunk_duration, sample_rate)
        elif stem_name == 'vocals':
            stem_chunks = split_into_chunks(track.targets['vocals'].audio.T[0], chunk_duration, sample_rate)
        
        stem_specs = [audio_to_spectrogram(chunk) for chunk in stem_chunks]
        
        # Save stem spectrograms
        save_dir = os.path.join(musdb_root, "spectrograms", track.subset, stem_name)
        os.makedirs(save_dir, exist_ok=True)
        save_spectrograms(stem_specs, save_dir, track.name)

