import os
import shutil
import librosa
from scipy.io import wavfile
import numpy as np

def spectrogram_to_audio(spec):
    audio = librosa.istft(spec, hop_length=3584)
    return audio

def convert_npy_to_wav(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for subfolder in dirs:
            subfolder_path = os.path.join(root, subfolder)
            output_subfolder = subfolder_path.replace(input_folder, output_folder)
            os.makedirs(output_subfolder, exist_ok=True)
            for file in os.listdir(subfolder_path):
                if file.endswith('.npy'):
                    npy_path = os.path.join(subfolder_path, file)
                    npy_data = np.load(npy_path)
                    reconstructed_audio = spectrogram_to_audio(npy_data)

                    # Normalize the audio to a reasonable range
                    reconstructed_audio /= np.max(np.abs(reconstructed_audio))

                    # Save reconstructed audio
                    output_wav_path = os.path.join(output_subfolder, file.replace('.npy', '.wav'))
                    wavfile.write(output_wav_path, 44100, reconstructed_audio.astype(np.float32))

# Source and destination folders
train_input_folder = 'musdb18/spectrograms/train'
test_input_folder = 'musdb18/spectrograms/test'
train_output_folder = 'musdb18/DeepMask/train'
test_output_folder = 'musdb18/DeepMask/test'

# Convert train set
convert_npy_to_wav(train_input_folder, train_output_folder)

# Convert test set
convert_npy_to_wav(test_input_folder, test_output_folder)

print("Conversion completed.")

