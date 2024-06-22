import os
import numpy as np

# Load the general drum mask
general_drum_mask = np.load('general_drum_mask.npy')

# Path to the train folder
train_folder = 'musdb18/spectrograms/train'

# Path to the new folder for the masked spectrograms
output_folder = os.path.join(train_folder, 'lq_drums')
os.makedirs(output_folder, exist_ok=True)

# Process every instance in the train folder
for chunk_file in os.listdir(train_folder):
    chunk_name, ext = os.path.splitext(chunk_file)
    if ext:  # Check if the file has an extension (not a directory)
        input_path = os.path.join(train_folder, chunk_file)
        original_spec = np.load(input_path).real

        # Mask the original spectrogram to extract drum frequencies
        masked_spec = original_spec * general_drum_mask

        # Save masked spectrogram with the original name
        output_path = os.path.join(output_folder, chunk_name + '.npy')
        np.save(output_path, masked_spec)

