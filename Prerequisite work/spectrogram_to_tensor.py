import numpy as np
import torch

def main():
    
    spectrogram_path = "masked_spectrogram.npy"
    masked_spectrogram = np.load(spectrogram_path)
    
    # Convert the numpy array to a PyTorch tensor
    spectrogram_tensor = torch.tensor(masked_spectrogram)
    
    # Print the shape of the PyTorch tensor
    print("Shape of the PyTorch tensor:", spectrogram_tensor.shape)

if __name__ == "__main__":
    main()

