from common import viz
from nussl.separation.deep import DeepMaskEstimation
from common.models import MaskInference
from nussl.evaluation import BSSEvalScale
from common import data
import torch
import nussl
import json
import numpy as np
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_MIXTURES = int(1e8)

# Normalize the audio signals
def normalize_signal(signal):
    max_val = np.max(np.abs(signal.audio_data))
    if max_val > 0:
        signal.audio_data /= max_val
    return signal

# Function to handle potential zero-energy signals
def handle_zero_energy(signal):
    energy = np.sum(signal.audio_data ** 2)
    if energy == 0:
        signal.audio_data += 1e-10  # Add a small value to avoid zero energy
    return signal

# Load your models
separator1 = nussl.separation.deep.DeepMaskEstimation(
    nussl.AudioSignal(), model_path='vocals/checkpoints/best.model.pth', device=DEVICE)
separator2 = nussl.separation.deep.DeepMaskEstimation(
    nussl.AudioSignal(), model_path='drums/checkpoints/best.model.pth', device=DEVICE)
separator3 = nussl.separation.deep.DeepMaskEstimation(
    nussl.AudioSignal(), model_path='bass/checkpoints/best.model.pth', device=DEVICE)

stft_params = nussl.STFTParams(window_length=512, hop_length=128)
test_folder = "musdb18/DeepMask/test"
test_dataset = data.mixer(stft_params, transform=None, fg_path=test_folder, num_mixtures=MAX_MIXTURES, coherent_prob=1.0)

# Ensure output directory exists
output_folder = Path("Results")
output_folder.mkdir(parents=True, exist_ok=True)

for i in range(500):
    item = test_dataset[i]
    mix_signal = normalize_signal(item['mix'])

    separator1.audio_signal = mix_signal
    separator2.audio_signal = mix_signal
    separator3.audio_signal = mix_signal

    estimates1 = separator1()
    estimates2 = separator2()
    estimates3 = separator3()

    estimates1[0] = handle_zero_energy(estimates1[0])
    estimates2[0] = handle_zero_energy(estimates2[0])
    estimates3[0] = handle_zero_energy(estimates3[0])

    source_keys = list(item['sources'].keys())
    estimates = {
        'vocals': estimates1[0],
        'drums': estimates2[0],
        'bass': estimates3[0],
        'other': mix_signal - estimates1[0] - estimates2[0] - estimates3[0]
    }

    sources = [handle_zero_energy(normalize_signal(item['sources'][k])) for k in source_keys]
    estimates = [handle_zero_energy(normalize_signal(estimates[k])) for k in source_keys]

    evaluator = BSSEvalScale(sources, estimates, source_labels=source_keys)
    try:
        scores = evaluator.evaluate()
    except np.linalg.LinAlgError as e:
        print(f"Skipping evaluation for mixture {i} due to a numerical error: {e}")
        continue

    # Save results to the output directory with sequential filenames
    output_file = output_folder / f"{i+1}.json"
    with open(output_file, 'w') as f:
        json.dump(scores, f, indent=4)

