from common import viz
from nussl.separation.deep import DeepMaskEstimation
from common.models import MaskInference
from common import data
import torch
import nussl

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_MIXTURES = int(1e8)

separator = nussl.separation.deep.DeepMaskEstimation(
    nussl.AudioSignal(), model_path='bass/checkpoints/best.model.pth',
    device=DEVICE,
)

stft_params = nussl.STFTParams(window_length=512, hop_length=128)

test_folder = "musdb18/DeepMask/test"
test_data = data.mixer(stft_params, transform=None, 
    fg_path=test_folder, num_mixtures=MAX_MIXTURES, coherent_prob=1.0)
item = test_data[227]

separator.audio_signal = item['mix']
estimates = separator()
estimates.append(item['mix'] - estimates[0])

viz.show_sources(estimates)

for i, estimate in enumerate(estimates):
    estimate.write_audio_to_file(f'estimated_source_{i}.wav')

print("Separated sources saved as estimated_source_0.wav, estimated_source_1.wav, etc.")
