from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import io as io1
import base64
from common.models import MaskInference
from nussl.separation.deep import DeepMaskEstimation
import torch
import nussl
import shutil
from pedalboard import *
from pedalboard.io import AudioFile
import os
import zipfile
from pydub import AudioSegment

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_MIXTURES = int(1e8)

@csrf_exempt
def create_export(request):
    if request.method == 'POST':
        # Check if audio files are provided
        if 'audio_files' not in request.FILES:
            return JsonResponse({'error': 'No audio files provided.'}, status=400)

        audio_files = request.FILES.getlist('audio_files')
        volumes = request.POST.getlist('volume_values')
        pans = request.POST.getlist('panning_values')

        # Check if the length of provided files matches the length of volumes and pans
        if len(audio_files) != len(volumes) or len(audio_files) != len(pans):
            return JsonResponse({'error': 'Mismatch in the number of audio files, volume values, and panning values.'}, status=400)

        # Temporary directory to store processed audio files
        temp_dir = 'temp_audio'
        os.makedirs(temp_dir, exist_ok=True)

        processed_audio_files = []

        # Apply volume and panning effects to each audio file
        for i, audio_file in enumerate(audio_files):
            volume = float(volumes[i])
            pan = float(pans[i])
            
            # Save the uploaded file
            with open(os.path.join(temp_dir, audio_file.name), 'wb') as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)

            # Load audio using Pydub
            audio = AudioSegment.from_file(os.path.join(temp_dir, audio_file.name))

            # Apply volume and panning effects
            audio = audio.pan(pan)
            audio = audio + volume

            # Save processed audio
            processed_audio_path = os.path.join(temp_dir, f'processed_{audio_file.name}')
            audio.export(processed_audio_path, format='wav')

            processed_audio_files.append(processed_audio_path)

        # Combine all processed audio files
        final_audio = None
        for processed_audio_path in processed_audio_files:
            audio = AudioSegment.from_file(processed_audio_path)
            if final_audio is None:
                final_audio = audio
            else:
                final_audio = final_audio.overlay(audio, position=0)

        # Export the final project as a WAV file
        final_project_path = os.path.join(temp_dir, 'final_project.wav')
        final_audio.export(final_project_path, format='wav')

        # Create a zip archive of processed audio files
        zip_filename = 'processed_audio.zip'
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            for processed_audio_path in processed_audio_files:
                zip_file.write(processed_audio_path, arcname=os.path.basename(processed_audio_path))
            zip_file.write(final_project_path, arcname='final_project.wav')

        # Encode the zip archive as Base64
        with open(zip_filename, 'rb') as zip_file:
            zip_base64 = base64.b64encode(zip_file.read()).decode('utf-8')

        # Remove temporary directory
        shutil.rmtree(temp_dir)

        return JsonResponse({'processed_audio_zip': zip_base64})

    return JsonResponse({'error': 'Invalid request method.'}, status=400)



@csrf_exempt
def process_audio_with_effects(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        effect_names = request.POST.getlist('effects')

        # Save the uploaded file
        with open('uploaded_audio.wav', 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Create a Pedalboard object with specified effects
        effects = []
        available_effects = {
            'pitch shifter': PitchShift(semitones=12),
            'reverb': Reverb(room_size=0.75),
            # Add more effects as needed
        }
        for effect_name in effect_names:
            # Clean up the effect name and split if there are multiple effects
            effect_names_split = [name.strip().lower() for name in effect_name.split(',')]
            for name in effect_names_split:
                effect = available_effects.get(name)
                if effect:
                    effects.append(effect)
                else:
                    print(f"Effect '{name}' not found. Skipping...")

        board = Pedalboard(effects)

        samplerate = 44100.0
        with AudioFile('uploaded_audio.wav').resampled_to(samplerate) as f:
            audio = f.read(f.frames)
        effected = board(audio, samplerate)
        with AudioFile('processed_audio.wav', 'w', samplerate, effected.shape[0]) as f:
            f.write(effected)

        # Read the processed audio file as bytes
        with open('processed_audio.wav', 'rb') as f:
            audio_bytes = f.read()

        # Remove the temporary uploaded audio file and the processed audio file
        os.remove('uploaded_audio.wav')
        #os.remove('processed_audio.wav')

        # Encode the processed audio as Base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return JsonResponse({'processed_audio': audio_base64})

    return JsonResponse({'error': 'Invalid request method or no file provided.'}, status=400)



@csrf_exempt
def separate_audio(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']

        # Save the uploaded file
        with open('uploaded_audio.wav', 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # List of model paths
        model_paths = ['vocals/checkpoints/best.model.pth', 'drums/checkpoints/best.model.pth', 'bass/checkpoints/best.model.pth']

        all_wav_data = []
        all_waveform_images = []

        # Resample the uploaded audio to 44100 Hz
        y, sr = librosa.load('uploaded_audio.wav', sr=44100)
        sf.write('uploaded_audio_resampled.wav', y, sr)

        # Load the original audio file
        original_audio_signal = nussl.AudioSignal('uploaded_audio_resampled.wav')
        original_signal_copy = original_audio_signal

        for model_path in model_paths:
            # Load the model
            separator = nussl.separation.deep.DeepMaskEstimation(
                nussl.AudioSignal(), model_path=model_path,
                device=DEVICE,
            )

            # Process the audio file
            separator.audio_signal = original_audio_signal
            estimates = separator()

            # If the model is for bass separation, subtract 
            if 'bass' in model_path:
                bass_estimate = estimates[0]
                estimates[0] = original_audio_signal - bass_estimate

                # Save bass estimate to WAV file
                #bass_output_file = 'bass_estimate.wav'
                #estimates[0].write_audio_to_file(bass_output_file)

            original_signal_copy = original_signal_copy - estimates[0]

            # Save separated sources to WAV files
            output_files = []
            waveform_images = []  # List to store waveform image data
            for i, estimate in enumerate(estimates):
                output_file = f'estimated_source_{i}_{os.path.basename(model_path)}.wav'
                estimate.write_audio_to_file(output_file)
                output_files.append(output_file)

                # Generate waveform image
                waveform_image_data = generate_waveform_image_data(output_file)
                waveform_images.append(waveform_image_data)  # Append waveform image data to the list

            # Encode WAV files as Base64 strings
            wav_data = []
            for output_file in output_files:
                with open(output_file, 'rb') as f:
                    wav_bytes = f.read()
                    wav_base64 = base64.b64encode(wav_bytes).decode('utf-8')
                    wav_data.append(wav_base64)
                # Remove the temporary WAV file
                os.remove(output_file)

            all_wav_data.append(wav_data)
            all_waveform_images.append(waveform_images)


        output_file = f'other.wav'
        original_signal_copy.write_audio_to_file(output_file)
        
        waveform_images = []
        # Generate waveform image
        waveform_image_data = generate_waveform_image_data(output_file)
        waveform_images.append(waveform_image_data)  # Append waveform image data to the list
        
        wav_data = []
        with open(output_file, 'rb') as f:
            wav_bytes = f.read()
            wav_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            wav_data.append(wav_base64)
        # Remove the temporary WAV file
        os.remove(output_file)
        
        all_waveform_images.append(waveform_images)
        all_wav_data.append(wav_data)
        

        return JsonResponse({
            'separated_files': all_wav_data,
            'waveform_images': all_waveform_images,  # Return the list of waveform image data
        })

    return JsonResponse({'error': 'Invalid request method or no file provided.'}, status=400)

def generate_waveform_image_data(audio_file_path):
    # Load audio file
    y, sr = librosa.load(audio_file_path, sr=None)

    # Generate time array
    t = np.arange(0, len(y)) / sr

    # Plot waveform without axis
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, color='b')
    plt.axis('off')  # Turn off axis
    plt.margins(0, 0)  # Remove margins
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove x-axis ticks
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Remove y-axis ticks

    # Encode waveform image as base64
    buffer = io1.BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    plt.close()
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')

    return image_data
    
@csrf_exempt
def get_image(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']

        # Save the uploaded file
        with open('uploaded_audio.wav', 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Generate waveform image data
        waveform_image_data = generate_waveform_image_data('uploaded_audio.wav')

        # Remove the temporary uploaded audio file
        os.remove('uploaded_audio.wav')

        return JsonResponse({'waveform_image': waveform_image_data})

    return JsonResponse({'error': 'Invalid request method or no file provided.'}, status=400)

