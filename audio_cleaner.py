import os
import torch
import librosa
import numpy as np
import soundfile as sf
import csv
from pathlib import Path
from tqdm import tqdm

def main():
    audio_dir = Path("videos/02_passed/audio_pool")
    if not audio_dir.exists():
        print("Audio pool not found.")
        return

    wav_files = list(audio_dir.rglob("*.wav"))
    print(f"Scanning {len(wav_files)} audio files using Silero VAD...")

    # Load Silero VAD from PyTorch Hub
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False,
                                  trust_repo=True)
    
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    total_trimmed = 0
    total_rejected = 0

    for wav_path in tqdm(wav_files):
        try:
            # Load with librosa to bypass torchaudio requirements
            y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
            wav = torch.from_numpy(y)
            
            # Get timestamps of actual speech
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, threshold=0.5)
            
            if not speech_timestamps:
                # No speech found at all - delete or move it
                os.remove(wav_path)
                total_rejected += 1
                continue

            # Calculate total duration of actual speech
            total_speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
            speech_seconds = total_speech_samples / 16000.0
            
            # If the speech is less than 5 seconds total across the 30s clip, it's not dense enough
            if speech_seconds < 5.0:
                os.remove(wav_path)
                total_rejected += 1
                continue
                
            # --- The Fix: Isolate the Longest Continuous Speech Segment ---
            # We want a solid block of talking to drive the lip sync, not broken fragments
            longest_segment = max(speech_timestamps, key=lambda x: x['end'] - x['start'])
            segment_duration = (longest_segment['end'] - longest_segment['start']) / 16000.0
            
            # If even the longest continuous block is less than 3 seconds, reject it
            if segment_duration < 3.0:
                os.remove(wav_path)
                total_rejected += 1
                continue

            # Extract just that longest, dense block of continuous speech
            dense_audio = wav[longest_segment['start']:longest_segment['end']].numpy()
            
            # Overwrite the original file with just the dense speech block
            sf.write(str(wav_path), dense_audio, 16000)
            total_trimmed += 1

        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    print(f"\nCleanup Complete!")
    print(f"Rejected (No/Low Speech): {total_rejected}")
    print(f"Successfully Trimmed to Dense Speech: {total_trimmed}")

if __name__ == "__main__":
    main()
