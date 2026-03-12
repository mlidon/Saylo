#!/usr/bin/env python3
import torch
from pathlib import Path
import wave
import numpy as np

SAMPLE_RATE = 16000

def load_wav_mono(path):
    with wave.open(str(path), 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == SAMPLE_RATE
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype('float32') / 32767.0
    return audio

def main():
    audio_path = Path("tests/output/record_test.wav")
    if not audio_path.exists():
        print("No existe tests/output/record_test.wav. Ejecuta antes record_test.py")
        return

    print("Cargando modelo Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (read_audio,
     get_speech_timestamps,
     save_audio,
     read_audio_batch,
     get_speech_timestamps_batch) = utils

    print("Leyendo audio...")
    wav = load_wav_mono(audio_path)

    print("Detectando voz...")
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLE_RATE
    )

    if not speech_timestamps:
        print("No se ha detectado voz.")
        return

    print("Segmentos de voz detectados (en muestras):")
    for seg in speech_timestamps:
        start_sec = seg['start'] / SAMPLE_RATE
        end_sec = seg['end'] / SAMPLE_RATE
        print(f"- Desde {start_sec:.2f}s hasta {end_sec:.2f}s")

if __name__ == "__main__":
    main()
