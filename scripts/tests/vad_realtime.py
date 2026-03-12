#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import torch
from pathlib import Path
import wave
import time

DEVICE_INDEX = 1
SAMPLE_RATE = 16000
CHANNELS = 1

# Silero requiere EXACTAMENTE 512 muestras por frame a 16kHz
FRAME_SIZE = 512

def save_wav(path, audio, samplerate):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_int16 = np.int16(audio * 32767)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

def main():
    print("Cargando Silero VAD...")
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

    print("Iniciando escucha en tiempo real...")
    buffer = []
    recording = False
    segment_index = 0

    with sd.InputStream(
        device=DEVICE_INDEX,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype='float32',
        blocksize=FRAME_SIZE
    ) as stream:

        while True:
            frame, _ = stream.read(FRAME_SIZE)
            frame = frame.flatten()

            audio_np = frame.astype(np.float32)

            # Silero espera EXACTAMENTE 512 muestras
            if len(audio_np) != 512:
                continue

            speech_prob = model(torch.from_numpy(audio_np), SAMPLE_RATE).item()

            if speech_prob > 0.5:
                if not recording:
                    print("→ Voz detectada, empezando segmento...")
                    recording = True
                    buffer = []
                buffer.append(audio_np)
            else:
                if recording:
                    print("→ Fin de voz, guardando segmento...")
                    recording = False
                    audio = np.concatenate(buffer)
                    out_path = Path(f"tests/output/segment_{segment_index}.wav")
                    save_wav(out_path, audio, SAMPLE_RATE)
                    print("Guardado:", out_path)
                    segment_index += 1

            time.sleep(0.005)

if __name__ == "__main__":
    main()
