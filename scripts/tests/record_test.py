#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
from pathlib import Path
import wave

DEVICE_INDEX = 1          # Tu "Headset Microphone"
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_SEC = 5          # Graba 5 segundos

def save_wav(path, audio, samplerate):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_int16 = np.int16(audio * 32767)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

def main():
    print(f"Grabando {DURATION_SEC} segundos desde el dispositivo {DEVICE_INDEX}...")
    recording = sd.rec(
        int(DURATION_SEC * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        device=DEVICE_INDEX,
    )
    sd.wait()
    out_path = Path("tests/output/record_test.wav")
    save_wav(out_path, recording.flatten(), SAMPLE_RATE)
    print("Grabación guardada en:", out_path)

if __name__ == "__main__":
    main()
