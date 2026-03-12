#!/usr/bin/env python3
import torch

def main():
    print("Loading Silero VAD model...")

    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )

    print("Silero VAD model loaded successfully.")

    # utils es una tupla, así que la desempaquetamos
    (read_audio,
     get_speech_timestamps,
     save_audio,
     read_audio_batch,
     get_speech_timestamps_batch) = utils

    print("Functions loaded:")
    print(" - read_audio")
    print(" - get_speech_timestamps")
    print(" - save_audio")
    print(" - read_audio_batch")
    print(" - get_speech_timestamps_batch")

if __name__ == "__main__":
    main()
