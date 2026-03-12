#!/usr/bin/env python3
from faster_whisper import WhisperModel

def main():
    print("Loading faster-whisper model (small)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    print("Model loaded successfully.")

    # Transcribir un pequeño archivo de prueba si existe
    audio_path = "tests/data/sample.wav"  # lo crearemos más adelante
    try:
        segments, info = model.transcribe(audio_path)
        print("Transcription language:", info.language)
        print("Segments:")
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    except Exception as e:
        print("No audio file or error during transcription:", e)

if __name__ == "__main__":
    main()
