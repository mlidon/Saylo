#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.audio_io.microphone import MicrophoneStream
from src.vad.silero_vad import SileroVAD, save_wav, FRAME_SIZE, SAMPLE_RATE
from src.stt.faster_whisper_stt import FasterWhisperSTT
from src.llm.ollama_client import OllamaClient
from src.tts.piper_tts import PiperTTS
from src.pipeline.conversation_pipeline import ConversationPipeline

VOICE_FRAMES_START = 5
VOICE_FRAMES_END = 10

def main():
    vad = SileroVAD(sample_rate=SAMPLE_RATE)
    stt = FasterWhisperSTT(model_size="medium", device="cpu", compute_type="int8")
    llm = OllamaClient(model="llama3")
    tts = PiperTTS()
    pipeline = ConversationPipeline(stt=stt, llm=llm, tts=tts)

    buffer: list[np.ndarray] = []
    recording = False
    voice_count = 0
    silence_count = 0
    segment_index = 0

    print("Saylo escuchando en modo asistente clásico...")

    with MicrophoneStream() as mic:
        while True:
            frame = mic.read_frame()
            if len(frame) != FRAME_SIZE:
                continue

            prob = vad.speech_prob(frame)

            if prob > 0.5:
                voice_count += 1
                silence_count = 0
            else:
                silence_count += 1
                voice_count = 0

            if not recording and voice_count >= VOICE_FRAMES_START:
                print("→ Voz detectada, empezando segmento...")
                recording = True
                buffer = []

            if recording:
                buffer.append(frame)

            if recording and silence_count >= VOICE_FRAMES_END:
                print("→ Fin de voz, procesando segmento...")
                recording = False
                audio = np.concatenate(buffer)

                seg_path = Path(f"tests/output/wav/segment_{segment_index}.wav")
                save_wav(seg_path, audio, SAMPLE_RATE)
                segment_index += 1

                transcript, reply, lang, mp3_in, ogg_in, mp3_out, ogg_out = pipeline.process_segment(seg_path)

                print(f"[Idioma detectado]: {lang}")
                print(f"[TÚ]: {transcript}")
                print(f"[SAYLO]: {reply}")
                print(f"Audio entrada: {mp3_in}")
                print(f"Audio salida:  {mp3_out}")

            time.sleep(0.005)

if __name__ == "__main__":
    main()
