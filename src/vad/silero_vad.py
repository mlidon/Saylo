from pathlib import Path
import numpy as np
import torch
import wave

SAMPLE_RATE = 16000
FRAME_SIZE = 512

VOICE_FRAMES_START = 5   # ~160 ms
VOICE_FRAMES_END = 10    # ~320 ms

class SileroVAD:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (self.read_audio,
         self.get_speech_timestamps,
         self.save_audio,
         self.read_audio_batch,
         self.get_speech_timestamps_batch) = utils

    def speech_prob(self, frame: np.ndarray) -> float:
        if len(frame) != FRAME_SIZE:
            return 0.0
        with torch.no_grad():
            prob = self.model(torch.from_numpy(frame), self.sample_rate).item()
        return float(prob)

def save_wav(path: Path, audio: np.ndarray, samplerate: int = SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_int16 = (audio * 32767).astype('int16')
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())
