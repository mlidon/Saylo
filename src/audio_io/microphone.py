import sounddevice as sd
import numpy as np

DEFAULT_DEVICE_INDEX = 1
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 512  # compatible con Silero

class MicrophoneStream:
    def __init__(self,
                 device_index: int = DEFAULT_DEVICE_INDEX,
                 samplerate: int = SAMPLE_RATE,
                 channels: int = CHANNELS,
                 frame_size: int = FRAME_SIZE):
        self.device_index = device_index
        self.samplerate = samplerate
        self.channels = channels
        self.frame_size = frame_size
        self._stream = None

    def __enter__(self):
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.samplerate,
            dtype='float32',
            blocksize=self.frame_size,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_frame(self) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("MicrophoneStream no está iniciado")
        frame, _ = self._stream.read(self.frame_size)
        return frame.flatten().astype(np.float32)
