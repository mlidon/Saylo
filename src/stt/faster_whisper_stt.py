from pathlib import Path
from typing import Literal, Tuple
from faster_whisper import WhisperModel

class FasterWhisperSTT:
    def __init__(self,
                 model_size: str = "medium",
                 device: Literal["cpu", "cuda"] = "cpu",
                 compute_type: str = "int8"):

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, audio_path: Path) -> Tuple[str, str]:
        """
        Transcribe audio y devuelve:
        - texto transcrito
        - idioma detectado (código ISO: 'es', 'en', 'ca', etc.)
        """

        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=False
        )

        # Texto concatenado
        text = "".join(seg.text for seg in segments).strip()

        # Idioma detectado por Whisper
        lang = info.language

        return text, lang
