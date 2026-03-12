from pathlib import Path
import subprocess

LANG_MODELS = {
    "es": Path("models/es/es_ES-davefx-medium.onnx"),
    "en": Path("models/en/en_US-lessac-medium.onnx"),
    "ca": Path("models/cat/ca_ES-upc_ona-medium.onnx"),
}

class PiperTTS:
    def __init__(self,
                 piper_exe: Path = Path("bin/piper.exe"),
                 model_path: Path = LANG_MODELS["es"],   # modelo por defecto
                 workdir: Path = Path("tests/output")):

        self.piper_exe = piper_exe
        self.model_path = model_path
        self.workdir = workdir

        # Crear carpetas
        (self.workdir / "wav").mkdir(parents=True, exist_ok=True)
        (self.workdir / "mp3").mkdir(parents=True, exist_ok=True)
        (self.workdir / "ogg").mkdir(parents=True, exist_ok=True)

    def set_language(self, lang: str):
        # Normalizar idioma (por si Whisper devuelve "es" o "es-ES")
        lang = lang.split("-")[0]

        if lang in LANG_MODELS:
            self.model_path = LANG_MODELS[lang]
        else:
            self.model_path = LANG_MODELS["es"]  # fallback seguro

    def synthesize(self, text: str, filename: str):
        wav_path = self.workdir / "wav" / f"{filename}.wav"
        mp3_path = self.workdir / "mp3" / f"{filename}.mp3"
        ogg_path = self.workdir / "ogg" / f"{filename}.ogg"

        # 1) Generar WAV con Piper
        cmd = [
            str(self.piper_exe),
            "--model", str(self.model_path),
            "--output_file", str(wav_path),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = proc.communicate(input=text.encode("utf-8"))

        if proc.returncode != 0:
            raise RuntimeError(f"Piper error: {stderr.decode('utf-8', errors='ignore')}")

        # 2) Convertir WAV → MP3
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame",
            "-b:a", "64k",
            str(mp3_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3) Convertir WAV → OGG/Opus
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-codec:a", "libopus",
            "-b:a", "32k",
            str(ogg_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4) Eliminar WAV
        wav_path.unlink()

        return mp3_path, ogg_path
