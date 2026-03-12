from pathlib import Path
import subprocess

class ConversationPipeline:
    def __init__(self, stt, llm, tts, workdir: Path = Path("tests/output")):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.workdir = workdir
        self.last_user = None
        self.last_assistant = None

        (self.workdir / "wav").mkdir(parents=True, exist_ok=True)
        (self.workdir / "mp3").mkdir(parents=True, exist_ok=True)
        (self.workdir / "ogg").mkdir(parents=True, exist_ok=True)

    def process_segment(self, wav_path: Path):
        # 1) Transcribir + detectar idioma
        transcript, lang = self.stt.transcribe(wav_path)

        # 2) Seleccionar voz según idioma
        self.tts.set_language(lang)

        # 3) Convertir entrada WAV → MP3
        mp3_in = self.workdir / "mp3" / wav_path.name.replace(".wav", ".mp3")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame",
            "-b:a", "64k",
            str(mp3_in)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4) Convertir entrada WAV → OGG
        ogg_in = self.workdir / "ogg" / wav_path.name.replace(".wav", ".ogg")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-codec:a", "libopus",
            "-b:a", "32k",
            str(ogg_in)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 5) Eliminar WAV original
        wav_path.unlink()

        # 6) Construir contexto conversacional
        context = ""
        if self.last_user and self.last_assistant:
            context = (
                f"Contexto previo: Usuario dijo '{self.last_user}'. "
                f"Saylo respondió '{self.last_assistant}'. "
            )

        # 7) Construir prompt para el LLM
        prompt = (
            f"{context}"
            f"Idioma detectado: {lang}. "
            f"Responde únicamente en {lang}. "
            f"Responde de forma breve y natural. "
            f"Usuario dijo: {transcript}"
        )

        # 8) LLM responde (IMPORTANTE: pasar lang)
        reply = self.llm.ask(prompt, lang)

        # Guardar memoria conversacional
        self.last_user = transcript
        self.last_assistant = reply

        # 9) TTS sintetiza y convierte
        mp3_out, ogg_out = self.tts.synthesize(reply, filename="reply")

        return transcript, reply, lang, mp3_in, ogg_in, mp3_out, ogg_out
