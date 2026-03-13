from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid

from src.stt.faster_whisper_stt import FasterWhisperSTT
from src.llm.ollama_client import OllamaClient
from src.tts.piper_tts import PiperTTS
from src.pipeline.conversation_pipeline import ConversationPipeline

app = FastAPI()
from fastapi.staticfiles import StaticFiles

app.mount("/panel", StaticFiles(directory="panel"), name="panel")

# MUY IMPORTANTE: permitir que el panel web pueda llamar al backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Puedes poner ["http://localhost"] si quieres limitarlo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar Saylo
stt = FasterWhisperSTT(model_size="medium", device="cpu", compute_type="int8")
llm = OllamaClient(model="llama3")
tts = PiperTTS()
pipeline = ConversationPipeline(stt=stt, llm=llm, tts=tts)

@app.get("/")
def serve_frontend():
    return FileResponse("panel/index.html")


@app.post("/api/text")
async def text_endpoint(payload: dict):
    text = payload["text"]
    lang = payload.get("lang", "es")

    reply = llm.ask(text, lang)
    mp3_out, ogg_out = tts.synthesize(reply, filename=f"reply_{uuid.uuid4()}")

    return {
        "user": text,
        "reply": reply,
        "lang": lang,
        "audio": str(mp3_out)
    }

@app.post("/api/audio")
async def audio_endpoint(file: UploadFile = File(...)):
    temp_wav = Path(f"temp_{uuid.uuid4()}.wav")

    with temp_wav.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcript, reply, lang, mp3_in, ogg_in, mp3_out, ogg_out = pipeline.process_segment(temp_wav)

    return {
        "transcript": transcript,
        "reply": reply,
        "lang": lang,
        "audio": str(mp3_out)
    }
