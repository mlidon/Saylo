#!/usr/bin/env python3
import subprocess
from pathlib import Path

def main():
    # Rutas adaptadas a tu entorno real
    piper_bin = Path("env/tools/piper/piper.exe")
    model_path = Path("env/tools/piper/models/es/es_ES-davefx-medium.onnx")

    if not piper_bin.exists():
        print("Piper binary not found:", piper_bin)
        return

    if not model_path.exists():
        print("Piper model not found:", model_path)
        return

    text = "Hola Marc, esta es una prueba de voz generada con Piper."
    output_wav = Path("tests/output/piper_test.wav")
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(piper_bin),
        "--model", str(model_path),
        "--output_file", str(output_wav)
    ]

    print("Running Piper TTS...")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input=text)

    print("Piper stdout:", stdout)
    print("Piper stderr:", stderr)
    print("Generated file:", output_wav)

if __name__ == "__main__":
    main()
