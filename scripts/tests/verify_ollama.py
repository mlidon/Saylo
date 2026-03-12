#!/usr/bin/env python3
import subprocess

def main():
    try:
        process = subprocess.Popen(
            ["ollama", "run", "llama3:8b"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate("Say 'Hello from Ollama'")

        print("Ollama stdout:")
        print(stdout)
        print("Ollama stderr:")
        print(stderr)

    except Exception as e:
        print("Error running Ollama:", e)

if __name__ == "__main__":
    main()