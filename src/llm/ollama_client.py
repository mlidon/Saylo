import subprocess

class OllamaClient:
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.history = []

        self.system_prompt = (
            "Eres Saylo, un asistente conversacional que responde siempre en el mismo idioma "
            "que el usuario. Mantén un tono natural, directo y breve. No cambies de idioma "
            "a menos que el usuario cambie. No inventes información. No mezcles idiomas."
        )

    def ask(self, user_message: str, lang: str = "es") -> str:
        # Guardar mensaje del usuario
        self.history.append({"role": "user", "content": user_message})

        # Construir prompt completo
        prompt = f"SYSTEM: {self.system_prompt}\n"
        prompt += f"SYSTEM: Responde únicamente en {lang}.\n"

        for msg in self.history:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"{role}: {content}\n"

        prompt += "ASSISTANT:"

        # Llamada a Ollama
        proc = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"Ollama error: {proc.stderr.decode('utf-8', errors='ignore')}")

        reply = proc.stdout.decode("utf-8").strip()

        # Guardar respuesta en historial
        self.history.append({"role": "assistant", "content": reply})

        return reply
