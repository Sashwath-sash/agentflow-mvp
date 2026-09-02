from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


class LocalAIError(RuntimeError):
    """Raised when the local Ollama runtime is unavailable."""


class OllamaProvider:
    """Free local model adapter for an Ollama server running on localhost."""

    def __init__(self, model: str | None = None, endpoint: str | None = None) -> None:
        self.model = model or os.getenv("AGENTFLOW_OLLAMA_MODEL", "llama3.2:3b")
        self.endpoint = (endpoint or os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")) + "/api/generate"

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise LocalAIError("prompt must not be empty")
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        request = Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except Exception as exc:
            raise LocalAIError("Ollama is unavailable; install it and start a local model") from exc
        return str(result.get("response", "")).strip()
