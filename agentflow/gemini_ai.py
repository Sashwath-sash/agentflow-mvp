from __future__ import annotations

import json
import os
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class GeminiError(RuntimeError):
    """Raised when Gemini cannot generate a response."""


class GeminiProvider:
    """Google Gemini REST adapter; usage is controlled by the user's free-tier limits."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("AGENTFLOW_GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY is not configured")
        if not prompt.strip():
            raise GeminiError("prompt must not be empty")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(self.model, safe='')}:generateContent"
        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
        request = Request(endpoint, data=payload, headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode()).get("error", {}).get("message", "provider rejected the request")
            except Exception:
                detail = "provider rejected the request"
            reason = {400: "invalid request or API key", 401: "authentication failed", 403: "key or model access denied", 404: "model unavailable", 429: "free quota exhausted; try later"}.get(exc.code, "provider unavailable")
            raise GeminiError(f"Gemini ({exc.code}): {reason}") from None
        except Exception as exc:
            raise GeminiError("Gemini request failed; check network and model configuration") from exc
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiError("Gemini returned no text") from exc
