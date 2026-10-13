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
        self.api_key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key
        self.model = model or os.getenv("AGENTFLOW_GEMINI_MODEL", "gemini-3.5-flash-lite")

    def available_models(self) -> list[str]:
        request = Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", headers={"x-goog-api-key": self.api_key or ""})
        try:
            with urlopen(request, timeout=20) as response:
                models = json.load(response).get("models", [])
            return [m["name"].removeprefix("models/") for m in models
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                    and "flash" in m["name"] and not any(x in m["name"] for x in ("image", "audio", "live", "tts"))]
        except Exception:
            raise GeminiError("Unable to list available Gemini models") from None

    def generate(self, prompt: str, _retry: bool = True) -> str:
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
            # Do not silently switch to an arbitrary preview model: its free
            # quota may be zero even when another listed model works.
            try:
                error = json.loads(exc.read().decode()).get("error", {})
            except Exception:
                error = {}
            detail = str(error.get("message", "provider rejected the request"))
            # Google can include project identifiers and other diagnostics. Show only
            # the quota category and reset hint, never the raw provider response.
            if exc.code == 429:
                lower = detail.lower()
                if "per minute" in lower or "requestsperminute" in lower or "inputtokensperminute" in lower:
                    reason = "per-minute rate limit reached; wait and retry"
                elif "per day" in lower or "requestsperday" in lower or "daily" in lower:
                    reason = "daily quota reached; wait for the project quota reset"
                elif "limit: 0" in lower or "limit: 0.0" in lower:
                    reason = "this project has zero free quota for this model; check its AI Studio limits"
                else:
                    reason = "quota or rate limit reached; check this model's limits in AI Studio"
                raise GeminiError(f"Gemini (429), model {self.model}: {reason}") from None
            reason = {400: "invalid request or API key", 401: "authentication failed", 403: "key or model access denied", 404: "model unavailable", 429: "free quota exhausted; try later"}.get(exc.code, "provider unavailable")
            raise GeminiError(f"Gemini ({exc.code}): {reason}") from None
        except Exception as exc:
            raise GeminiError("Gemini request failed; check network and model configuration") from exc
        try:
            self.last_usage = {"total_tokens": result.get("usageMetadata", {}).get("totalTokenCount", 0)}
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiError("Gemini returned no text") from exc
