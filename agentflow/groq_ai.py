"""Explicit Groq provider. No automatic model/provider fallback."""
import json
import os
import time
import httpx

class GroqError(RuntimeError):
    pass

class GroqProvider:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("AGENTFLOW_GROQ_MODEL", "openai/gpt-oss-20b")

    def generate(self, prompt):
        if not self.api_key:
            raise GroqError("No saved Groq key. Save it in Groq connection settings.")
        if not prompt.strip():
            raise GroqError("Prompt must not be empty")
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "max_completion_tokens": 4096, "reasoning_effort": "low"}
        for attempt in range(4):
            try:
                response = httpx.post("https://api.groq.com/openai/v1/chat/completions", json=payload,
                                      headers={"Authorization": "Bearer " + self.api_key,
                                               "User-Agent": "AgentFlow/0.3 (academic research project)",
                                               "Accept": "application/json"}, timeout=60)
                response.raise_for_status()
                result = response.json()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                delay = exc.response.headers.get("retry-after", "")
                try:
                    seconds = float(delay)
                except ValueError:
                    seconds = 0
                if status == 429 and attempt < 3 and 0 < seconds <= 65:
                    wait = seconds + 1
                    callback = getattr(self, "on_rate_limit", None)
                    if callback:
                        callback(wait, attempt + 1)
                    time.sleep(wait)
                    continue
                reasons = {401: "key rejected", 403: "model access denied", 404: "configured model unavailable",
                           429: "rate or quota limit reached; your key remains saved", 413: "request exceeds token limit"}
                reason = reasons.get(status, "provider rejected request")
                if status == 403 and "error code: 1010" in exc.response.text:
                    reason = "Cloudflare blocked the HTTP client before model authorization"
                if delay:
                    reason += "; retry after " + delay + " seconds"
                raise GroqError(f"Groq ({status}), model {self.model}: {reason}") from None
            except (httpx.RequestError, ValueError):
                raise GroqError("Groq network request failed; check connectivity") from None
        self.last_usage = {"total_tokens": result.get("usage", {}).get("total_tokens", 0)}
        try:
            choice = result["choices"][0]
            text = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                raise GroqError("Groq output exceeded response budget; use a smaller task")
            if not isinstance(text, str) or not text.strip():
                raise GroqError("Groq returned no answer")
            return text.strip()
        except (KeyError, IndexError, TypeError):
            raise GroqError("Groq returned an invalid response") from None
