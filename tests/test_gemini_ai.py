import json
import io
from urllib.error import HTTPError

from agentflow.gemini_ai import GeminiProvider


def test_gemini_provider_parses_response(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr("agentflow.gemini_ai.urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr("agentflow.gemini_ai.json.load", lambda _: {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})
    assert GeminiProvider(api_key="test-key").generate("Say hello") == "hello"


def test_gemini_provider_requires_key():
    from agentflow.gemini_ai import GeminiError
    try:
        GeminiProvider(api_key="").generate("hello")
    except GeminiError as exc:
        assert "GEMINI_API_KEY" in str(exc)
    else:
        raise AssertionError("missing key should fail")


def test_zero_quota_error_does_not_expose_provider_details(monkeypatch):
    def limited(request, timeout):
        body = b'{"error":{"message":"Quota exceeded for metric x, limit: 0, project 12345"}}'
        raise HTTPError(request.full_url, 429, "limited", {}, io.BytesIO(body))
    monkeypatch.setattr("agentflow.gemini_ai.urlopen", limited)
    from agentflow.gemini_ai import GeminiError
    try:
        GeminiProvider(api_key="test-secret").generate("hello")
    except GeminiError as exc:
        assert "zero free quota" in str(exc)
        assert "12345" not in str(exc)
        assert "test-secret" not in str(exc)
    else:
        raise AssertionError("quota failure should be shown")
