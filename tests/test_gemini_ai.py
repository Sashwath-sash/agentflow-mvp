import json

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
