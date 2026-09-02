import json

from agentflow.local_ai import OllamaProvider


def test_ollama_provider_sends_local_request(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def fake_open(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        monkeypatch.setattr("agentflow.local_ai.json.load", lambda _: {"response": "local answer"})
        return Response()

    monkeypatch.setattr("agentflow.local_ai.urlopen", fake_open)
    assert OllamaProvider(model="test-model").generate("hello") == "local answer"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["stream"] is False
