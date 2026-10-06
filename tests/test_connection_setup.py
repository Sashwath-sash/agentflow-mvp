import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from fastapi.testclient import TestClient
from agentflow.api import app
from agentflow.gemini_ai import GeminiProvider


def test_saved_key_survives_environment_reset():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        target=Path(directory)/".env"
        with patch("agentflow.api.ENV_FILE",target),patch.dict(os.environ,{},clear=True):
            client=TestClient(app)
            result=client.post("/settings/gemini",json={"key":"test-only-placeholder"})
            assert result.status_code==200
            assert "test-only-placeholder" not in result.text
            os.environ.pop("GEMINI_API_KEY")
            from dotenv import load_dotenv
            load_dotenv(target)
            assert client.get("/settings/gemini").json()["configured"] is True


def test_removed_model_discovers_replacement():
    calls=[]
    def request(req,timeout):
        calls.append(req.full_url)
        if len(calls)==1:
            raise HTTPError(req.full_url,404,"removed",{},io.BytesIO(b"{}"))
        if len(calls)==2:
            return io.BytesIO(json.dumps({"models":[{"name":"models/gemini-test-flash",
                "supportedGenerationMethods":["generateContent"]}]}).encode())
        return io.BytesIO(b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}')
    with patch("agentflow.gemini_ai.urlopen",side_effect=request):
        provider=GeminiProvider(api_key="test-only-placeholder",model="removed-model")
        assert provider.generate("ping")=="ok"
        assert provider.model=="gemini-test-flash"
        assert len(calls)==3
        assert all("test-only-placeholder" not in url for url in calls)

