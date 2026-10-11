import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import httpx
import pytest
from fastapi.testclient import TestClient
from agentflow.api import app
from agentflow.groq_ai import GroqProvider, GroqError
from agentflow.workflow import model_for

def test_groq_saved_without_echo_and_reloaded():
    with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True), patch('agentflow.api.ENV_FILE', Path(d)/'.env'):
        c=TestClient(app)
        r=c.post('/settings/groq',json={'key':'test-secret'})
        assert r.status_code==200 and 'test-secret' not in r.text
        os.environ.pop('GROQ_API_KEY')
        from dotenv import load_dotenv
        load_dotenv(Path(d)/'.env')
        assert c.get('/settings/groq').json()['configured']
        assert isinstance(model_for('groq'),GroqProvider)

def test_groq_response_and_no_fallback():
    request=httpx.Request('POST','https://api.groq.com/openai/v1/chat/completions')
    good=httpx.Response(200,json={"choices":[{"message":{"content":"OK"},"finish_reason":"stop"}],"usage":{"total_tokens":2}},request=request)
    with patch.dict(os.environ, {'GROQ_API_KEY':'test-secret'}), patch('agentflow.groq_ai.httpx.post',return_value=good):
        p=GroqProvider()
        assert p.generate('test')=='OK' and p.last_usage['total_tokens']==2
    limited=httpx.Response(429,text='limit',headers={'retry-after':'30'},request=request)
    with patch.dict(os.environ, {'GROQ_API_KEY':'test-secret'}), patch('agentflow.groq_ai.httpx.post',return_value=limited) as call:
        with pytest.raises(GroqError,match='30 seconds'): GroqProvider().generate('test')
        assert call.call_count==1
