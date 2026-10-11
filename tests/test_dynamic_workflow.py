import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from agentflow.api import app
from agentflow import workflow as wf
from agentflow.intelligence import validate_plan, evidence_context


def graph(fid):
    return {"tasks": [
        {"id": "extract", "tool": "read", "role": "Document Analyst", "instruction": "Read evidence", "dependencies": [], "args": {"file_id": fid}},
        {"id": "report", "tool": "report", "role": "Writer", "instruction": "Summarize the evidence", "dependencies": ["extract"], "args": {}}]}


@pytest.mark.parametrize("attack", ["cycle", "unknown_file", "extra_tool", "missing_evidence", "duplicate"])
def test_untrusted_plans_cannot_expand_permissions(attack):
    raw = graph("allowed")
    if attack == "cycle": raw["tasks"][0]["dependencies"] = ["report"]
    if attack == "unknown_file": raw["tasks"][0]["args"]["file_id"] = "secret"
    if attack == "extra_tool": raw["tasks"][0]["tool"] = "shell"
    if attack == "missing_evidence": raw["tasks"][1]["dependencies"] = []
    if attack == "duplicate": raw["tasks"][1]["id"] = "extract"
    with pytest.raises(ValueError): validate_plan(raw, ["allowed"])


def test_dynamic_planning_review_repair_and_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"), patch.dict("os.environ", {"GEMINI_API_KEY": "test-not-real"}):
            client = TestClient(app)
            fid = client.post("/upload", files={"file": ("paper.txt", b"Accuracy was 80 percent.")}).json()["id"]
            answers = [json.dumps(graph(fid)), "Accuracy was 90 percent.",
                       json.dumps({"passed": False, "coverage": 0.5, "issues": ["Accuracy must be 80 percent"]}),
                       "Use the actual 80 percent result.", "Accuracy was 80 percent (paper.txt, page 1).",
                       json.dumps({"passed": True, "coverage": 1, "issues": []})]
            with patch.object(wf.GeminiProvider, "generate", side_effect=answers):
                response = client.post("/execute", json={"objective": "Summarize accuracy", "file_ids": [fid], "provider": "gemini"})
            state = client.get("/runs/" + response.json()["id"]).json()
            assert state["status"] == "completed", state
            assert state["planning_mode"] == "AI-generated validated graph"
            assert state["tasks"][0]["attempts"] == 1
            assert state["tasks"][-1]["attempts"] == 2
            assert state["metrics"]["model_calls"] == 6
            assert state["events"][0]["action"] == "branch retry"
            assert "80 percent" in state["tasks"][-1]["output"]["markdown"]


def test_bad_planner_never_silently_falls_back():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"), patch.dict("os.environ", {"GEMINI_API_KEY": "test-not-real"}):
            client = TestClient(app)
            fid = client.post("/upload", files={"file": ("paper.txt", b"Evidence")}).json()["id"]
            with patch.object(wf.GeminiProvider, "generate", return_value="not JSON"):
                r = client.post("/execute", json={"objective": "Summarize", "file_ids": [fid], "provider": "gemini"})
            state = wf.get_run(r.json()["id"])
            assert state["status"] == "incomplete"
            assert "Planning failed" in state["error"]
            assert all(t["attempts"] == 0 for t in state["tasks"])


def test_context_includes_later_pages_and_documents():
    output = [{"kind": "document", "name": "paper.pdf", "pages": [{"page": i, "text": "Evidence " * 1000} for i in range(1, 11)]}]
    context = evidence_context(output, 2000)
    assert '"page": 10' in context and "TRUNCATED" in context


def test_generated_python_is_not_an_available_tool():
    raw = graph("allowed")
    raw["tasks"][0]["tool"] = "python"
    with pytest.raises(ValueError):
        validate_plan(raw, ["allowed"])


def test_missing_key_fails_before_work_starts():
    with patch.dict("os.environ", {}, clear=True):
        r = TestClient(app).post("/execute", json={"objective": "Search papers", "search": True, "provider": "gemini"})
        assert r.status_code == 400 and "saved Gemini key" in r.json()["detail"]
