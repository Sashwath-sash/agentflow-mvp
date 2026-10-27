import json
import tempfile
import json
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
                       json.dumps({"action": "retry", "revised_instruction": "Use the actual 80 percent result.",
                                   "rationale": "The review found the result was misstated"}),
                       "Accuracy was 80 percent (paper.txt, page 1).",
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
            roles = {agent["role"] for agent in state["agents"]}
            assert {"Planner Agent", "Document Analyst", "Writer", "Semantic Review Agent", "Replanner Agent"} <= roles
            assert any(h["kind"] == "failure_feedback" for h in state["handoffs"])
            assert any(h["kind"] == "revised_instruction" for h in state["handoffs"])
            assert state["tasks"][-1]["agent"]["approved_tools"] == ["report"]


def test_agent_dashboard_and_dependency_graph_are_served():
    page = TestClient(app).get("/").text
    assert 'id="agents"' in page
    assert 'id="handoffs"' in page
    assert 'id="graph"' in page
    assert "function renderGraph(s)" in page
    assert "uploads.push({...data,signature})" in page
    assert "Maximum 10 files total" in page


def test_missing_papers_or_crossref_are_rejected_before_model_call():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"), patch.dict("os.environ", {"GEMINI_API_KEY": "test-not-real"}):
            client = TestClient(app)
            fid = client.post("/upload", files={"file": ("results.csv", b"model,top1\ninception_v3,77.5\n")}).json()["id"]
            with patch.object(wf.GeminiProvider, "generate") as model:
                response = client.post("/execute", json={"objective": "Compare the two papers and find Crossref results", "file_ids": [fid], "provider": "gemini"})
                assert response.status_code == 400
                assert "fewer than two documents" in response.json()["detail"]
                assert model.call_count == 0
            paper_ids = [client.post("/upload", files={"file": (f"paper-{i}.txt", b"Evidence")}).json()["id"] for i in (1, 2)]
            response = client.post("/execute", json={"objective": "Compare the two papers and find Crossref results", "file_ids": paper_ids, "provider": "none"})
            assert response.status_code == 400
            assert "Turn on 'Search Crossref'" in response.json()["detail"]


def test_old_run_with_missing_inputs_cannot_retry_or_replan():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"):
            file = wf.add_file("results.csv", b"model,top1\ninception_v3,77.5\n")
            state = wf.run_workflow("Summarize data", [file["id"]])
            state["objective"] = "Compare the two papers and find Crossref results"
            state["file_ids"] = [file["id"]]
            state["search"] = False
            state["status"] = "incomplete"
            state["tasks"][-1]["status"] = "failed"
            wf.save_run(state)
            response = TestClient(app).post("/runs/" + state["id"] + "/retry")
            assert response.status_code == 400
            assert "start a new run" in response.json()["detail"]
            task = state["tasks"][-1]
            task.update(attempts=1, error="Missing comparison of two uploaded papers")
            with patch.object(wf.GeminiProvider, "generate") as model:
                assert wf.replan_failed_branch(state, task) is False
                assert model.call_count == 0
            assert state["agents"][-1]["decision"] == "stop"
            assert state["handoffs"][-1]["kind"] == "input_change_required"


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


def test_context_preserves_abstract_and_numeric_results():
    pages = [{"page": 1, "text": "Abstract. A total of 201 participants completed the survey. " + "Background " * 300}]
    pages += [{"page": i, "text": "Discussion " * 500} for i in range(2, 8)]
    pages += [{"page": 8, "text": "Results Table 1. 37% used an app; 63% did not. " + "Details " * 200},
              {"page": 9, "text": "Average result was 85.58; comparison was 87.30. " + "Details " * 200}]
    context = evidence_context([{"kind": "document", "name": "study.pdf", "pages": pages}],
                               7000, "Compare participant numbers and numerical findings")
    assert "201 participants" in context
    assert "37%" in context
    assert "85.58" in context


def test_generated_doi_must_have_evidence():
    evidence = [{"kind": "document", "pages": [{"page": 1, "text": "DOI: 10.1000/real-paper"}]},
                {"kind": "sources", "sources": [{"doi": "10.2000/retrieved-paper"}]}]
    wf.validate_doi_provenance("See 10.1000/real-paper and 10.2000/retrieved-paper.", evidence)
    with pytest.raises(ValueError, match="DOI absent"):
        wf.validate_doi_provenance("See 10.3000/invented-paper.", evidence)
    wf.validate_doi_links("[10.1000/real-paper](https://doi.org/10.1000/real-paper)")
    with pytest.raises(ValueError, match="linked destination"):
        wf.validate_doi_links("[https://doi.org/10.1000/real-paper](https://doi.org/1r1000/real-paper)")


def test_comparison_agent_rejects_unprovided_crossref_or_csv_claims():
    docs = [{"kind": "document", "name": name, "pages": [{"page": 1, "text": "A documented method and result."}]}
            for name in ("first.pdf", "second.pdf")]
    parents = [{"output": doc} for doc in docs]
    with patch.object(wf, "ask", return_value="The CSV and Crossref results prove the comparison"):
        with pytest.raises(ValueError, match="outside its two uploaded documents"):
            wf.execute_tool({"tool": "compare", "instruction": "Compare papers"}, parents,
                            {"provider": "gemini", "objective": "Compare the two papers"})


def test_requested_search_cannot_be_silently_omitted():
    with pytest.raises(ValueError, match="must search"):
        validate_plan(graph("allowed"), ["allowed"], search=True)


def test_requested_comparison_and_chart_cannot_be_omitted():
    files = [{"id": "a", "suffix": ".pdf"}, {"id": "b", "suffix": ".pdf"},
             {"id": "data", "suffix": ".csv"}]
    raw = {"tasks": [
        {"id": fid, "tool": "read", "role": "Reader", "instruction": "Read input", "dependencies": [], "args": {"file_id": fid}}
        for fid in ("a", "b", "data")
    ] + [{"id": "report", "tool": "report", "role": "Writer", "instruction": "Report",
          "dependencies": ["a", "b", "data"], "args": {}}]}
    with pytest.raises(ValueError, match="must compare"):
        validate_plan(raw, ["a", "b", "data"], files=files, objective="Compare the two papers and make a chart")
    raw["tasks"].insert(-1, {"id": "compare", "tool": "compare", "role": "Comparator", "instruction": "Compare",
                              "dependencies": ["a", "b"], "args": {}})
    raw["tasks"][-1]["dependencies"].append("compare")
    with pytest.raises(ValueError, match="requested data chart"):
        validate_plan(raw, ["a", "b", "data"], files=files, objective="Compare the two papers and make a chart")


def test_data_discovery_chart_is_required_for_every_uploaded_table():
    files = [{"id": "first", "suffix": ".csv"}, {"id": "second", "suffix": ".xlsx"}]
    raw = {"tasks": [
        {"id": fid, "tool": "read", "role": "Reader", "instruction": "Read data",
         "dependencies": [], "args": {"file_id": fid}}
        for fid in ("first", "second")
    ] + [{"id": "report", "tool": "report", "role": "Writer", "instruction": "Report",
          "dependencies": ["first", "second"], "args": {}}]}
    with pytest.raises(ValueError, match="requested data chart"):
        validate_plan(raw, ["first", "second"], files=files, objective="Analyze these datasets")
    raw["tasks"].insert(-1, {"id": "charts", "tool": "chart", "role": "Visualization",
                             "instruction": "Discover charts", "dependencies": ["first"], "args": {}})
    raw["tasks"][-1]["dependencies"].append("charts")
    with pytest.raises(ValueError, match="every uploaded data file"):
        validate_plan(raw, ["first", "second"], files=files, objective="Analyze these datasets")


def test_generated_python_is_not_an_available_tool():
    raw = graph("allowed")
    raw["tasks"][0]["tool"] = "python"
    with pytest.raises(ValueError):
        validate_plan(raw, ["allowed"])


def test_single_document_cannot_trigger_comparison():
    raw = graph("allowed")
    raw["tasks"].insert(1, {"id":"compare", "tool":"compare", "role":"Comparison", "instruction":"Compare papers", "dependencies":["extract"], "args":{}})
    raw["tasks"][-1]["dependencies"]=["extract","compare"]
    with pytest.raises(ValueError, match="two document reads"):
        validate_plan(raw, ["allowed"], files=[{"id":"allowed","suffix":".pdf"}])


def test_missing_key_fails_before_work_starts():
    with patch.dict("os.environ", {}, clear=True):
        r = TestClient(app).post("/execute", json={"objective": "Search papers", "search": True, "provider": "gemini"})
        assert r.status_code == 400 and "saved Gemini key" in r.json()["detail"]
