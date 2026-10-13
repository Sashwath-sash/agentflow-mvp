import tempfile
from pathlib import Path
from unittest.mock import patch

from agentflow import workflow as wf
from agentflow.evaluate import evaluate_run, format_checklist


def test_format_checklist_is_explicitly_not_factual_accuracy():
    state = {"tasks": [{"tool": "read", "output": {"kind": "data", "name": "data.csv",
                         "summary": {"rows": 12, "columns": 3}}}]}
    result = format_checklist("data.csv has 12 rows and 3 columns. Limitations apply.", state)
    assert result["passed"] == result["total"] == 3
    assert "not establish factual correctness" in result["note"]


def test_one_call_baseline_uses_saved_evidence_without_reexecuting_tools():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"):
            file = wf.add_file("data.csv", b"model,score\na,80\nb,90\n")
            state = wf.run_workflow("Summarize the data", [file["id"]])

            class FakeModel:
                last_usage = {"total_tokens": 123}
                def generate(self, prompt):
                    assert "data.csv" in prompt and "2" in prompt
                    return "data.csv has 2 rows and 2 columns. Limitations: small dataset."

            with patch.object(wf, "model_for", return_value=FakeModel()) as model:
                result = evaluate_run(state["id"], provider="gemini")
            assert model.call_count == 1
            assert result["single_call_baseline"]["model_calls"] == 1
            assert result["single_call_baseline"]["provider_tokens"] == 123
            assert (Path(directory) / ("evaluation-" + state["id"] + ".json")).is_file()
            assert (Path(directory) / ("baseline-" + state["id"] + ".md")).is_file()
