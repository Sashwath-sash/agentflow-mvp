"""Optional one-call synthesis baseline for a saved AgentFlow research run.

Run with: python -m agentflow.evaluate RUN_ID
This makes one call to the run's selected provider and never replaces its result.
"""
import argparse
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from . import workflow as wf


def format_checklist(report, state):
    """Transparent surface checks, not a factual accuracy measurement."""
    checks = {}
    for task in state["tasks"]:
        if task["tool"] != "read" or not task.get("output"):
            continue
        output = task["output"]
        checks["mentions " + output["name"]] = output["name"].lower() in report.lower()
        if output["kind"] == "data":
            summary = output["summary"]
            checks["reports data dimensions"] = (str(summary["rows"]) in report.replace(",", "") and
                                                 str(summary["columns"]) in report)
    if any(task["tool"] == "search" for task in state["tasks"]):
        checks["separates retrieved sources"] = bool(re.search(r"related sources|crossref|scholarly records", report, re.I))
    if sum(task["tool"] == "read" and task.get("output", {}).get("kind") == "document" for task in state["tasks"]):
        checks["uses page references"] = bool(re.search(r"\b(?:PDF\s+)?pages?\s+\d+|\bp\.\s*\d+", report, re.I))
    checks["states limitations"] = bool(re.search(r"\blimitations?\b|\bmissing evidence\b", report, re.I))
    return {"passed": sum(checks.values()), "total": len(checks), "checks": checks,
            "note": "Surface requirement checklist only; it does not establish factual correctness."}


def evaluate_run(run_id, provider=None):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    state = wf.get_run(run_id)
    if state["status"] != "completed" or not state["tasks"][-1].get("output"):
        raise ValueError("Evaluate a completed run with a generated report")
    selected = provider or state["provider"]
    if selected == "none":
        raise ValueError("A one-call language-model baseline needs Gemini, Groq, or Ollama")
    evidence = [task["output"] for task in state["tasks"]
                if task["tool"] in {"read", "search", "chart"} and task.get("output")]
    prompt = ("You are one generalist research assistant. Write a structured Markdown report for the objective using "
              "only these already-executed, approved tool outputs. Cite PDF filenames and pages; separate Crossref metadata "
              "from uploaded papers; state missing evidence. Do not invent results or treat class counts as sample sizes. "
              "This is a one-call baseline with no planner, specialist comparison agent, or reviewer.\nOBJECTIVE: "
              + state["objective"] + "\nEVIDENCE:\n" + wf.evidence_context(evidence, 24000, state["objective"]))
    model = wf.model_for(selected)
    started = time.monotonic()
    baseline_report = model.generate(prompt)
    elapsed = round(time.monotonic() - started, 3)
    validation_error = None
    try:
        wf.validate_pdf_page_citations(baseline_report, [item for item in evidence if item["kind"] == "document"])
        wf.validate_doi_provenance(baseline_report, evidence)
        wf.validate_doi_links(baseline_report)
    except ValueError as exc:
        validation_error = str(exc)
    multi_report = state["tasks"][-1]["output"]["markdown"]
    result = {
        "run_id": run_id,
        "method": "One-call generalist synthesis of the same saved read/search/chart outputs versus the saved AgentFlow run. No baseline planning or tool selection is measured.",
        "agentflow": {"status": state["status"], "elapsed_seconds": state.get("metrics", {}).get("elapsed_seconds"),
                      "model_calls": state.get("metrics", {}).get("model_calls"),
                      "provider_tokens": state.get("metrics", {}).get("provider_tokens"),
                      "format_checklist": format_checklist(multi_report, state)},
        "single_call_baseline": {"elapsed_seconds": elapsed, "model_calls": 1,
                                 "provider_tokens": (getattr(model, "last_usage", None) or {}).get("total_tokens"),
                                 "provenance_check_passed": validation_error is None,
                                 "provenance_error": validation_error,
                                 "format_checklist": format_checklist(baseline_report, state)},
        "limitations": ["The checklist measures visible requirements, not scientific accuracy.",
                        "The baseline receives pre-executed tools and does not test tool selection.",
                        "One case is a demonstration, not statistical evidence of superiority."],
    }
    (wf.ROOT / ("evaluation-" + run_id + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (wf.ROOT / ("baseline-" + run_id + ".md")).write_text(baseline_report, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="ID of a completed AI run")
    parser.add_argument("--provider", choices=["gemini", "groq", "ollama"], help="Override the saved provider")
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run_id, args.provider), indent=2))


if __name__ == "__main__":
    main()
