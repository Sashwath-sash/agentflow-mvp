# AgentFlow

Dynamic Multi-Agent Task Orchestration System for Academic Research and Document Analysis

AgentFlow is a tool-bounded framework that turns a high-level research objective into a dependency-aware execution plan. This first baseline focuses on the deterministic foundation: task models, capability-based tool assignment, dependency ordering, and an inspectable execution result.

## Baseline scope

- Accept a research objective.
- Build a small dependency-aware task graph.
- Configure task-specific agent instances from reusable templates.
- Assign only approved tools to each task.
- Execute deterministic placeholder workers and retain structured state.
- Expose the workflow through a small FastAPI endpoint.

LLM calls, PDF extraction, web retrieval, semantic review, checkpoint persistence, and branch-level recovery are planned as subsequent milestones. They are intentionally not represented as completed functionality in this baseline.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn agentflow.api:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## Test

```powershell
pytest
```

## Milestones

1. Baseline graph, registry, executor and API.
2. Document ingestion and extraction tools.
3. Research-source retrieval and citation records.
4. Hybrid validation and branch-level recovery.
5. End-to-end report generation and evaluation harness.
