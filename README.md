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

The baseline now includes real text/PDF extraction through `agentflow.document_tools.read_document`, structured-data inspection through `agentflow.data_tools.inspect_table`, Crossref-backed scholarly source retrieval through `agentflow.research_tools.CrossrefSearchProvider`, deterministic result validation through `agentflow.validation.validate_task_result`, a readable workflow-card interface, optional local Ollama planning through `agentflow.smart_planner.SmartPlanner`, a Gemini adapter through `agentflow.gemini_ai.GeminiProvider`, and Markdown report generation. When no local model is available, the application transparently uses the deterministic fallback planner.

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

1. Baseline graph, registry, executor and API. (complete)
2. Document ingestion and extraction tools. (complete)
3. Structured-data inspection tools. (complete)
4. Research-source retrieval and citation records. (complete)
5. Deterministic validation foundation. (complete)
6. Hybrid validation, checkpoints and branch-level recovery.

## Optional free local AI

Install Ollama from `https://ollama.com`, start it, and download a model such as `llama3.2:3b`. Then set `AGENTFLOW_OLLAMA_MODEL` if a different local model is desired. No paid API key is required.

Gemini can be called through `POST /ai/generate` when `GEMINI_API_KEY` is configured locally. The key is never stored in source control.
6. End-to-end report generation and evaluation harness.
