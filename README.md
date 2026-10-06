# AgentFlow

Dynamic Multi-Agent Task Orchestration System for Academic Research and Document Analysis Operations.

## Run locally

Use the same Python interpreter for installation and startup:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn agentflow.api:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001.

1. Enter the research objective.
2. Upload up to 10 text-based PDFs, TXT/Markdown documents, CSV or Excel files.
3. Optionally enable scholarly source retrieval.
4. Select local evidence processing, Gemini, or Ollama.
5. Run and download the report. Failed tasks show the cause. Retry preserves completed evidence.

Gemini requires a valid provider key and available quota. Use the password field in the page to configure this server session, or set GEMINI_API_KEY before startup. An ignored .env file is also supported. No keys belong in Git. Ollama requires an installed runtime/model. Choosing an unavailable AI provider fails the report task visibly.

## Implemented and tested

- PDF extraction with page references, UTF-8 document reading.
- Persistent uploads connected to actual execution.
- CSV/XLSX/XLS inspection, missing values, numeric statistics and JSON-safe nulls.
- Crossref scholarly metadata retrieval (not general web search or full-text access).
- Lexical document comparison; Gemini/Ollama can synthesize and review provided evidence.
- Data-quality PNG charts, Markdown evidence reports and downloads.
- Input-dependent task selection, dependency checks, explicit failures and bounded manual retries.
- SQLite task checkpoints and stored outputs; successful tasks are reused on retry.
- Background execution with UI polling, visible tool results, safe text rendering.
- Local Gemini session configuration; secrets excluded from Git.

The active application uses agentflow/workflow.py. Earlier planner/runtime prototype modules remain for historical compatibility and are not used by /execute.

## Remaining scope

This is a working evidence-processing prototype, not the complete originally proposed framework.
Planning uses deterministic input-aware rules, not an LLM-generated task graph or LangGraph.
Semantic review requires an available language model; it is not a factual guarantee.
PDF OCR, arbitrary Python execution/Docker sandboxing, automatic replanning, evaluation
against a single-agent baseline, and production authentication/deployment are not implemented.
Document report excerpts cover the first three pages (1800 characters each). AI evidence is capped
at 24000 characters and explicitly identified as bounded. Review original papers before academic use.
Background tasks are local-process tasks, not a durable job queue.

Uploads, reports and checkpoints are retained under .agentflow/ and ignored by Git.

## Verification

```powershell
python -m pytest -q
python tests/live_smoke.py
```

Live smoke tests require the server on port 8001. Optional browser verification:
install Playwright and use installed Microsoft Edge, then run python tests/browser_smoke.py.
The integration suite uploads PDF, CSV and XLSX, checks computed output and artifact downloads,
rejects malformed uploads, and simulates provider failure/recovery. Mock AI tests do not establish
that a real provider credential works.

