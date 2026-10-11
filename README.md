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
4. Select Groq AI, local evidence processing, Gemini, or Ollama.
5. Run and download the report. Failed tasks show the cause. Retry preserves completed evidence.

Groq is the currently tested online AI provider. Save its key once in the page, then use **Test connection**. The key is persisted in the ignored local .env file and loaded after restarts. The explicit primary model is `openai/gpt-oss-20b`; the application does not silently switch models or providers. Free plan rate limits still apply. Gemini is also selectable but its currently configured project returned a zero quota error. Ollama requires an installed runtime/model. No keys belong in Git.

## Implemented and tested

- PDF extraction with page references, UTF-8 document reading.
- Persistent uploads connected to actual execution.
- CSV/XLSX/XLS inspection, missing values, numeric statistics and JSON-safe nulls.
- Crossref scholarly metadata retrieval (not general web search or full-text access).
- Lexical document comparison in local mode; Groq can compare documents semantically, synthesize findings and review the report.
- Data-quality PNG charts, Markdown evidence reports and downloads.
- LangGraph controller with runtime agent configurations and dependency scheduling.
- AI-generated task graphs validated by Pydantic, file/tool permission checks and cycle detection. No-AI mode uses explicit input-aware rules.
- Bounded automatic failed-branch retries and model-guided instruction repair; successful task outputs are reused.
- AI semantic document comparison and structured report review. Failed semantic reviews trigger bounded revision, not false success.
- Execution metrics, model-call counts, provider-reported tokens when available, reviewer-estimated coverage, and recent-run history.
- SQLite task checkpoints and stored outputs; successful tasks are reused on retry.
- Background execution with UI polling, visible tool results, safe text rendering.
- Persistent local Groq and Gemini configuration; secrets excluded from Git.

The active application uses agentflow/workflow.py. Earlier planner/runtime prototype modules remain for historical compatibility and are not used by /execute.

## Remaining scope

Live Groq operation was verified with two synthetic PDFs, a generated task graph, semantic comparison, review, and a downloadable report. Rate limits and external availability remain outside the application.
Semantic review requires an available language model; it is not a factual guarantee.
Docker and arbitrary generated-code execution are intentionally excluded at the user's request. Data tools use fixed approved Pandas/Matplotlib functions.
PDF OCR, production authentication/deployment, and an empirical single-agent baseline evaluation are not implemented.
Branch recovery revises failed-task instructions, not graph topology or permissions. Crossref retrieves scholarly metadata, not full-text web research.
Document report excerpts cover the first three pages (1800 characters each). AI context distributes a bounded budget across all pages/documents,
marks truncation, and is capped at 24000 characters. It is not exhaustive full-document analysis. Review original papers before academic use.
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

