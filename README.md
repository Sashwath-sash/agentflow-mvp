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
2. Upload up to 10 PDFs, TXT/Markdown documents, CSV or Excel files. Scanned PDF pages use local OCR.
3. Optionally enable scholarly source retrieval.
4. Select Groq AI, local evidence processing, Gemini, or Ollama.
5. Run and download the report. Failed tasks show the cause. Retry preserves completed evidence.

For a quota-free demo, leave **Reliable offline evidence comparison** selected. With two
text-based PDFs it produces a page-linked evidence comparison and downloadable report.
This is deterministic extraction and sentence ranking, not AI interpretation; check
the cited pages before presenting a conclusion. A saved Groq key is not used unless
Groq is explicitly selected.

Save an online provider key once in the page, then use **Test connection**. Keys are persisted in the ignored local .env file and loaded after restarts; the application does not silently switch models or providers. Free-plan rate limits still apply. Gemini and Groq have both completed local research runs, but a successful connection test does not guarantee enough quota for a full run. Ollama requires an installed runtime/model. No keys belong in Git.

For the review demo, attach two text-selectable papers and an optional small CSV. The file picker retains files added in separate selections. If the objective asks for Crossref results, tick **Search Crossref**. Start a new run after changing files or search; retry uses the previous run's fixed inputs. Show the agent roster, dependency graph, evidence panels, and downloadable report. A CSV with zero missing values now produces an explicit no-missing-values graphic rather than an empty axis.

## Implemented and tested

- PDF extraction with page references, UTF-8 document reading.
- Local RapidOCR for scanned pages without a text layer (first 12 such pages per PDF).
- Diagram Analysis Agent for substantial captioned architecture/flow figures, including raster and vector figures. It saves figure images and records OCR labels, embedded PDF text, and left/centre/right label placement with page provenance. It does not verify arrows or infer causal links from image pixels.
- Persistent uploads connected to actual execution.
- CSV/XLSX/XLS inspection, missing values, numeric statistics and JSON-safe nulls.
- Crossref scholarly metadata retrieval using uploaded PDF title hints when available; exact-paper metadata is labelled separately from unreviewed candidate records. This is not general web search or full-text access.
- Lexical document comparison in local mode; Groq can compare documents semantically, synthesize findings and review the report.
- Automatic, bounded table profiling and up to four PNG charts per uploaded dataset: category comparisons, dated trends, distributions, numeric relationships, and missingness when relevant. The chart panel states the selected columns, aggregation, sampling, and why other columns were not plotted. Markdown evidence reports include the charts and the same selection notes.
- LangGraph controller with runtime agent configurations and dependency scheduling.
- AI-generated task graphs validated by Pydantic, file/tool permission checks and cycle detection. No-AI mode uses explicit input-aware rules.
- Bounded automatic failed-branch retries and model-guided instruction repair; successful task outputs are reused.
- AI semantic document comparison and structured report review. Failed semantic reviews trigger bounded revision, not false success. Generated DOIs must occur in the uploaded documents or retrieved metadata.
- Execution metrics, model-call counts, provider-reported tokens when available, reviewer-estimated coverage, and recent-run history.
- SQLite task checkpoints and stored outputs; successful tasks are reused on retry.
- Background execution with UI polling, visible tool results, safe text rendering.
- Persistent local Groq and Gemini configuration; secrets excluded from Git.

The active application uses agentflow/workflow.py. Earlier planner/runtime prototype modules remain for historical compatibility and are not used by /execute.

## Remaining scope

Live Groq operation was verified with two synthetic PDFs, a generated task graph, semantic comparison, review, and a downloadable report. Rate limits and external availability remain outside the application.
Semantic review requires an available language model; it is not a factual guarantee. Its coverage percentage is a model estimate, not an externally verified accuracy score. Inception-style benchmarks can differ by split, crop count and ensemble configuration; compare them with those qualifiers.
Docker and arbitrary generated-code execution are intentionally excluded at the user's request. Data tools use fixed approved Pandas/Matplotlib functions.
OCR is limited to the first 12 pages per PDF without a selectable text layer, and recognition may misread small or rotated labels. Complex plots, photographs, and exact flowchart arrow connectivity are outside the visual worker's scope. Production authentication/deployment are not implemented. A one-call synthesis baseline utility exists (`python -m agentflow.evaluate RUN_ID`), but a statistically powered multi-case comparison, including baseline tool selection, is not implemented. The utility uses one provider call and may be subject to quota.

Chart discovery is rule-based and limited to four justified views per dataset, not a promise to explain every column or infer causation. Name a specific column in the objective to prioritize it. Repeated dates or categories use a stated sum or mean; very large categorical sets and scatter plots are bounded and disclose their selection.
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

For a local visual smoke test with real papers:

```powershell
python tests/live_visual_smoke.py "C:\path\to\paper.pdf"
```

