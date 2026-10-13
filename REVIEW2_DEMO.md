# AgentFlow Review 2 demo notes

## What the system does

AgentFlow accepts research papers, tables, and an objective. A LangGraph controller runs a validated dependency graph of runtime-configured roles. Approved Python functions read PDFs/CSV files, retrieve Crossref metadata, compare documents, create a data-quality chart, and write a report. A language-model reviewer assesses coverage; deterministic checks validate schemas, generated files, PDF page ranges, and DOI provenance/link consistency. These checks do **not** prove every scientific claim true.

The roles are not separately trained models. The planner, comparison worker, report writer, reviewer, and replanner use role-specific instructions with the selected provider model. File reading, metadata retrieval, and charting use fixed Python tools.

## Demonstration

1. Start the local server with `python -m uvicorn agentflow.api:app --host 127.0.0.1 --port 8001` and open <http://127.0.0.1:8001/>. Do not launch a second server on the same port.
2. Upload `1512.00567v3.pdf`, `1409.4842v1.pdf`, and `imagenet-model-results.csv`. Adding files in separate selections is supported. Confirm all three appear before running.
3. Check **Search Crossref**. Select a provider only after **Test connection** succeeds; a connection test does not guarantee sufficient quota. Offline mode is available for a quota-free extraction/comparison demo but does not perform semantic AI review.
4. Objective: “Compare the two uploaded papers’ methods and numerical findings with filename and page citations. Analyze the CSV’s dimensions, missing values and numerical statistics, make a missing-values chart, retrieve Crossref metadata, and write a report that separates uploaded-paper metadata from candidate related records. Do not invent missing evidence or treat class counts as sample sizes.”
5. Show the seven task outputs, agent roster, hand-offs, dependency graph, chart, and downloadable Markdown report. The CSV has no missing values, so the graphic explicitly says this instead of displaying an empty axis.

## Measured example and honest interpretation

One completed Gemini run on the three files executed seven tasks in 21.0 seconds, with four model calls and 26,808 provider-reported tokens. Its visible-requirement checklist passed 7/7. A one-call generalist synthesis baseline, given the same pre-executed read/search/chart outputs, took 17.4 seconds and 10,124 tokens and also passed 7/7 visible requirements. Its report contained a mismatched DOI hyperlink caught during a later deterministic provenance audit. The earlier AgentFlow report passed that audit. The current application rejects such DOI links in future runs.

This **single case does not establish that multi-agent orchestration is better**. It shows an integration and its cost: more role-specific stages and checks, but extra latency and tokens. The checklist measures visible requirements, not scientific accuracy. The baseline did not perform its own planning or tool selection, so this is not a complete tool-selection comparison. The machine-readable record is `.agentflow/evaluation-5668a4a887fb4b80b6b4f899fdc296aa.json`.

## Boundaries to state in the viva

- Crossref returns metadata candidates, not reviewed full text. Exact metadata matches for uploaded papers are labelled separately.
- Language-model evidence is bounded; omitted PDF passages are not evidence. Review cited pages manually before relying on a conclusion.
- ImageNet class count is not a sample size. Benchmark error rates may differ by validation/test split, crop strategy, and ensemble configuration.
- OCR for scanned PDFs, durable external job queues, public deployment, and a statistically powered multi-case baseline study are not implemented.
- Free online-provider quota or network availability cannot be guaranteed during the presentation. Have offline mode and the saved completed run ready.
