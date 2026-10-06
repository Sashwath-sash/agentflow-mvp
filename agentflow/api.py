from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .planner import Planner
from .gemini_ai import GeminiError, GeminiProvider
from .research_tools import CrossrefSearchProvider, ResearchSearchError
from .runtime import RuntimeExecutor
from .reporting import render_markdown_report
from .smart_planner import SmartPlanner
from .data_tools import DataReadError, inspect_table
from .document_tools import DocumentReadError, read_document
from pathlib import Path
import tempfile
import uuid

app = FastAPI(title="AgentFlow", version="0.1.0")
planner = Planner()
smart_planner = SmartPlanner()
executor = RuntimeExecutor()
research_provider = CrossrefSearchProvider()
gemini_provider = GeminiProvider()

PAGE = """<!doctype html>
<html><head><title>AgentFlow</title><style>
body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#172033}
h1{margin-bottom:4px}.sub{color:#64748b}textarea{width:100%;min-height:110px;padding:12px;font-size:16px;border:1px solid #cbd5e1;border-radius:8px}
button{margin-top:12px;padding:10px 18px;background:#2563eb;color:white;border:0;border-radius:7px;font-size:15px;cursor:pointer}
pre{background:#f1f5f9;padding:16px;border-radius:8px;white-space:pre-wrap}.card{margin-top:24px}.task{border:1px solid #dbe3ef;border-radius:9px;padding:14px;margin:10px 0;background:white}.task h3{margin:0 0 6px}.meta{color:#475569;font-size:14px;margin:4px 0}.ok{color:#15803d;font-weight:bold}
</style></head><body><h1>AgentFlow</h1><div class="sub">Dynamic Multi-Agent Task Orchestration System</div>
<div class="card"><label for="objective"><b>What should AgentFlow do?</b></label>
<textarea id="objective">Compare two research papers and write a report</textarea><br><button onclick="run()">Run workflow</button></div>
<div class="card"><h2>Check an input file</h2><input id="file" type="file" accept=".pdf,.txt,.md,.csv,.xlsx,.xls"/><button onclick="checkFile()">Upload and check</button><pre id="fileResult">No file checked yet.</pre></div>
<div class="card"><h2>Workflow result</h2><div id="result">Enter an objective and run the workflow.</div></div>
<script>async function run(){const objective=document.getElementById('objective').value;const out=document.getElementById('result');out.textContent='Planning and executing...';try{const r=await fetch('/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({objective})});const data=await r.json();if(!r.ok){out.textContent=data.detail||'Request failed';return}out.innerHTML='<p><b>Planning mode:</b> '+data.planning_mode+'</p><p><b>Execution order:</b> '+data.order.join(' → ')+'</p>'+data.results.map(x=>'<div class="task"><h3>'+x.agent.role+' <span class="ok">✓ '+x.status+'</span></h3><div class="meta"><b>Task:</b> '+x.task_id+' — '+x.agent.objective+'</div><div class="meta"><b>Approved tools:</b> '+(x.agent.approved_tools.join(', ')||'none')+'</div><div class="meta"><b>Dependencies:</b> '+(x.output.dependencies.join(', ')||'none')+'</div><div class="meta ok">Validation: '+(x.output.validation||'not passed')+'</div></div>').join('')+'<details><summary>Generated Markdown report</summary><pre>'+data.report_markdown+'</pre></details>'}catch(e){out.textContent='Request failed: '+e}}</script>
</script><script>async function checkFile(){const file=document.getElementById('file').files[0];const out=document.getElementById('fileResult');if(!file){out.textContent='Choose a file first.';return}const form=new FormData();form.append('file',file);const structured=/\.(csv|xlsx|xls)$/i.test(file.name);out.textContent='Checking file...';try{const r=await fetch(structured?'/data/inspect':'/documents/inspect',{method:'POST',body:form});out.textContent=JSON.stringify(await r.json(),null,2)}catch(e){out.textContent='Check failed: '+e}}</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


class PlanRequest(BaseModel):
    objective: str


class ResearchRequest(BaseModel):
    query: str
    limit: int = 5


class AIRequest(BaseModel):
    prompt: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plan")
def plan(request: PlanRequest):
    try:
        return planner.build_graph(request.objective)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/execute")
def execute(request: PlanRequest):
    try:
        graph = smart_planner.build_graph(request.objective)
        result = executor.execute(graph)
        return {**result.model_dump(), "planning_mode": smart_planner.last_mode, "report_markdown": render_markdown_report(result, smart_planner.last_mode)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/research")
def research(request: ResearchRequest):
    try:
        return {"query": request.query, "sources": research_provider.search(request.query, request.limit)}
    except ResearchSearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/ai/generate")
def ai_generate(request: AIRequest):
    try:
        return {"model": gemini_provider.model, "text": gemini_provider.generate(request.prompt)}
    except GeminiError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    allowed = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Supported files: PDF, TXT, MD, CSV, XLSX, XLS")
    target = Path(tempfile.gettempdir()) / f"agentflow-{uuid.uuid4().hex}{suffix}"
    target.write_bytes(await upload.read())
    return target


@app.post("/documents/inspect")
async def inspect_document(file: UploadFile = File(...)):
    target = await _save_upload(file)
    try:
        text = read_document(target)
        return {"filename": file.filename, "type": "document", "characters": len(text), "preview": text[:1000]}
    except DocumentReadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        target.unlink(missing_ok=True)


@app.post("/data/inspect")
async def inspect_data(file: UploadFile = File(...)):
    target = await _save_upload(file)
    try:
        return {"filename": file.filename, "type": "structured_data", "summary": inspect_table(target)}
    except DataReadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        target.unlink(missing_ok=True)
