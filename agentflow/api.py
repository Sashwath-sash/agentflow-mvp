from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .planner import Planner
from .gemini_ai import GeminiError, GeminiProvider
from .research_tools import CrossrefSearchProvider, ResearchSearchError
from .runtime import RuntimeExecutor
from .reporting import render_markdown_report
from .smart_planner import SmartPlanner

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
<div class="card"><h2>Workflow result</h2><div id="result">Enter an objective and run the workflow.</div></div>
<script>async function run(){const objective=document.getElementById('objective').value;const out=document.getElementById('result');out.textContent='Planning and executing...';try{const r=await fetch('/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({objective})});const data=await r.json();if(!r.ok){out.textContent=data.detail||'Request failed';return}out.innerHTML='<p><b>Execution order:</b> '+data.order.join(' → ')+'</p>'+data.results.map(x=>'<div class="task"><h3>'+x.agent.role+' <span class="ok">✓ '+x.status+'</span></h3><div class="meta"><b>Task:</b> '+x.task_id+' — '+x.agent.objective+'</div><div class="meta"><b>Approved tools:</b> '+(x.agent.approved_tools.join(', ')||'none')+'</div><div class="meta"><b>Dependencies:</b> '+(x.output.dependencies.join(', ')||'none')+'</div><div class="meta ok">Validation: '+(x.output.validation||'not passed')+'</div></div>').join('')}catch(e){out.textContent='Request failed: '+e}}</script>
</body></html>"""


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
