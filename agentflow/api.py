from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .planner import Planner
from .runtime import RuntimeExecutor

app = FastAPI(title="AgentFlow", version="0.1.0")
planner = Planner()
executor = RuntimeExecutor()

PAGE = """<!doctype html>
<html><head><title>AgentFlow</title><style>
body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#172033}
h1{margin-bottom:4px}.sub{color:#64748b}textarea{width:100%;min-height:110px;padding:12px;font-size:16px;border:1px solid #cbd5e1;border-radius:8px}
button{margin-top:12px;padding:10px 18px;background:#2563eb;color:white;border:0;border-radius:7px;font-size:15px;cursor:pointer}
pre{background:#f1f5f9;padding:16px;border-radius:8px;white-space:pre-wrap}.card{margin-top:24px}
</style></head><body><h1>AgentFlow</h1><div class="sub">Dynamic Multi-Agent Task Orchestration System</div>
<div class="card"><label for="objective"><b>What should AgentFlow do?</b></label>
<textarea id="objective">Compare two research papers and write a report</textarea><br><button onclick="run()">Run workflow</button></div>
<div class="card"><h2>Workflow result</h2><pre id="result">Enter an objective and run the workflow.</pre></div>
<script>async function run(){const objective=document.getElementById('objective').value;const out=document.getElementById('result');out.textContent='Planning and executing...';try{const r=await fetch('/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({objective})});out.textContent=JSON.stringify(await r.json(),null,2)}catch(e){out.textContent='Request failed: '+e}}</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


class PlanRequest(BaseModel):
    objective: str


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
        return executor.execute(planner.build_graph(request.objective))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
