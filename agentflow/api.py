from pathlib import Path
import os
import re
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv, set_key, dotenv_values
from . import workflow as wf
from .gemini_ai import GeminiProvider, GeminiError
from .groq_ai import GroqProvider, GroqError
from contextlib import asynccontextmanager

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_FILE)
@asynccontextmanager
async def lifespan(app):
    wf.recover_interrupted()
    yield


app = FastAPI(title="AgentFlow", version="0.2.0", lifespan=lifespan)


class GeminiSettings(BaseModel):
    key: SecretStr


@app.post("/settings/gemini")
def configure_gemini(settings: GeminiSettings):
    key = settings.key.get_secret_value().strip()
    if not key or len(key) > 500:
        raise HTTPException(400, "Enter a valid Gemini key")
    set_key(str(ENV_FILE), "GEMINI_API_KEY", key)
    os.environ["GEMINI_API_KEY"] = key
    return {"configured": True, "message": "Key saved on this computer. It will load automatically after restarts."}


@app.get("/settings/gemini")
def gemini_status():
    return {"configured": bool(os.getenv("GEMINI_API_KEY")), "saved": bool(dotenv_values(ENV_FILE).get("GEMINI_API_KEY"))}


class RunRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=4000)
    file_ids: list[str] = Field(default_factory=list, max_length=10)
    search: bool = False
    provider: str = "none"
    dynamic: bool = True
    auto_recover: bool = True


@app.post("/settings/groq")
def configure_groq(settings: GeminiSettings):
    key = settings.key.get_secret_value().strip()
    if not key or len(key) > 500 or any(c.isspace() for c in key):
        raise HTTPException(400, "Enter a valid Groq key")
    set_key(str(ENV_FILE), "GROQ_API_KEY", key)
    os.environ["GROQ_API_KEY"] = key
    return {"configured": True, "message": "Groq key saved locally for future restarts. Click Test connection to verify access."}


@app.get("/settings/groq")
def groq_status():
    return {"configured": bool(os.getenv("GROQ_API_KEY")), "saved": bool(dotenv_values(ENV_FILE).get("GROQ_API_KEY")), "model": GroqProvider().model}


@app.post("/settings/groq/test")
def test_groq():
    provider = GroqProvider()
    try:
        provider.generate("Reply with only OK.")
        return {"message": "Groq connection verified. Ready to start a new analysis.", "model": provider.model}
    except GroqError as exc:
        raise HTTPException(503, str(exc)) from None


@app.get("/", response_class=HTMLResponse)
def home():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "pdf": True, "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
            "ai_note": "Configured does not mean provider access has been verified"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read(10 * 1024 * 1024 + 1)
        return wf.add_file(file.filename or "upload", content)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        await file.close()


# Compatibility endpoints now retain uploads for execution.
app.post("/documents/inspect")(upload)
app.post("/data/inspect")(upload)


@app.post("/plan")
def plan(request: RunRequest):
    try:
        return {"tasks": wf.plan(request.objective, request.file_ids, request.search),
                "planning_mode": "input-aware rules"}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@app.post("/execute")
def execute(request: RunRequest, background: BackgroundTasks):
    if request.provider not in {"none", "gemini", "ollama", "groq"}:
        raise HTTPException(400, "Choose none, groq, gemini or ollama")
    if request.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise HTTPException(400, "Save your Groq key in Groq connection settings first")
    if request.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(400, "No saved Gemini key. Save it once in connection settings, or choose local evidence mode.")
    try:
        tasks = wf.plan(request.objective, request.file_ids, request.search)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    run_id = wf.uuid.uuid4().hex
    state = {"id": run_id, "objective": request.objective, "provider": request.provider,
             "tasks": tasks, "status": "queued", "file_ids": request.file_ids, "search": request.search,
             "dynamic": request.dynamic and request.provider != "none", "auto_recover": request.auto_recover,
             "planning_mode": "input-aware rules", "metrics": {}, "events": []}
    wf.save_run(state)
    background.add_task(wf.run_workflow, "", [], run_id=run_id)
    return state


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    try:
        return wf.get_run(run_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


@app.post("/runs/{run_id}/retry")
def retry(run_id: str, background: BackgroundTasks):
    state = get_run(run_id)
    if state["status"] in {"running", "queued"}:
        raise HTTPException(409, "This run is already active")
    if any(t["status"] != "completed" and t["attempts"] >= 3 for t in state["tasks"]):
        raise HTTPException(400, "Retry budget reached. Correct the inputs or provider configuration and start a new run.")
    state["status"] = "queued"
    wf.save_run(state)
    background.add_task(wf.run_workflow, "", [], run_id=run_id)
    return state


@app.get("/artifacts/{name}")
def artifact(name: str):
    if not re.fullmatch(r"[a-f0-9]{32}(?:-\d+)?\.(?:md|png)", name):
        raise HTTPException(404)
    path = wf.ROOT / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=name)


class AIRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=24000)


@app.post("/ai/generate")
def generate(request: AIRequest):
    try:
        provider = GeminiProvider()
        return {"model": provider.model, "text": provider.generate(request.prompt)}
    except GeminiError as exc:
        raise HTTPException(503, str(exc)) from None


@app.get("/capabilities")
def capabilities():
    return {"gemini_configured": bool(os.getenv("GEMINI_API_KEY")), "execution": "Fixed approved Python functions; no generated-code execution",
            "runtime": "LangGraph", "persistence": "SQLite task checkpoints",
            "tools": ["read", "search", "compare", "chart", "report"],
            "limits": {"files": 10, "file_mb": 10, "pdf_pages": 200, "task_attempts": 3, "model_calls": 30}}


@app.get("/runs")
def history():
    with wf.connect() as db:
        rows = db.execute("SELECT state FROM runs ORDER BY rowid DESC LIMIT 30").fetchall()
    states = [wf.json.loads(row[0]) for row in rows]
    return [{"id": s["id"], "objective": s["objective"], "status": s["status"], "metrics": s.get("metrics", {})} for s in states]


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


@app.post("/research")
def research(request: ResearchRequest):
    try:
        return {"sources": [wf.asdict(s) for s in wf.CrossrefSearchProvider().search(request.query, request.limit)]}
    except Exception as exc:
        raise HTTPException(502, str(exc)) from None

