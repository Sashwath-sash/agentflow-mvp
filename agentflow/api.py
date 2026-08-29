from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .planner import Planner
from .runtime import RuntimeExecutor

app = FastAPI(title="AgentFlow", version="0.1.0")
planner = Planner()
executor = RuntimeExecutor()


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
