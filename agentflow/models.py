from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskSpec(BaseModel):
    id: str
    objective: str
    role: str
    required_capabilities: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class TaskGraph(BaseModel):
    objective: str
    tasks: list[TaskSpec]


class ToolSpec(BaseModel):
    name: str
    capability: str
    description: str


class AgentInstance(BaseModel):
    role: str
    objective: str
    approved_tools: list[str] = Field(default_factory=list)


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    agent: AgentInstance
    output: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    objective: str
    order: list[str]
    results: list[TaskResult]
