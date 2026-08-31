from __future__ import annotations

from .models import TaskResult, TaskSpec


def validate_task_result(task: TaskSpec, result: TaskResult, completed_ids: set[str]) -> list[str]:
    """Run deterministic checks before a worker result is accepted."""
    errors: list[str] = []
    missing = [dependency for dependency in task.depends_on if dependency not in completed_ids]
    if missing:
        errors.append(f"dependencies incomplete: {', '.join(missing)}")
    if result.task_id != task.id:
        errors.append("result task id does not match task")
    if not result.agent.role or not result.agent.objective:
        errors.append("agent role and objective are required")
    if not result.output:
        errors.append("worker output must not be empty")
    return errors
