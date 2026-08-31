from agentflow.models import AgentInstance, TaskResult, TaskSpec, TaskStatus
from agentflow.validation import validate_task_result


def test_validation_accepts_completed_dependencies():
    task = TaskSpec(id="report", objective="write", role="Writer", depends_on=["compare"])
    result = TaskResult(task_id="report", status=TaskStatus.COMPLETED, agent=AgentInstance(role="Writer", objective="write"), output={"text": "ok"})
    assert validate_task_result(task, result, {"compare"}) == []


def test_validation_reports_missing_dependency():
    task = TaskSpec(id="report", objective="write", role="Writer", depends_on=["compare"])
    result = TaskResult(task_id="report", status=TaskStatus.COMPLETED, agent=AgentInstance(role="Writer", objective="write"), output={"text": "ok"})
    assert "dependencies incomplete: compare" in validate_task_result(task, result, set())
