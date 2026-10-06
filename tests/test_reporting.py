from agentflow.models import AgentInstance, ExecutionResult, TaskResult, TaskStatus
from agentflow.reporting import render_markdown_report


def test_report_contains_workflow_summary():
    result = ExecutionResult(objective="test", order=["one"], results=[TaskResult(task_id="one", status=TaskStatus.COMPLETED, agent=AgentInstance(role="Worker", objective="do it", approved_tools=["tool"]), output={"validation": "passed"})])
    report = render_markdown_report(result, "deterministic-fallback")
    assert "# AgentFlow Workflow Report" in report
    assert "Worker" in report
    assert "deterministic-fallback" in report
