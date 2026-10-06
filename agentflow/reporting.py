from .models import ExecutionResult


def render_markdown_report(result: ExecutionResult, planning_mode: str) -> str:
    lines = [f"# AgentFlow Workflow Report", "", f"**Objective:** {result.objective}", f"**Planning mode:** {planning_mode}", "", "## Execution Summary", ""]
    for item in result.results:
        tools = ", ".join(item.agent.approved_tools) or "none"
        lines.extend([f"### {item.agent.role}", f"- Status: **{item.status.value}**", f"- Task: {item.task_id}", f"- Tools: {tools}", f"- Validation: {item.output.get('validation', 'failed')}", ""])
    return "\n".join(lines)
