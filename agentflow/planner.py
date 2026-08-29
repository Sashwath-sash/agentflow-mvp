from .models import TaskGraph, TaskSpec


class Planner:
    """Deterministic baseline planner; an LLM planner can replace this later."""

    def build_graph(self, objective: str) -> TaskGraph:
        objective = objective.strip()
        if not objective:
            raise ValueError("objective must not be empty")
        tasks = [
            TaskSpec(id="extract_documents", objective="Extract relevant document content", role="Document Analysis Agent", required_capabilities=["document.read"]),
            TaskSpec(id="research_sources", objective="Find and record relevant academic sources", role="Academic Web Research Agent", required_capabilities=["web.search"]),
            TaskSpec(id="compare_findings", objective="Compare document evidence with researched findings", role="Document Comparison Agent", required_capabilities=["document.compare"], depends_on=["extract_documents", "research_sources"]),
            TaskSpec(id="draft_report", objective="Draft a structured report from validated findings", role="Report-Writing Agent", required_capabilities=["report.write"], depends_on=["compare_findings"]),
        ]
        return TaskGraph(objective=objective, tasks=tasks)
