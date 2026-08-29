from agentflow.planner import Planner
from agentflow.registry import ToolRegistry
from agentflow.runtime import RuntimeExecutor


def test_planner_builds_dependency_graph():
    graph = Planner().build_graph("Compare two papers and write a report")
    assert [task.id for task in graph.tasks] == ["extract_documents", "research_sources", "compare_findings", "draft_report"]
    assert graph.tasks[2].depends_on == ["extract_documents", "research_sources"]


def test_executor_orders_dependencies_and_configures_tools():
    result = RuntimeExecutor().execute(Planner().build_graph("Research a topic"))
    assert result.order == ["extract_documents", "research_sources", "compare_findings", "draft_report"]
    assert result.results[0].agent.approved_tools == ["pdf_text_reader"]
    assert result.results[-1].agent.approved_tools == ["report_writer"]


def test_registry_does_not_assign_unrequested_tools():
    assert [tool.name for tool in ToolRegistry().tools_for(["web.search"])] == ["academic_web_search"]
