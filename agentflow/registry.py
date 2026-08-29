from .models import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools = {
            "document.read": ToolSpec(name="pdf_text_reader", capability="document.read", description="Read PDF or text content."),
            "web.search": ToolSpec(name="academic_web_search", capability="web.search", description="Search for academic web sources."),
            "document.compare": ToolSpec(name="document_comparator", capability="document.compare", description="Compare extracted evidence."),
            "report.write": ToolSpec(name="report_writer", capability="report.write", description="Write a structured report artifact."),
        }

    def tools_for(self, capabilities: list[str]) -> list[ToolSpec]:
        return [self._tools[c] for c in capabilities if c in self._tools]
