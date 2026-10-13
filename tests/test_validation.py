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
from agentflow.workflow import validate_pdf_page_citations


def test_pdf_page_citation_range():
    document = {"name": "paper.pdf", "pages": [{"page": n, "text": ""} for n in range(1, 14)]}
    validate_pdf_page_citations("Finding (paper.pdf, pp. 3-7).", [document])
    import pytest
    with pytest.raises(ValueError, match="nonexistent PDF page 30"):
        validate_pdf_page_citations("Finding (paper.pdf, pp. 1, 5, 30).", [document])


def test_printed_article_page_range():
    import pytest
    pages = [{"page": n, "text": "Journal\n" + str(n + 21) + "\n"} for n in range(1, 14)]
    document = {"name": "paper.pdf", "pages": pages}
    validate_pdf_page_citations("Citation, 22–34 (paper.pdf, p. 1)", [document])
    with pytest.raises(ValueError, match="ends at 34"):
        validate_pdf_page_citations("Citation, 22–35 (paper.pdf, p. 1)", [document])
    pages[0]["text"] += "doi:10.1234/paper.1\n"
    with pytest.raises(ValueError, match="ends at 34"):
        validate_pdf_page_citations("Citation, 22–35. DOI: 10.1234/paper.1", [document])


def test_unrelated_dois_cannot_be_attributed_to_paper():
    import pytest
    document = {"name": "paper.pdf", "pages": [{"page": 1, "text": "doi:10.1234/paper.1"}]}
    with pytest.raises(ValueError, match="mixed with unrelated"):
        validate_pdf_page_citations("DOI/URL: 10.1234/paper.1 and 10.9999/other", [document])
