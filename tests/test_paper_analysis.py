from agentflow.paper_analysis import compare_documents, extractive_summary


def test_comparison_selects_page_linked_evidence_for_each_paper():
    docs = [
        {"name": "first.pdf", "text": "Agents coordinate document review and cite source pages. " * 3,
         "pages": [{"page": 2, "text": "Agents coordinate document review and cite source pages. "
                    "The workflow extracts paper evidence before writing a report."}]},
        {"name": "second.pdf", "text": "Agents coordinate research review and compare source pages. " * 3,
         "pages": [{"page": 5, "text": "Agents coordinate research review and compare source pages. "
                    "The workflow checks paper evidence before publishing a report."}]},
    ]
    result = compare_documents(docs, "Compare how agents review paper evidence")
    assert result["kind"] == "comparison"
    assert len(result["papers"]) == 2
    assert result["papers"][0]["evidence"][0]["page"] == 2
    assert result["papers"][1]["evidence"][0]["page"] == 5
    assert "agents" in result["shared_terms"]
    assert "not semantic" in result["method"]


def test_extractive_summary_prefers_abstract_and_keeps_page_reference():
    document = {"name": "paper.pdf", "pages": [{"page": 1, "text":
        "Title\nAbstract\nThe study interviewed twenty learners about mobile devices. "
        "Results showed both deliberate and ad hoc learning practices.\nKeywords: learning, mobile\n"
        "1. Introduction\n" + "Unrelated background sentence. " * 100}]}
    summary = extractive_summary(document, "summarise")
    assert len(summary) == 2
    assert all(item["page"] == 1 for item in summary)
    assert "Results showed" in summary[1]["text"]
