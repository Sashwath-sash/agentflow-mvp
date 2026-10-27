import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pymupdf
from fastapi.testclient import TestClient
from agentflow.api import app
from agentflow import workflow as wf
from agentflow.research_tools import ResearchSource


def test_upload_execute_pdf_csv_excel_report_and_checkpoint():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory)/"state.db"):
            client = TestClient(app)
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_text((72,72), "Paper A: agents use tools and validate evidence.")
            content = pdf.tobytes()
            pdf.close()
            ids=[]
            for name, data in [("paper.pdf",content),("second.txt",b"Paper B: agents use tools for research."),
                               ("data.csv",b"name,score\na,5\nb,\n")]:
                r=client.post("/upload",files={"file":(name,data)})
                assert r.status_code==200,r.text
                ids.append(r.json()["id"])
            excel=io.BytesIO()
            pd.DataFrame({"topic":["a"],"score":[5]}).to_excel(excel,index=False)
            r=client.post("/upload",files={"file":("table.xlsx",excel.getvalue())})
            assert r.status_code==200,r.text
            ids.append(r.json()["id"])
            r=client.post("/execute",json={"objective":"Compare papers and inspect data","file_ids":ids})
            assert r.status_code==200,r.text
            state=client.get("/runs/"+r.json()["id"]).json()
            assert state["status"]=="completed",state
            report=state["tasks"][-1]["output"]["markdown"]
            assert "Paper A" in report and "Rows: 2" in report
            assert "Shared terms" in report and "NaN" not in report
            assert client.get(state["tasks"][-1]["output"]["download"]).status_code==200
            charts=next(t["output"]["charts"] for t in state["tasks"] if t["tool"]=="chart")
            assert client.get(charts[0]).content.startswith(b"\x89PNG")
            assert wf.get_run(state["id"])["status"]=="completed"


def test_rejects_bad_pdf_and_empty_input():
    client=TestClient(app)
    assert client.post("/upload",files={"file":("broken.pdf",b"bad pdf")}).status_code==400
    assert client.post("/execute",json={"objective":"summarize"}).status_code==400
    assert client.post("/upload",files={"file":("bad.exe",b"bad")}).status_code==400


def test_zero_missing_values_chart_is_explicit_not_blank():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"):
            file = wf.add_file("complete.csv", b"model,score\na,80\nb,90\n")
            state = wf.run_workflow("Chart missing values in the data", [file["id"]])
            chart = next(task["output"] for task in state["tasks"] if task["tool"] == "chart")
            assert chart["messages"] == ["No missing values in 2 rows and 2 columns."]
            assert (Path(directory) / chart["charts"][0].rsplit("/", 1)[-1]).is_file()


def test_csv_metric_values_are_plotted_and_included_in_report():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"):
            file = wf.add_file("models.csv", b"model,top1,top5\nAlpha,71.4,90.2\nBeta,76.1,92.5\n")
            state = wf.run_workflow("Plot top1 by model and summarize the results", [file["id"]])
            assert state["status"] == "completed"
            chart = next(task["output"] for task in state["tasks"] if task["tool"] == "chart")
            assert "top1 by model" in chart["messages"][0]
            assert "models.csv" in chart["messages"][0]
            assert "missing values" not in chart["messages"][0].lower()
            report = next(task["output"]["markdown"] for task in state["tasks"] if task["tool"] == "report")
            assert "71.4" in report and "76.1" in report
            assert "![Data chart]" in report


def test_crossref_uses_uploaded_paper_titles_and_labels_matches():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"):
            ids = []
            for title in ("Rethinking the Inception Architecture for Computer Vision", "Going deeper with convolutions"):
                pdf = pymupdf.open()
                pdf.new_page().insert_text((72, 72), title + "\nAuthor Name\nAbstract: research evidence")
                ids.append(wf.add_file(title + ".pdf", pdf.tobytes())["id"])
                pdf.close()
            source = ResearchSource("Rethinking the Inception Architecture for Computer Vision", ["Author"], 2016,
                                    "10.1000/inception", "https://doi.org/10.1000/inception")
            unrelated = ResearchSource("Going deeper with Image Transformers", ["Author"], 2021,
                                       "10.1000/transformer", "https://doi.org/10.1000/transformer")
            with patch.object(wf.CrossrefSearchProvider, "search", return_value=[source, unrelated]) as search:
                result = wf.execute_tool({"tool": "search", "args": {"query": "unrelated residual network"}}, [], {"file_ids": ids})
            query = search.call_args.args[0]
            assert "Inception Architecture" in query and "Going deeper" in query
            assert result["sources"][0]["relation_to_upload"] == "uploaded_paper_metadata"
            assert len(result["sources"]) == 1


def test_provider_failure_and_retry_preserves_extraction():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf,"ROOT",Path(directory)),patch.object(wf,"DB",Path(directory)/"state.db"):
            file=wf.add_file("paper.txt",b"Research evidence: tool accuracy was 80 percent.")
            with patch.object(wf.GeminiProvider,"generate",side_effect=RuntimeError("quota exhausted")):
                state=wf.run_workflow("Summarize",[file["id"]],provider="gemini")
            assert state["status"]=="incomplete"
            assert state["tasks"][-1]["status"]=="failed"
            with patch.object(wf.GeminiProvider,"generate",side_effect=["Evidence-based result.",
                '{"passed":true,"coverage":1,"issues":[]}']) as model:
                fixed=wf.run_workflow("",[],run_id=state["id"])
                assert model.call_count==2
            assert fixed["status"]=="completed"
            assert fixed["tasks"][0]["attempts"]==1
            assert fixed["tasks"][-1]["attempts"]==2


def test_interrupted_run_can_be_resumed():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf,"ROOT",Path(directory)),patch.object(wf,"DB",Path(directory)/"state.db"):
            file=wf.add_file("paper.txt",b"Evidence for checkpoint test")
            state=wf.run_workflow("Summarize",[file["id"]])
            state["status"]="running"
            state["tasks"][-1]["status"]="running"
            wf.save_run(state)
            wf.recover_interrupted()
            assert wf.get_run(state["id"])["status"]=="incomplete"
            fixed=wf.run_workflow("",[],run_id=state["id"])
            assert fixed["status"]=="completed"
            assert fixed["tasks"][0]["attempts"]==1


def test_two_pdf_offline_report_contains_page_linked_comparison():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory)/"state.db"):
            client = TestClient(app)
            ids = []
            for name, sentence in [
                ("method-a.pdf", "The agents compare cited paper evidence with deterministic validation."),
                ("method-b.pdf", "The agents review cited paper evidence with structured validation."),
            ]:
                pdf = pymupdf.open()
                page = pdf.new_page()
                page.insert_text((72, 72), sentence)
                response = client.post("/upload", files={"file": (name, pdf.tobytes())})
                pdf.close()
                assert response.status_code == 200, response.text
                ids.append(response.json()["id"])
            response = client.post("/execute", json={
                "objective": "Compare validation in the two papers", "file_ids": ids, "provider": "none",
            })
            assert response.status_code == 200, response.text
            state = client.get("/runs/" + response.json()["id"]).json()
            assert state["status"] == "completed", state
            comparison = next(task["output"] for task in state["tasks"] if task["tool"] == "compare")
            assert len(comparison["papers"]) == 2
            report = state["tasks"][-1]["output"]["markdown"]
            assert "### method-a.pdf" in report and "### method-b.pdf" in report
            assert "Page 1:" in report and "not semantic" in report

