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


def test_provider_failure_and_retry_preserves_extraction():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf,"ROOT",Path(directory)),patch.object(wf,"DB",Path(directory)/"state.db"):
            file=wf.add_file("paper.txt",b"Research evidence: tool accuracy was 80 percent.")
            with patch.object(wf.GeminiProvider,"generate",side_effect=RuntimeError("quota exhausted")):
                state=wf.run_workflow("Summarize",[file["id"]],provider="gemini")
            assert state["status"]=="incomplete"
            assert state["tasks"][-1]["status"]=="failed"
            with patch.object(wf.GeminiProvider,"generate",return_value="Evidence-based result.") as model:
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

