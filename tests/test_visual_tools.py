import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pymupdf
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient

from agentflow import workflow as wf
from agentflow.api import app


def _diagram_png() -> bytes:
    image = Image.new("RGB", (700, 460), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 42)
    draw.rounded_rectangle((170, 35, 530, 145), radius=12, outline="black", width=4)
    draw.text((265, 62), "Input", fill="black", font=font)
    draw.line((350, 146, 350, 265), fill="black", width=5)
    draw.polygon([(340, 250), (360, 250), (350, 275)], fill="black")
    draw.rounded_rectangle((170, 275, 530, 405), radius=12, outline="black", width=4)
    draw.text((250, 310), "Output", fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _pdf_with_figure() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 60), "This paper proposes a simple processing workflow.")
    page.insert_image(pymupdf.Rect(70, 100, 542, 420), stream=_diagram_png())
    page.insert_text((72, 445), "Figure 1: Input to output architecture.")
    content = pdf.tobytes()
    pdf.close()
    return content


def test_figure_agent_extracts_real_image_labels_and_report():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"):
            file = wf.add_file("architecture.pdf", _pdf_with_figure())
            plan = wf.plan("Explain the architecture diagram", [file["id"]])
            assert [task["tool"] for task in plan] == ["read", "visual", "report"]
            state = wf.run_workflow("Explain the architecture diagram", [file["id"]])
            assert state["status"] == "completed", state
            figures = next(task["output"] for task in state["tasks"] if task["tool"] == "visual")
            labels = " ".join(label["text"] for label in figures["figures"][0]["labels"]).lower()
            assert "input" in labels and "output" in labels
            assert "Figure 1" in figures["figures"][0]["caption"]
            assert (Path(directory) / figures["figures"][0]["image"].split("/")[-1]).is_file()
            assert TestClient(app).get(figures["figures"][0]["image"]).content.startswith(b"\x89PNG")
            report = state["tasks"][-1]["output"]["markdown"]
            assert "Paper figures and diagrams" in report and "OCR labels" in report


def test_scanned_page_is_read_by_local_ocr():
    image = Image.new("RGB", (1200, 600), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
    draw.text((70, 100), "Scanned research evidence", fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG")
    pdf = pymupdf.open()
    pdf.new_page().insert_image(pymupdf.Rect(0, 0, 612, 300), stream=output.getvalue())
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        path = Path(directory) / "scan.pdf"
        path.write_bytes(pdf.tobytes())
        evidence = wf.read_evidence({"path": path, "suffix": ".pdf", "name": "scan.pdf"})
        assert "scanned research evidence" in evidence["text"].lower()
        assert evidence["ocr_pages"] == [1]
    pdf.close()


def test_full_page_scanned_diagram_routes_to_visual_agent():
    image = Image.new("RGB", (1000, 900), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 45)
    for index, label in enumerate(("Inputs", "Embedding", "Positional Encoding",
                                   "Multi-Head Attention", "Add & Norm", "Feed Forward",
                                   "Linear", "Softmax", "Outputs")):
        y = 25 + index * 95
        draw.rectangle((120, y, 880, y + 65), outline="black", width=3)
        draw.text((160, y + 6), label, fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG")
    pdf = pymupdf.open()
    pdf.new_page(width=500, height=450).insert_image(
        pymupdf.Rect(0, 0, 500, 450), stream=output.getvalue())
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(wf, "ROOT", Path(directory)), patch.object(wf, "DB", Path(directory) / "state.db"):
            file = wf.add_file("scan-diagram.pdf", pdf.tobytes())
            tasks = wf.plan("Explain the scanned workflow diagram", [file["id"]])
            assert [task["tool"] for task in tasks] == ["read", "visual", "report"]
            state = wf.run_workflow("Explain the scanned workflow diagram", [file["id"]])
            assert state["status"] == "completed", state
            figure = next(task["output"] for task in state["tasks"] if task["tool"] == "visual")["figures"][0]
            assert figure["page"] == 1
            assert "scanned diagram" in figure["caption"]
            assert "multi-head attention" in " ".join(label["text"].lower() for label in figure["labels"])
    pdf.close()
