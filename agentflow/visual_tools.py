"""Bounded, local OCR and figure inspection for academic PDFs.

This extracts visible labels and page/caption context. It does not claim to
recover arbitrary diagram edges or to understand photographs and plots.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from rapidocr import RapidOCR


_engine: RapidOCR | None = None
_DIAGRAM_TERMS = re.compile(
    r"architecture|framework|pipeline|flowchart|workflow|system|model|overview|scaled dot.product|multi.head|"
    r"module|network|process|procedure|pre.training|fine.tuning|diagram|feed.forward|positional.encoding", re.I)


def _ocr_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def _recognize(png: bytes) -> list[dict]:
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode PDF page image for OCR")
    result = _ocr_engine()(image)
    boxes = result.boxes if result.boxes is not None else []
    texts = result.txts if result.txts is not None else []
    scores = result.scores if result.scores is not None else []
    lines = []
    for box, value, score in zip(boxes, texts, scores):
        text = str(value).strip()
        if text and float(score) >= 0.35:
            points = np.asarray(box, dtype=float)
            lines.append({"text": text, "confidence": round(float(score), 3),
                          "x": round(float(points[:, 0].mean() / image.shape[1]), 3),
                          "y": round(float(points[:, 1].mean() / image.shape[0]), 3)})
    return sorted(lines, key=lambda line: (line["y"], line["x"]))


def ocr_page(page: pymupdf.Page) -> str:
    """Read a scanned page when its normal PDF text layer is empty."""
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7), alpha=False)
    return "\n".join(line["text"] for line in _recognize(pixmap.tobytes("png")))


def _scanned_diagram(page: pymupdf.Page) -> tuple[bytes, list[dict]] | None:
    """Recognize diagram-like full-page scans, which have no PDF caption/text layer."""
    if page.get_text().strip():
        return None
    if not any(rect.get_area() / page.rect.get_area() >= .8
               for image in page.get_images(full=True)
               for rect in page.get_image_rects(image[0])):
        return None
    png = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).tobytes("png")
    labels = _recognize(png)
    matches = [match.group().lower() for line in labels
               for match in _DIAGRAM_TERMS.finditer(line["text"])]
    return (png, labels) if len(labels) >= 8 and (len(set(matches)) >= 2 or len(matches) >= 3) else None


def has_substantial_figure(path: Path) -> bool:
    with pymupdf.open(path) as pdf:
        for page in pdf:
            if _scanned_diagram(page):
                return True
            for image in page.get_images(full=True):
                for rect in page.get_image_rects(image[0]):
                    fraction = rect.get_area() / page.rect.get_area()
                    if (rect.width >= 100 and rect.height >= 80 and .045 <= fraction <= .65
                            and _DIAGRAM_TERMS.search(_caption(page, rect))):
                        return True
            if len(page.get_drawings()) >= 8 and _vector_captions(page):
                return True
    return False


def _caption(page: pymupdf.Page, rect: pymupdf.Rect) -> str:
    candidates = []
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue
        text = re.sub(r"\s+", " ", block[4]).strip()
        match = re.search(r"^(?:Figure|Fig\.)\s*\d+\s*[:.]\s*[^.]{0,180}\.?", text, re.I)
        if not match:
            continue
        box = pymupdf.Rect(block[:4])
        distance = min(abs(box.y0 - rect.y1), abs(rect.y0 - box.y1))
        candidates.append((distance, match.group(0)[:260]))
    return min(candidates)[1] if candidates and min(candidates)[0] < 110 else "Caption not detected"


def _layout(lines: list[dict]) -> list[dict]:
    """Expose relative visual placement, not unverified arrow connectivity."""
    regions = []
    for name, predicate in (("left", lambda x: x < .42),
                            ("centre", lambda x: .42 <= x <= .58),
                            ("right", lambda x: x > .58)):
        labels = [line["text"] for line in lines if predicate(line["x"])]
        if labels:
            regions.append({"region": name, "labels_top_to_bottom": labels[:24]})
    return regions


def _vector_captions(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    candidates = []
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue
        text = re.sub(r"\s+", " ", block[4]).strip()
        match = re.search(r"^(?:Figure|Fig\.)\s*\d+\s*[:.]\s*[^.]{0,180}\.?", text, re.I)
        if match and _DIAGRAM_TERMS.search(match.group(0)):
            candidates.append((match.group(0), pymupdf.Rect(block[:4])))
    return candidates


def inspect_figures(path: Path, artifact_dir: Path, artifact_prefix: str,
                    max_figures: int = 6) -> list[dict]:
    """Render substantial embedded figures, OCR their labels, and retain provenance."""
    figures = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, 1):
            scan = _scanned_diagram(page)
            if scan:
                png, labels = scan
                filename = f"{artifact_prefix}-{len(figures)}.png"
                (artifact_dir / filename).write_bytes(png)
                figures.append({"file": path.name, "page": page_number,
                                "caption": "Full-page scanned diagram; no embedded caption",
                                "page_context": "OCR-only page; inspect the image for arrows and connections",
                                "image": "/artifacts/" + filename,
                                "labels": labels[:50], "layout": _layout(labels),
                                "method": "OCR labels and placement; arrows and semantic relationships are not verified"})
                if len(figures) >= max_figures:
                    return figures
                continue
            seen = set()
            used_captions = set()
            for image in page.get_images(full=True):
                for rect in page.get_image_rects(image[0]):
                    fraction = rect.get_area() / page.rect.get_area()
                    caption = _caption(page, rect)
                    if (rect.width < 100 or rect.height < 80 or not .045 <= fraction <= .65
                            or not _DIAGRAM_TERMS.search(caption)):
                        continue
                    key = tuple(round(value) for value in rect)
                    if key in seen:
                        continue
                    seen.add(key)
                    original = cv2.imdecode(np.frombuffer(pdf.extract_image(image[0])["image"], dtype=np.uint8),
                                            cv2.IMREAD_COLOR)
                    if original is None:
                        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=rect, alpha=False)
                        png = pixmap.tobytes("png")
                    else:
                        encoded, data = cv2.imencode(".png", original)
                        if not encoded:
                            raise ValueError("Could not encode extracted PDF figure")
                        png = data.tobytes()
                    labels = _recognize(png)
                    filename = f"{artifact_prefix}-{len(figures)}.png"
                    (artifact_dir / filename).write_bytes(png)
                    figures.append({"file": path.name, "page": page_number,
                                    "caption": caption,
                                    "page_context": re.sub(r"\s+", " ", page.get_text()).strip()[:900],
                                    "image": "/artifacts/" + filename,
                                    "labels": labels[:50], "layout": _layout(labels),
                                    "method": "OCR labels and their visual placement; arrows and semantic relationships are not verified"})
                    used_captions.add(caption)
                    if len(figures) >= max_figures:
                        return figures
            if len(page.get_drawings()) < 8:
                continue
            for caption, caption_rect in _vector_captions(page):
                if caption in used_captions or caption_rect.y0 < 125:
                    continue
                clip = pymupdf.Rect(40, max(30, caption_rect.y0 - 350),
                                    page.rect.width - 40, caption_rect.y0 - 6)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False)
                png = pixmap.tobytes("png")
                labels = _recognize(png)
                if not labels:
                    continue
                filename = f"{artifact_prefix}-{len(figures)}.png"
                (artifact_dir / filename).write_bytes(png)
                figures.append({"file": path.name, "page": page_number,
                                "caption": caption,
                                "embedded_text_labels": [re.sub(r"\s+", " ", block[4]).strip()
                                                         for block in page.get_text("blocks", clip=clip)
                                                         if len(block) >= 7 and block[6] == 0][:30],
                                "page_context": re.sub(r"\s+", " ", page.get_text()).strip()[:900],
                                "image": "/artifacts/" + filename,
                                "labels": labels[:50], "layout": _layout(labels),
                                "method": "OCR labels and their visual placement; arrows and semantic relationships are not verified"})
                if len(figures) >= max_figures:
                    return figures
    return figures
