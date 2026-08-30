from __future__ import annotations

from pathlib import Path


class DocumentReadError(ValueError):
    """Raised when a supported document cannot be read."""


def read_document(path: str | Path) -> str:
    """Extract text from a UTF-8 text file or a PDF using PyMuPDF."""
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise DocumentReadError(f"document not found: {source}")
    suffix = source.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return source.read_text(encoding="utf-8")
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise DocumentReadError("PDF support requires PyMuPDF (package: pymupdf)") from exc
        with fitz.open(source) as document:
            return "\n".join(page.get_text() for page in document)
    raise DocumentReadError(f"unsupported document type: {suffix or 'none'}")
