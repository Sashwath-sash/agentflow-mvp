from pathlib import Path

import pytest

from agentflow.document_tools import DocumentReadError, read_document


def test_reads_utf8_text_file():
    source = Path(__file__).parent / ".test-paper.txt"
    source.write_text("AgentFlow coordinates dependent tasks.", encoding="utf-8")
    try:
        assert read_document(source) == "AgentFlow coordinates dependent tasks."
    finally:
        source.unlink(missing_ok=True)


def test_rejects_unsupported_type():
    source = Path(__file__).parent / ".test-image.png"
    source.write_bytes(b"not an image")
    try:
        with pytest.raises(DocumentReadError, match="unsupported document type"):
            read_document(source)
    finally:
        source.unlink(missing_ok=True)


def test_rejects_missing_file():
    with pytest.raises(DocumentReadError, match="document not found"):
        read_document(Path(__file__).parent / ".missing.txt")
