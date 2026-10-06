from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DataReadError(ValueError):
    """Raised when a supported tabular file cannot be analysed."""


def inspect_table(path: str | Path) -> dict[str, Any]:
    """Read CSV/XLSX data and return a JSON-serialisable structural summary."""
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise DataReadError(f"data file not found: {source}")
    try:
        if source.suffix.lower() == ".csv":
            frame = pd.read_csv(source)
        elif source.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(source)
        else:
            raise DataReadError(f"unsupported data type: {source.suffix or 'none'}")
    except DataReadError:
        raise
    except Exception as exc:
        raise DataReadError(f"could not read data file: {source}") from exc

    numeric = frame.select_dtypes(include="number")
    numeric_summary = numeric.describe().round(4).to_dict() if not numeric.empty else {}
    return {
        "file": source.name,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": [str(name) for name in frame.columns],
        "column_types": {str(name): str(dtype) for name, dtype in frame.dtypes.items()},
        "missing_values": {str(name): int(count) for name, count in frame.isna().sum().items()},
        "numeric_summary": numeric_summary,
    }
