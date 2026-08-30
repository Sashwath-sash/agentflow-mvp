from pathlib import Path

import pandas as pd

from agentflow.data_tools import inspect_table


def test_inspects_csv_structure():
    source = Path(__file__).parent / ".test-data.csv"
    pd.DataFrame({"topic": ["agents", "tools"], "score": [4, 5]}).to_csv(source, index=False)
    try:
        summary = inspect_table(source)
        assert summary["rows"] == 2
        assert summary["columns"] == 2
        assert summary["column_names"] == ["topic", "score"]
        assert summary["missing_values"]["score"] == 0
        assert "mean" in summary["numeric_summary"]["score"]
    finally:
        source.unlink(missing_ok=True)
