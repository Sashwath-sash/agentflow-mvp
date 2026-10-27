"""Chart choice must follow actual column types and disclose sampling/aggregation."""

import pytest
import tempfile
from pathlib import Path

from agentflow.chart_tools import make_charts
from agentflow.data_tools import inspect_table


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        yield Path(directory)


def charts_for(tmp_path, name, content, objective):
    source = tmp_path / name
    source.write_text(content, encoding="utf-8")
    return make_charts(source, objective, inspect_table(source))


def test_time_trend_uses_sum_for_revenue_and_mean_for_rate(tmp_path):
    charts, profile = charts_for(
        tmp_path, "sales.csv",
        "date,revenue,conversion_rate\n2026-01-01,10,0.1\n2026-01-01,20,0.2\n"
        "2026-02-01,30,0.3\n2026-03-01,40,0.4\n",
        "Discover trends in the data",
    )
    trends = [chart for chart in charts if chart[2]["type"] == "time_series"]
    assert len(trends) == 2
    revenue = next(chart for chart in trends if "revenue" in chart[2]["columns"])
    rate = next(chart for chart in trends if "conversion_rate" in chart[2]["columns"])
    assert list(revenue[0].axes[0].lines[0].get_ydata()) == [30, 30, 40]
    assert list(rate[0].axes[0].lines[0].get_ydata()) == pytest.approx([0.15, 0.3, 0.4])
    assert "sum" in revenue[1] and "mean" in rate[1]
    assert profile["chart_count"] <= 4


def test_numeric_text_relationship_and_identifier_exclusion(tmp_path):
    charts, profile = charts_for(
        tmp_path, "models.csv",
        "model_id,model,top1,param_count\n1,A,80,\"1,000\"\n2,B,81,\"2,000\"\n"
        "3,C,82,\"3,000\"\n4,D,83,\"4,000\"\n5,E,84,\"5,000\"\n",
        "Analyze relationship between top1 and param_count",
    )
    scatter = next(chart for chart in charts if chart[2]["type"] == "scatter")
    assert set(scatter[2]["columns"]) == {"top1", "param_count"}
    assert "causation" in scatter[1]
    assert any(item["column"] == "model_id" and "identifier" in item["reason"]
               for item in profile["omitted_columns"])


def test_categorical_only_gets_frequency_chart(tmp_path):
    charts, _ = charts_for(tmp_path, "labels.csv", "group\nA\nB\nA\nC\nA\n", "Explore this dataset")
    assert charts[0][2]["type"] == "frequency"
    assert "observed rows" in charts[0][1]


def test_high_cardinality_explicit_category_discloses_top_twenty(tmp_path):
    rows = "\n".join(f"M{i},{i}" for i in range(25))
    charts, _ = charts_for(tmp_path, "many.csv", "model,score\n" + rows + "\n",
                           "Plot score by model")
    assert charts[0][2]["selection"] == "highest 20 of 25 categories"
    assert charts[0][2]["rows_used"] == 25


def test_explicit_missing_values_does_not_hide_zero_missing(tmp_path):
    charts, _ = charts_for(tmp_path, "complete.csv", "model,score\nA,80\nB,90\n",
                           "Chart missing values")
    assert len(charts) == 1
    assert charts[0][2]["type"] == "missingness"
    assert "No missing values" in charts[0][1]
