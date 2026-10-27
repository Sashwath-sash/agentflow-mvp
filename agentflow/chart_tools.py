"""Select a bounded set of evidence-led charts for uploaded CSV/Excel tables."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


MAX_CHARTS = 4
MAX_CATEGORIES = 20
_ID = re.compile(r"(^|[_\s-])(id|identifier|zip|postcode|code|index)([_\s-]|$)", re.I)
_DATE = re.compile(r"date|time|month|year|period|day", re.I)
_DISCOVER = re.compile(r"discover|explor|overview|trend|analy[sz]|inspect|all columns|patterns", re.I)
_RELATION = re.compile(r"relation|correlat|versus|\bvs\b|scatter|association", re.I)
_MISSING = re.compile(r"\b(missing|nulls?|completeness)\b", re.I)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def _mentioned(column: str, objective: str) -> bool:
    words = re.findall(r"[a-z0-9]+", column.lower())
    haystack = re.findall(r"[a-z0-9]+", objective.lower())
    return bool(words) and all(word in haystack for word in words)


def _profile_columns(frame: pd.DataFrame):
    numeric, dates, excluded = {}, {}, {}
    for column in frame.columns:
        name, series = str(column), frame[column]
        if _DATE.search(name):
            try:
                if name.lower() == "year" and pd.api.types.is_numeric_dtype(series):
                    parsed = pd.to_datetime(series.astype("Int64").astype("string"), format="%Y", errors="coerce")
                else:
                    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                if parsed.notna().sum() >= 3 and parsed.notna().mean() >= .8 and parsed.nunique() >= 3:
                    dates[name] = parsed
                    continue
            except (TypeError, ValueError):
                pass
        if _ID.search(name):
            excluded[name] = "identifier, not a measured value"
            continue
        if pd.api.types.is_bool_dtype(series):
            excluded[name] = "boolean category"
            continue
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            raw = series.astype("string").str.strip()
            nonempty = raw.notna() & raw.ne("")
            if not nonempty.any():
                excluded[name] = "empty column"
                continue
            values = pd.to_numeric(raw.str.replace(",", "", regex=False)
                                   .str.replace("%", "", regex=False), errors="coerce")
            if values.notna().sum() / nonempty.sum() < .95:
                continue
        else:
            continue
        if values.notna().sum() < 2:
            excluded[name] = "fewer than two numeric observations"
        elif values.nunique(dropna=True) < 2:
            excluded[name] = "constant numeric value"
        else:
            numeric[name] = values
    return numeric, dates, excluded


def _figure(width=11, height=5.5):
    fig = Figure(figsize=(width, height), layout="constrained")
    return fig, fig.subplots()


def _aggregation(metric: str, objective: str) -> str:
    if re.search(r"\b(average|mean)\b", objective, re.I):
        return "mean"
    if re.search(r"\b(total|sum)\b", objective, re.I):
        return "sum"
    if re.search(r"rate|ratio|percent|pct|score|accuracy|top\d|average|mean|price", metric, re.I):
        return "mean"
    if re.search(r"revenue|sales|amount|count|units|quantity|volume|spend|cost|profit", metric, re.I):
        return "sum"
    return "mean"


def _missing_chart(frame: pd.DataFrame, title: str):
    counts = frame.isna().sum().sort_values(ascending=False).head(MAX_CATEGORIES)
    fig, ax = _figure(11, max(4, min(8, len(counts) * .32 + 2)))
    if counts.sum():
        ax.barh([str(x) for x in counts.index], counts.values, color="#2563eb")
        ax.invert_yaxis()
        ax.set_xlabel("Missing rows")
        message = f"Missing-value counts for {title}; {len(counts)} columns shown."
    else:
        ax.axis("off")
        ax.text(.5, .5, "No missing values", ha="center", va="center",
                fontsize=20, transform=ax.transAxes)
        message = f"No missing values in {len(frame):,} rows and {len(frame.columns)} columns."
    ax.set_title(f"{title} - missing values", fontsize=11)
    return fig, message, {"type": "missingness", "columns": list(map(str, counts.index)),
                          "selection": "largest missing counts"}


def _category_chart(category: str, metric: str, labels: pd.Series, values: pd.Series,
                    title: str, objective: str):
    data = pd.DataFrame({"category": labels.astype("string"), "value": values}).dropna()
    data = data[data["category"].str.strip().ne("")]
    if len(data) < 2:
        return None
    unique = data["category"].nunique()
    repeated = unique < len(data)
    if repeated:
        method = _aggregation(metric, objective)
        plotted = data.groupby("category")["value"].agg(method).sort_values(ascending=False)
        aggregation = f"{method} of repeated categories"
    else:
        plotted = data.set_index("category")["value"]
        aggregation = "individual complete rows"
    if unique > MAX_CATEGORIES:
        plotted = plotted.sort_values(ascending=False).head(MAX_CATEGORIES)
        selection = f"highest {len(plotted)} of {unique} categories"
    else:
        selection = f"all {unique} categories"
    fig, ax = _figure(12, max(4.5, min(9, len(plotted) * .34 + 2)))
    numbers = plotted.to_numpy(dtype=float)
    names = [str(label)[:55] for label in plotted.index]
    span = float(max(numbers) - min(numbers))
    close_values = min(numbers) > 0 and span < .15 * float(max(numbers))
    if close_values:
        positions = list(range(len(numbers)))
        ax.scatter(numbers, positions, color="#2563eb", s=55)
        ax.set_yticks(positions, names)
        padding = max(span * .2, float(max(numbers)) * .005)
        ax.set_xlim(float(min(numbers)) - padding, float(max(numbers)) + padding * 2)
        ax.grid(axis="x", alpha=.2)
        if len(numbers) <= 12:
            for number, position in zip(numbers, positions):
                ax.annotate(f"{number:.3f}", (number, position), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=8)
        kind = "dot"
    else:
        ax.barh(names, numbers, color="#2563eb")
        if min(numbers) >= 0:
            ax.set_xlim(left=0)
        kind = "bar"
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"{title} - {metric} by {category}", fontsize=11)
    message = f"{metric} by {category} from {title}: {selection}; {aggregation}; {len(data)} complete rows."
    if kind == "dot":
        message += " Dot plot uses a zoomed value axis."
    return fig, message, {"type": kind, "columns": [category, metric],
                          "selection": selection, "aggregation": aggregation, "rows_used": len(data)}


def _time_chart(date_name: str, metric: str, dates: pd.Series, values: pd.Series,
                title: str, objective: str):
    data = pd.DataFrame({"date": dates, "value": values}).dropna()
    if data["date"].nunique() < 3:
        return None
    repeated = data["date"].duplicated().any()
    method = _aggregation(metric, objective)
    plotted = data.groupby("date")["value"].agg(method).sort_index()
    original_points = len(plotted)
    if len(plotted) > 500:
        plotted = plotted.iloc[::math.ceil(len(plotted) / 500)]
    fig, ax = _figure(11, 5)
    ax.plot(plotted.index, plotted.values, color="#2563eb", linewidth=2,
            marker="o" if len(plotted) <= 20 else None)
    ax.set_ylabel(metric)
    ax.set_xlabel(date_name)
    ax.set_title(f"{title} - {metric} over {date_name}", fontsize=11)
    ax.grid(axis="y", alpha=.2)
    message = f"Time trend: {metric} over {date_name}; {len(data)} complete rows."
    if repeated:
        message += f" Values sharing a date used {method}."
    if original_points > len(plotted):
        message += f" Displayed {len(plotted)} of {original_points} dates at a regular interval."
    return fig, message, {"type": "time_series", "columns": [date_name, metric],
                          "selection": "chronological dates", "rows_used": len(data),
                          "aggregation": method if repeated else "none"}


def _histogram(metric: str, values: pd.Series, title: str):
    data = values.dropna()
    if len(data) < 2:
        return None
    fig, ax = _figure(10, 5)
    ax.hist(data, bins=min(20, max(5, int(math.sqrt(len(data))))), color="#2563eb")
    ax.set_xlabel(metric)
    ax.set_ylabel("Row count")
    ax.set_title(f"{title} - distribution of {metric}", fontsize=11)
    return fig, f"Distribution of {metric} across {len(data):,} non-missing rows.", {
        "type": "histogram", "columns": [metric], "selection": "all non-missing rows", "rows_used": len(data)}


def _scatter(x_name: str, y_name: str, x: pd.Series, y: pd.Series, title: str):
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 5 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return None
    total = len(data)
    if total > 2000:
        data = data.sample(n=2000, random_state=0)
    fig, ax = _figure(9, 6)
    ax.scatter(data["x"], data["y"], color="#2563eb", s=20, alpha=.55)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{title} - {y_name} versus {x_name}", fontsize=11)
    correlation = data["x"].corr(data["y"])
    selection = "all complete pairs" if total <= 2000 else "fixed random sample of 2,000 complete pairs"
    message = (f"Relationship between {x_name} and {y_name}: {selection}; "
               f"Pearson r={correlation:.2f}. Association does not establish causation.")
    return fig, message, {"type": "scatter", "columns": [x_name, y_name],
                          "selection": selection, "rows_used": len(data)}


def _frequency(column: str, values: pd.Series, title: str):
    counts = values.dropna().astype("string").value_counts().head(MAX_CATEGORIES)
    if counts.empty:
        return None
    fig, ax = _figure(10, max(4, min(8, len(counts) * .32 + 2)))
    ax.barh([str(x)[:55] for x in counts.index], counts.values, color="#2563eb")
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Row count")
    ax.set_title(f"{title} - category frequency", fontsize=11)
    return fig, f"Most frequent {len(counts)} values of {column}; counts are observed rows.", {
        "type": "frequency", "columns": [column], "selection": "most frequent categories"}


def make_charts(path: Path, objective: str, summary: dict):
    """Return selected figures and a transparent profile of plotted/omitted columns."""
    frame = _read(path)
    title = summary.get("file", path.name)
    numeric, dates, excluded = _profile_columns(frame)
    ordered = sorted(numeric, key=lambda name: (
        -(100 if _mentioned(name, objective) else 0)
        -(15 if re.search(r"top\d+|accuracy|score|rate|revenue|sales|value|amount", name, re.I) else 0)
        +(5 if re.search(r"error|_err$", name, re.I) and not _mentioned(name, objective) else 0),
        list(map(str, frame.columns)).index(name)))
    requested = [name for name in ordered if _mentioned(name, objective)]
    discovery = bool(_DISCOVER.search(objective)) or not requested
    metrics = requested[:3] if requested else ordered[:2]
    if requested and discovery and len(metrics) < 2:
        metrics += [name for name in ordered if name not in metrics][:2 - len(metrics)]
    categories = [str(name) for name in frame.columns if str(name) not in numeric and
                  str(name) not in dates and str(name) not in excluded and
                  frame[name].dropna().nunique() >= 2]
    category = next((name for name in categories if _mentioned(name, objective)),
                    categories[0] if categories and frame[categories[0]].nunique() <= MAX_CATEGORIES else None)
    date = next((name for name in dates if _mentioned(name, objective)), next(iter(dates), None))
    missing_requested = bool(_MISSING.search(objective))
    only_missing = missing_requested and not requested and not _DISCOVER.search(objective)
    charts = []
    if missing_requested:
        charts.append(_missing_chart(frame, title))
    if not only_missing:
        for metric in metrics:
            if len(charts) >= MAX_CHARTS:
                break
            result = _time_chart(date, metric, dates[date], numeric[metric], title, objective) if date else None
            if result is None and category:
                result = _category_chart(category, metric, frame[category], numeric[metric], title, objective)
            if result is None:
                result = _histogram(metric, numeric[metric], title)
            if result:
                charts.append(result)
        if len(ordered) >= 2 and len(charts) < MAX_CHARTS and (_RELATION.search(objective) or discovery):
            pair = metrics[:2] if len(metrics) >= 2 else ordered[:2]
            if len(pair) == 2:
                result = _scatter(pair[0], pair[1], numeric[pair[0]], numeric[pair[1]], title)
                if result:
                    charts.append(result)
        if not charts and categories:
            result = _frequency(categories[0], frame[categories[0]], title)
            if result:
                charts.append(result)
        if frame.isna().any().any() and len(charts) < MAX_CHARTS and not missing_requested:
            charts.append(_missing_chart(frame, title))
    if not charts:
        fig, ax = _figure(9, 4)
        ax.axis("off")
        ax.text(.5, .5, "No variable values available for a meaningful chart", ha="center",
                va="center", transform=ax.transAxes)
        charts.append((fig, f"{title} has no variable numeric or categorical values to plot.",
                       {"type": "no_data", "columns": [], "selection": "none"}))
    plotted = {name for _, _, detail in charts if detail["type"] != "missingness"
               for name in detail["columns"]}
    omitted = []
    for column in frame.columns:
        name = str(column)
        if name not in plotted:
            if name in excluded:
                reason = excluded[name]
            elif only_missing:
                reason = "only missingness requested"
            elif name in numeric:
                reason = "numeric metric not selected; name this column in the objective to plot it"
            elif name in dates:
                reason = "date column not paired with a selected metric"
            elif name in categories and frame[name].nunique() > MAX_CATEGORIES:
                reason = "many distinct labels; name this column to request a top-20 category chart"
            else:
                reason = "text category not paired with a selected metric"
            omitted.append({"column": name, "reason": reason})
    profile = {"file": title, "rows": len(frame), "columns": len(frame.columns),
               "chart_count": len(charts), "plotted_columns": sorted(plotted),
               "omitted_columns": omitted[:30], "additional_omitted_count": max(0, len(omitted) - 30),
               "limit": MAX_CHARTS}
    return charts, profile


def make_chart(path: Path, objective: str, summary: dict) -> tuple[Figure, str]:
    """Compatibility helper for callers that expect one figure."""
    charts, _ = make_charts(path, objective, summary)
    return charts[0][0], charts[0][1]
