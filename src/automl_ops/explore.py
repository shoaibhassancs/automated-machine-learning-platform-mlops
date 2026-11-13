"""Exploration data for the preview page (BUILD_PLAN Step 6).

`explore_frame` returns plain, JSON-ready structures: the first rows, a histogram per numeric
column, a Pearson correlation summary for numeric columns, and (for a classification target) the
class balance. No plotting library is used; rendering is the front end's job. Nothing is stored and
no value is converted in the data: numbers are parsed only to compute these summaries.

Defaults (10 sample rows, 20 histogram bins) are provisional (docs/exploration.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from automl_ops.profile import DatasetProfile, significant
from automl_ops.task import TaskConfig, class_counts

EXPLORE_FORMAT = 1
CLASS_LIST_LIMIT = 50  # classes listed individually; the rest are summarised
TOP_PAIR_LIMIT = 10


@dataclass(frozen=True)
class ExploreSettings:
    sample_rows: int = 10
    histogram_bins: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {"sample_rows": self.sample_rows, "histogram_bins": self.histogram_bins}


DEFAULT_EXPLORE_SETTINGS = ExploreSettings()


def _number(value: float) -> float | None:
    """A JSON-safe number: rounded to 12 significant digits, None if not finite."""
    return significant(value) if math.isfinite(value) else None


def _floats(series: pd.Series) -> np.ndarray:
    """A numeric-kind column as floats, NaN where the cell is missing.

    The column kind (integer or float) guarantees every non-missing cell is a clean number.
    """
    return np.array([np.nan if pd.isna(v) else float(str(v)) for v in series], dtype="float64")


def sample_rows(frame: pd.DataFrame, count: int) -> list[dict[str, str | None]]:
    """The first `count` rows, cells as the original text; a missing cell is None."""
    head = frame.head(count)
    rows: list[dict[str, str | None]] = []
    for record in head.itertuples(index=False, name=None):
        rows.append(
            {
                str(name): (None if pd.isna(cell) else str(cell))
                for name, cell in zip(frame.columns, record, strict=True)
            }
        )
    return rows


def histogram(values: np.ndarray, bins: int) -> dict[str, Any]:
    """Equal-width histogram (NumPy's `histogram`): `bins + 1` edges and `bins` counts."""
    if values.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "edges": [significant(float(e)) for e in edges],
        "counts": [int(c) for c in counts],
    }


def correlation_summary(numeric: pd.DataFrame) -> dict[str, Any]:
    """Pearson correlation between numeric columns, using the rows where both values exist.

    A pair involving a constant column (no spread) has no correlation and is None.
    """
    names = [str(c) for c in numeric.columns]
    if len(names) < 2:
        return {"columns": names, "matrix": [[None] * len(names) for _ in names], "top_pairs": []}
    corr = numeric.corr(method="pearson").to_numpy(dtype="float64")
    matrix: list[list[float | None]] = [
        [_number(float(corr[i, j])) for j in range(len(names))] for i in range(len(names))
    ]
    pairs: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = matrix[i][j]
            if r is not None:
                pairs.append({"a": names[i], "b": names[j], "r": r})
    pairs.sort(key=lambda p: (-abs(p["r"]), p["a"], p["b"]))
    return {"columns": names, "matrix": matrix, "top_pairs": pairs[:TOP_PAIR_LIMIT]}


def class_balance(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    counts = class_counts(frame, target)
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    listed = [
        {"value": value, "count": count, "share": significant(count / total)}
        for value, count in ordered[:CLASS_LIST_LIMIT]
    ]
    smallest = ordered[-1][1] if ordered else 0
    largest = ordered[0][1] if ordered else 0
    return {
        "total_classes": len(ordered),
        "rows": total,
        "classes": listed,
        "classes_not_listed": max(0, len(ordered) - CLASS_LIST_LIMIT),
        "largest_class_share": significant(largest / total) if total else None,
        "smallest_class_rows": smallest,
        "largest_to_smallest_ratio": significant(largest / smallest) if smallest else None,
    }


def explore_frame(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    config: TaskConfig | None = None,
    settings: ExploreSettings = DEFAULT_EXPLORE_SETTINGS,
) -> dict[str, Any]:
    """Everything the preview page needs, as plain data. Pure: `frame` is not changed.

    Histograms and correlations cover columns whose profile kind is `integer` or `float`.
    `class_balance` is filled only for a classification config whose target exists.
    """
    numeric_names = [c.name for c in profile.columns if c.kind in ("integer", "float")]
    numeric = pd.DataFrame({name: _floats(frame[name]) for name in numeric_names})
    histograms = []
    for name in numeric_names:
        column = profile.column(name)
        assert column is not None and column.numeric is not None
        histograms.append(
            {
                "column": name,
                "count": column.count,
                "missing": column.missing,
                "min": column.numeric.minimum,
                "max": column.numeric.maximum,
                **histogram(numeric[name].dropna().to_numpy(), settings.histogram_bins),
            }
        )
    balance = None
    if config is not None and config.task == "classification" and config.target in frame.columns:
        balance = class_balance(frame, config.target)
    return {
        "explore_format": EXPLORE_FORMAT,
        "settings": settings.to_dict(),
        "rows": len(frame),
        "columns": [c.name for c in profile.columns],
        "sample_rows": sample_rows(frame, settings.sample_rows),
        "histograms": histograms,
        "correlation": correlation_summary(numeric),
        "class_balance": balance,
    }
