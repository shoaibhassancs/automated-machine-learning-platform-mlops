"""Exploration tests. Expected values come from hand-computed cases and from NumPy called directly
on the invented numbers (an independent reference), not from the module under test."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import numpy as np
import pytest

from automl_ops.explore import ExploreSettings, explore_frame
from automl_ops.profile import profile_frame, read_text_frame
from automl_ops.task import TaskConfig


def explore(
    columns: dict[str, list[str]],
    config: TaskConfig | None = None,
    settings: ExploreSettings | None = None,
) -> dict[str, Any]:
    names = list(columns)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    for i in range(len(columns[names[0]])):
        writer.writerow([columns[name][i] for name in names])
    frame = read_text_frame(out.getvalue().encode("utf-8"))
    profile = profile_frame(frame)
    if settings is None:
        return explore_frame(frame, profile, config)
    return explore_frame(frame, profile, config, settings)


def hist(result: dict[str, Any], column: str) -> dict[str, Any]:
    found: list[dict[str, Any]] = [h for h in result["histograms"] if h["column"] == column]
    if found:
        return found[0]
    raise AssertionError(f"no histogram for {column}")


# --- sample rows ---------------------------------------------------------------------------


def test_sample_rows_are_the_first_rows_as_original_text() -> None:
    columns = {"a": ["007", "", "3"], "b": ["x", "NA", "z"]}
    result = explore(columns)
    assert result["sample_rows"] == [
        {"a": "007", "b": "x"},
        {"a": None, "b": "NA"},  # empty cell: None; NA text kept
        {"a": "3", "b": "z"},
    ]


def test_sample_rows_default_is_ten_and_follows_settings() -> None:
    columns = {"a": [str(i) for i in range(25)]}
    assert [r["a"] for r in explore(columns)["sample_rows"]] == [str(i) for i in range(10)]
    assert len(explore(columns, settings=ExploreSettings(sample_rows=3))["sample_rows"]) == 3


# --- histograms ----------------------------------------------------------------------------


def test_histogram_hand_computed() -> None:
    result = explore({"v": [str(i) for i in range(10)]}, settings=ExploreSettings(histogram_bins=2))
    h = hist(result, "v")
    assert h["edges"] == [0.0, 4.5, 9.0] and h["counts"] == [5, 5]
    assert (h["count"], h["missing"], h["min"], h["max"]) == (10, 0, 0.0, 9.0)


def test_histogram_matches_numpy_on_seeded_data() -> None:
    rng = np.random.default_rng(7)
    values = np.round(rng.normal(50, 12, size=500), 3)
    result = explore({"v": [repr(float(x)) for x in values]})
    expected_counts, expected_edges = np.histogram(values, bins=20)
    h = hist(result, "v")
    assert h["counts"] == expected_counts.tolist() and sum(h["counts"]) == 500
    assert h["edges"] == pytest.approx(expected_edges.tolist(), rel=1e-11)
    assert len(h["counts"]) == 20 and len(h["edges"]) == 21  # 20 bins by default


def test_histogram_ignores_missing_and_covers_only_numeric_columns() -> None:
    columns = {
        "num": ["1", "2", "", "4"],
        "cat": ["a", "b", "a", "b"],
        "code": ["007", "012", "5", "6"],
        "mixed": ["1", "2", "3", "x"],
    }
    result = explore(columns)
    assert [h["column"] for h in result["histograms"]] == ["num"]
    h = hist(result, "num")
    assert h["missing"] == 1 and sum(h["counts"]) == 3


def test_constant_numeric_column_still_gets_a_histogram() -> None:
    h = hist(explore({"v": ["5"] * 8}), "v")
    assert sum(h["counts"]) == 8 and len(h["counts"]) == 20


# --- correlation ---------------------------------------------------------------------------


def test_correlation_perfect_and_reversed() -> None:
    result = explore(
        {"a": ["1", "2", "3", "4"], "b": ["2", "4", "6", "8"], "c": ["4", "3", "2", "1"]}
    )
    corr = result["correlation"]
    assert corr["columns"] == ["a", "b", "c"]
    assert corr["matrix"][0][1] == pytest.approx(1.0) and corr["matrix"][0][2] == pytest.approx(
        -1.0
    )
    assert corr["matrix"][1][1] == pytest.approx(1.0)
    assert {(p["a"], p["b"]) for p in corr["top_pairs"]} == {("a", "b"), ("a", "c"), ("b", "c")}


def test_correlation_matches_numpy_on_seeded_data() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=200)
    y = 0.6 * x + rng.normal(size=200)
    z = rng.normal(size=200)
    cols = {"x": x, "y": y, "z": z}
    result = explore({k: [repr(float(v)) for v in arr] for k, arr in cols.items()})
    reference = np.corrcoef(np.vstack([x, y, z]))
    got = np.array(result["correlation"]["matrix"], dtype=float)
    assert np.allclose(got, reference, atol=1e-10)


def test_correlation_uses_rows_where_both_values_exist() -> None:
    a = ["1", "2", "3", "4", "5", ""]
    b = ["2", "1", "4", "3", "", "9"]
    result = explore({"a": a, "b": b})
    reference = np.corrcoef([1, 2, 3, 4], [2, 1, 4, 3])[0, 1]  # only rows 1-4 have both
    assert result["correlation"]["matrix"][0][1] == pytest.approx(reference, abs=1e-10)


def test_correlation_with_a_constant_column_is_none_not_nan() -> None:
    result = explore({"a": ["1", "2", "3"], "k": ["5", "5", "5"]})
    assert result["correlation"]["matrix"][0][1] is None
    assert result["correlation"]["top_pairs"] == []
    json.dumps(result, allow_nan=False)  # no NaN anywhere


def test_text_columns_are_not_in_the_correlation_and_one_numeric_column_is_fine() -> None:
    result = explore({"a": ["1", "2", "3"], "t": ["x", "y", "z"]})
    assert result["correlation"] == {"columns": ["a"], "matrix": [[None]], "top_pairs": []}


def test_top_pairs_are_ranked_by_absolute_correlation() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=100)
    cols = {"a": base, "b": base + rng.normal(scale=0.1, size=100), "c": rng.normal(size=100)}
    pairs = explore({k: [repr(float(v)) for v in arr] for k, arr in cols.items()})["correlation"][
        "top_pairs"
    ]
    strengths = [abs(p["r"]) for p in pairs]
    assert strengths == sorted(strengths, reverse=True)
    assert (pairs[0]["a"], pairs[0]["b"]) == ("a", "b")


# --- class balance -------------------------------------------------------------------------


def test_class_balance_for_classification() -> None:
    columns = {"x": [str(i) for i in range(10)], "y": ["a"] * 6 + ["b"] * 3 + ["c"]}
    balance = explore(columns, TaskConfig("y", "classification"))["class_balance"]
    assert balance["total_classes"] == 3 and balance["rows"] == 10
    assert balance["classes"] == [
        {"value": "a", "count": 6, "share": 0.6},
        {"value": "b", "count": 3, "share": 0.3},
        {"value": "c", "count": 1, "share": 0.1},
    ]
    assert balance["largest_class_share"] == 0.6 and balance["smallest_class_rows"] == 1
    assert balance["largest_to_smallest_ratio"] == 6.0


def test_class_balance_is_absent_for_regression_or_without_a_task() -> None:
    columns = {"x": [str(i) for i in range(10)], "y": [str(i) for i in range(10)]}
    assert explore(columns, TaskConfig("y", "regression"))["class_balance"] is None
    assert explore(columns)["class_balance"] is None


def test_class_balance_with_many_classes_is_summarised() -> None:
    labels = [f"c{i:03d}" for i in range(80)] * 2
    columns = {"x": [str(i) for i in range(160)], "y": labels}
    balance = explore(columns, TaskConfig("y", "classification"))["class_balance"]
    assert balance["total_classes"] == 80 and len(balance["classes"]) == 50
    assert balance["classes_not_listed"] == 30 and balance["smallest_class_rows"] == 2


def test_class_balance_counts_exact_text() -> None:
    columns = {"x": ["1", "2", "3"], "y": ["1", "1.0", "1"]}
    balance = explore(columns, TaskConfig("y", "classification"))["class_balance"]
    assert [(c["value"], c["count"]) for c in balance["classes"]] == [("1", 2), ("1.0", 1)]


# --- JSON, determinism, purity -------------------------------------------------------------


def test_output_is_deterministic_and_strict_json() -> None:
    columns = {
        "a": ["1", "2", "", "4"],
        "b": ["x", "y", "x", "y"],
        "c": ["0.5", "1.5", "2.5", "9"],
    }
    config = TaskConfig("b", "classification")
    first, second = explore(columns, config), explore(columns, config)
    assert first == second
    text = json.dumps(first, sort_keys=True, allow_nan=False)
    loaded = json.loads(text)
    assert set(loaded) == {
        "explore_format",
        "settings",
        "rows",
        "columns",
        "sample_rows",
        "histograms",
        "correlation",
        "class_balance",
    }
    assert loaded["explore_format"] == 1 and loaded["rows"] == 4


def test_exploring_does_not_change_the_frame() -> None:
    frame = read_text_frame(b"a,b\n1,x\n,y\n3,x\n")
    before = frame.copy()
    explore_frame(frame, profile_frame(frame), TaskConfig("b", "classification"))
    assert frame.equals(before)
