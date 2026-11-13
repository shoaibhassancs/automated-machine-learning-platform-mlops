"""Task configuration tests with small invented CSVs: the suggestion, every blocking condition, each
warning, and the provisional thresholds exactly at their boundaries."""

from __future__ import annotations

import csv
import io
import json

import pytest

from automl_ops.profile import DatasetProfile, profile_frame, read_text_frame
from automl_ops.task import (
    TaskConfig,
    TaskReport,
    TaskSettings,
    TaskType,
    check_task_config,
    class_counts,
    suggest_task,
)


def build(columns: dict[str, list[str]]) -> tuple[DatasetProfile, dict[str, int], bytes]:
    names = list(columns)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    for i in range(len(columns[names[0]])):
        writer.writerow([columns[name][i] for name in names])
    data = out.getvalue().encode("utf-8")
    frame = read_text_frame(data)
    return profile_frame(frame), class_counts(frame, names[-1]), data


def check(
    columns: dict[str, list[str]],
    target: str,
    task: TaskType,
    excluded: tuple[str, ...] = (),
    settings: TaskSettings | None = None,
) -> TaskReport:
    profile, _, data = build(columns)
    counts = class_counts(read_text_frame(data), target) if target in columns else {}
    config = TaskConfig(target=target, task=task, excluded=excluded)
    if settings is None:
        return check_task_config(profile, counts, config)
    return check_task_config(profile, counts, config, settings)


def codes(report: TaskReport, severity: str | None = None) -> list[str]:
    return [f.code for f in report.findings if severity in (None, f.severity)]


def with_classes(sizes: dict[str, int]) -> dict[str, list[str]]:
    labels = [label for label, n in sizes.items() for _ in range(n)]
    return {"x": [str(i) for i in range(len(labels))], "y": labels}


# --- suggestion (advisory) -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "task"),
    [
        (["a", "b", "a", "c"], "classification"),
        (["1.5", "2.5", "3"], "regression"),
        (["0", "1"] * 10, "classification"),
        ([str(i % 20) for i in range(60)], "classification"),  # exactly 20 distinct: not more
        ([str(i % 21) for i in range(63)], "regression"),  # 21 distinct: more than 20
        (["007", "012", "5"], "classification"),  # a code is not a quantity
        (["1", "2", "x"], "classification"),  # mixed: not purely numeric
    ],
)
def test_suggestion_rule(values: list[str], task: str) -> None:
    profile, _, _ = build({"x": [str(i) for i in range(len(values))], "y": values})
    suggestion = suggest_task(profile, "y")
    assert suggestion is not None and suggestion.task == task and suggestion.reason


def test_no_suggestion_without_a_usable_target() -> None:
    profile, _, _ = build({"x": ["1", "2"], "y": ["", ""]})
    assert suggest_task(profile, "y") is None and suggest_task(profile, "nope") is None


def test_suggestion_cutoff_follows_the_setting() -> None:
    profile, _, _ = build({"x": [str(i) for i in range(30)], "y": [str(i % 5) for i in range(30)]})
    suggestion = suggest_task(profile, "y", TaskSettings(whole_number_regression_min_distinct=4))
    assert suggestion is not None and suggestion.task == "regression"


def test_the_suggestion_can_be_overridden_without_error() -> None:
    columns = {"x": [str(i) for i in range(60)], "y": [str(i % 30) for i in range(60)]}
    report = check(columns, "y", "classification")  # suggestion would be regression
    assert report.suggestion is not None and report.suggestion.task == "regression"
    assert report.ok and "task_differs_from_suggestion" in codes(report, "info")


# --- target and column errors ---------------------------------------------------------------


def test_clean_configuration_has_no_findings() -> None:
    report = check(with_classes({"a": 30, "b": 30}), "y", "classification")
    assert report.findings == () and report.ok


def test_unknown_target_is_an_error() -> None:
    report = check(with_classes({"a": 10, "b": 10}), "nope", "classification")
    assert "unknown_target" in codes(report, "error") and not report.ok


def test_missing_target_values_and_constant_target_are_errors() -> None:
    missing = check({"x": ["1", "2", "3"], "y": ["a", "", "b"]}, "y", "classification")
    assert "target_missing" in codes(missing, "error")
    constant = check({"x": ["1", "2", "3"], "y": ["a", "a", "a"]}, "y", "classification")
    assert "target_constant" in codes(constant, "error")


def test_target_listed_as_excluded_is_an_error() -> None:
    report = check(with_classes({"a": 10, "b": 10}), "y", "classification", excluded=("y",))
    assert "target_excluded" in codes(report, "error")


def test_unknown_excluded_column_is_an_error() -> None:
    report = check(with_classes({"a": 10, "b": 10}), "y", "classification", excluded=("zzz",))
    assert ("unknown_excluded_column", "zzz") in [(f.code, f.column) for f in report.findings]
    assert not report.ok


def test_no_feature_columns_left_is_an_error() -> None:
    columns = with_classes({"a": 10, "b": 10})
    assert "no_feature_columns" in codes(check(columns, "y", "classification", ("x",)), "error")
    assert "no_feature_columns" not in codes(check(columns, "y", "classification"))


def test_feature_columns_exclude_target_and_excluded() -> None:
    profile, _, _ = build({"a": ["1", "2"], "b": ["x", "y"], "c": ["p", "q"], "t": ["u", "v"]})
    config = TaskConfig("t", "classification", ("b",))
    assert config.feature_columns(profile) == ("a", "c")


# --- classification: class size and class count --------------------------------------------


def test_a_class_with_one_row_is_an_error_but_two_rows_is_not() -> None:
    one = check(with_classes({"a": 30, "b": 1}), "y", "classification")
    assert "class_too_small" in codes(one, "error") and not one.ok
    assert "small_class" not in codes(one)  # reported once, as the error
    two = check(with_classes({"a": 30, "b": 2}), "y", "classification")
    assert "class_too_small" not in codes(two) and two.ok


@pytest.mark.parametrize(("smallest", "warns"), [(4, True), (5, False)])
def test_small_class_warning_boundary_is_below_five(smallest: int, warns: bool) -> None:
    report = check(with_classes({"a": 30, "b": smallest}), "y", "classification")
    assert ("small_class" in codes(report, "warning")) == warns and report.ok


@pytest.mark.parametrize(("classes", "warns"), [(50, False), (51, True)])
def test_many_classes_is_a_warning_never_an_error(classes: int, warns: bool) -> None:
    report = check(with_classes({f"c{i:02d}": 6 for i in range(classes)}), "y", "classification")
    assert ("many_classes" in codes(report, "warning")) == warns
    assert report.ok  # a high class count never rejects the configuration


def test_numeric_labels_with_many_distinct_values_are_accepted_as_classes() -> None:
    labels = [str(i % 120) for i in range(1200)]
    report = check({"x": [str(i) for i in range(1200)], "y": labels}, "y", "classification")
    assert report.ok and "many_classes" in codes(report, "warning")


def test_decimal_class_labels_are_info_and_keep_exact_text() -> None:
    columns = {"x": [str(i) for i in range(40)], "y": ["1.5", "2.5", "1", "1.0"] * 10}
    report = check(columns, "y", "classification")
    assert report.ok and "decimal_class_labels" in codes(report, "info")
    assert class_counts(read_text_frame(b"y\n1\n1.0\n1\n"), "y") == {"1": 2, "1.0": 1}


def test_class_limits_follow_the_settings() -> None:
    settings = TaskSettings(
        max_classes_warning=2, min_class_size_warning=10, min_class_size_error=3
    )
    columns = with_classes({"a": 40, "b": 12, "c": 12})
    assert codes(check(columns, "y", "classification", settings=settings)) == ["many_classes"]
    columns = with_classes({"a": 40, "b": 2})
    assert "class_too_small" in codes(check(columns, "y", "classification", settings=settings))


# --- regression ----------------------------------------------------------------------------


def test_regression_on_a_numeric_target_is_fine() -> None:
    columns = {"x": [str(i) for i in range(60)], "y": [f"{i}.5" for i in range(60)]}
    assert check(columns, "y", "regression").findings == ()


@pytest.mark.parametrize("values", [["a", "b"] * 20, ["1", "2", "x"] * 14, ["007", "012"] * 20])
def test_regression_needs_a_numeric_target(values: list[str]) -> None:
    columns = {"x": [str(i) for i in range(len(values))], "y": values}
    report = check(columns, "y", "regression")
    assert "target_not_numeric" in codes(report, "error") and not report.ok


def test_regression_with_few_distinct_values_is_only_info() -> None:
    columns = {"x": [str(i) for i in range(40)], "y": [str(i % 3) for i in range(40)]}
    report = check(columns, "y", "regression")
    # both are informational: few distinct values, and the suggestion would have been classification
    assert report.ok and set(codes(report, "info")) == {
        "few_distinct_values_for_regression",
        "task_differs_from_suggestion",
    }
    assert codes(report, "error") == [] and codes(report, "warning") == []


def test_class_rules_do_not_apply_to_regression() -> None:
    columns = {"x": [str(i) for i in range(40)], "y": [str(i % 2) for i in range(39)] + ["9"]}
    report = check(columns, "y", "regression")
    assert "class_too_small" not in codes(report) and "small_class" not in codes(report)


# --- ordering, determinism, JSON -----------------------------------------------------------


def test_findings_are_ordered_errors_first() -> None:
    columns = with_classes({"a": 30, "b": 1})
    report = check(columns, "y", "regression", excluded=("zzz",))
    order = {"error": 0, "warning": 1, "info": 2}
    ranks = [order[f.severity] for f in report.findings]
    assert ranks == sorted(ranks) and report.errors >= 2


def test_report_is_deterministic_and_json_safe() -> None:
    columns = with_classes({"a": 30, "b": 3})
    first = check(columns, "y", "classification")
    second = check(columns, "y", "classification")
    assert first == second
    d = json.loads(json.dumps(first.to_dict(), sort_keys=True, allow_nan=False))
    assert set(d) == {
        "ok",
        "config",
        "suggestion",
        "errors",
        "warnings",
        "infos",
        "settings",
        "findings",
    }
    assert d["config"] == {"target": "y", "task": "classification", "excluded": []}


def test_checking_does_not_change_the_inputs() -> None:
    profile, _, data = build(with_classes({"a": 30, "b": 3}))
    frame = read_text_frame(data)
    before = frame.copy()
    counts = class_counts(frame, "y")
    snapshot = dict(counts)
    check_task_config(profile, counts, TaskConfig("y", "classification"))
    assert frame.equals(before) and counts == snapshot
