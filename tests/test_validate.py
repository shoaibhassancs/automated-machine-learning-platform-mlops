"""Validation rule tests with small invented CSVs: each rule fires, does not fire, and sits exactly
where its provisional threshold says."""

from __future__ import annotations

import csv
import io
import json

import pytest

from automl_ops.profile import DatasetProfile, ProfileSettings, profile_csv_bytes
from automl_ops.validate import Finding, Thresholds, ValidationReport, validate


def make_profile(
    columns: dict[str, list[str]], settings: ProfileSettings | None = None
) -> DatasetProfile:
    names = list(columns)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    for i in range(len(columns[names[0]])):
        writer.writerow([columns[name][i] for name in names])
    data = out.getvalue().encode("utf-8")
    return profile_csv_bytes(data) if settings is None else profile_csv_bytes(data, settings)


def codes(report: ValidationReport, severity: str | None = None) -> list[tuple[str, str | None]]:
    return [(f.code, f.column) for f in report.findings if severity in (None, f.severity)]


def varied(n: int, prefix: str = "v") -> list[str]:
    """n values with enough variety (not constant, not ID-like) for a categorical column."""
    return [f"{prefix}{i % 7}" for i in range(n)]


def clean(n: int = 200) -> dict[str, list[str]]:
    return {
        "num": [str(i % 37) for i in range(n)],
        "cat": varied(n),
        "target": [str(i % 2) for i in range(n)],
    }


def finding(report: ValidationReport, code: str, column: str | None = None) -> Finding:
    for f in report.findings:
        if f.code == code and f.column == column:
            return f
    raise AssertionError(f"no {code} for {column}: {codes(report)}")


# --- clean data ----------------------------------------------------------------------------


def test_clean_data_has_no_findings() -> None:
    report = validate(make_profile(clean()), target="target")
    assert report.findings == ()
    assert report.ok and (report.errors, report.warnings, report.infos) == (0, 0, 0)


# --- dataset-level rules and their boundaries ----------------------------------------------


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (5, ("too_few_rows", "error")),
        (29, ("too_few_rows", "error")),
        (30, ("few_rows", "warning")),
        (99, ("few_rows", "warning")),
        (100, None),
        (500, None),
    ],
)
def test_row_count_thresholds(rows: int, expected: tuple[str, str] | None) -> None:
    report = validate(make_profile(clean(rows)))
    row_findings = [
        (f.code, f.severity) for f in report.findings if f.code in ("too_few_rows", "few_rows")
    ]
    assert row_findings == ([expected] if expected else [])
    assert report.ok == (expected is None or expected[1] != "error")


def test_duplicate_rows_are_info_and_counted() -> None:
    columns = clean(200)
    columns = {k: v + v[:10] for k, v in columns.items()}  # 10 exact copies appended
    report = validate(make_profile(columns))
    f = finding(report, "duplicate_rows")
    assert f.severity == "info" and f.details["extra_copies"] >= 10
    assert report.ok


# --- column rules --------------------------------------------------------------------------


def test_all_missing_column() -> None:
    columns = clean() | {"blank": [""] * 200}
    report = validate(make_profile(columns))
    assert finding(report, "all_missing", "blank").severity == "warning"
    assert ("high_missing", "blank") not in codes(report)  # reported once, as all_missing


@pytest.mark.parametrize(("missing_rows", "fires"), [(80, False), (81, True), (120, True)])
def test_high_missing_boundary_is_strictly_more_than_40_percent(
    missing_rows: int, fires: bool
) -> None:
    values = [""] * missing_rows + varied(200 - missing_rows)
    report = validate(make_profile(clean() | {"m": values}))
    assert (("high_missing", "m") in codes(report)) == fires


def test_high_missing_reports_the_numbers() -> None:
    report = validate(make_profile(clean() | {"m": [""] * 100 + varied(100)}))
    f = finding(report, "high_missing", "m")
    assert f.details == {"missing": 100, "missing_pct": 50.0}


def test_constant_column() -> None:
    report = validate(make_profile(clean() | {"k": ["same"] * 200}))
    f = finding(report, "constant_column", "k")
    assert f.severity == "warning" and f.details["value"] == "same"


def test_constant_ignores_missing_cells() -> None:
    report = validate(make_profile(clean() | {"k": ["same"] * 100 + [""] * 100}))
    assert ("constant_column", "k") in codes(report)


def test_mixed_types_lists_examples() -> None:
    values = [str(i) for i in range(190)] + ["n/a?"] * 5 + ["abc"] * 5
    report = validate(make_profile(clean() | {"m": values}))
    f = finding(report, "mixed_types", "m")
    assert f.severity == "warning"
    assert {e["value"] for e in f.details["non_numeric_examples"]} == {"n/a?", "abc"}


def test_na_like_values_warn_and_are_not_treated_as_missing() -> None:
    values = ["NA"] * 3 + ["N/A"] * 2 + varied(195)
    report = validate(make_profile(clean() | {"c": values}))
    f = finding(report, "na_like_values", "c")
    assert f.details["cells"] == 5
    assert "only empty cells count as missing" in f.message
    assert ("high_missing", "c") not in codes(report)


def test_na_like_total_is_complete_beyond_the_example_limit() -> None:
    values = ["NA", "na", "N/A", "null", "none", "nil", "nan", "?"] * 3 + varied(176)
    report = validate(make_profile(clean() | {"c": values}))
    assert finding(report, "na_like_values", "c").details["cells"] == 24


def test_leading_zero_values_warn_and_column_is_not_numeric() -> None:
    values = ["007", "012", "100", "5"] * 50
    profile = make_profile(clean() | {"zip": values})
    report = validate(profile)
    f = finding(report, "leading_zero_values", "zip")
    assert f.severity == "warning" and f.details["kind"] == "code"
    assert "may be codes or identifiers" in f.message and "original text is kept" in f.message
    zip_profile = profile.column("zip")
    assert zip_profile is not None and zip_profile.kind == "code" and zip_profile.numeric is None


def test_zero_alone_and_decimals_do_not_trigger_the_leading_zero_rule() -> None:
    report = validate(make_profile(clean() | {"z": ["0", "0.5", "10", "7"] * 50}))
    assert ("leading_zero_values", "z") not in codes(report)


@pytest.mark.parametrize(("rows", "fires"), [(49, False), (50, True)])
def test_id_like_integer_needs_every_value_unique_and_enough_rows(rows: int, fires: bool) -> None:
    columns = {
        "id": [str(i) for i in range(rows)],
        "cat": varied(rows),
        "target": [str(i % 2) for i in range(rows)],
    }
    assert (("id_like_integer", "id") in codes(validate(make_profile(columns)))) == fires


def test_integer_column_with_one_repeat_is_not_id_like() -> None:
    ids = [str(i) for i in range(199)] + ["0"]
    assert ("id_like_integer", "id") not in codes(validate(make_profile(clean() | {"id": ids})))


def test_float_column_with_all_distinct_values_is_normal() -> None:
    floats = [f"{i}.5" for i in range(200)]
    assert ("id_like_integer", "f") not in codes(validate(make_profile(clean() | {"f": floats})))


def test_id_like_text_boundary() -> None:
    # 200 values: 180 distinct (90%) fires; 179 distinct (89.5%) does not.
    def column(distinct: int) -> list[str]:
        return [f"id{i}" for i in range(distinct)] + ["id0"] * (200 - distinct)

    assert ("id_like_text", "t") in codes(validate(make_profile(clean() | {"t": column(180)})))
    assert ("id_like_text", "t") not in codes(validate(make_profile(clean() | {"t": column(179)})))


def test_many_categories_for_text_that_is_not_id_like() -> None:
    values = [f"cat{i % 60}" for i in range(300)]  # 60 distinct over 300 rows = 20%
    report = validate(make_profile(clean(300) | {"t": values}))
    f = finding(report, "many_categories", "t")
    assert f.details == {"unique": 60, "limit": 50}
    assert ("id_like_text", "t") not in codes(report)


def test_fifty_categories_is_not_many_categories() -> None:
    values = [f"cat{i % 50}" for i in range(300)]
    assert ("many_categories", "t") not in codes(validate(make_profile(clean(300) | {"t": values})))


def test_custom_thresholds_are_used_and_reported() -> None:
    profile = make_profile(clean(200))
    loose = validate(profile)
    strict = validate(profile, thresholds=Thresholds(min_rows_warning=500))
    assert ("few_rows", None) not in codes(loose) and ("few_rows", None) in codes(strict)
    assert strict.to_dict()["thresholds"]["min_rows_warning"] == 500


# --- target quality (and only quality) -----------------------------------------------------


def test_target_that_does_not_exist() -> None:
    report = validate(make_profile(clean()), target="nope")
    f = finding(report, "unknown_target", "nope")
    assert f.severity == "error" and not report.ok
    assert f.details["columns"] == ["num", "cat", "target"]


def test_target_with_missing_values() -> None:
    columns = clean(200)
    columns["target"] = [""] * 12 + columns["target"][12:]
    report = validate(make_profile(columns), target="target")
    f = finding(report, "target_missing", "target")
    assert f.severity == "error" and f.details["missing"] == 12 and not report.ok


def test_constant_target() -> None:
    columns = clean(200)
    columns["target"] = ["yes"] * 200
    report = validate(make_profile(columns), target="target")
    assert finding(report, "target_constant", "target").severity == "error"


def test_without_a_target_no_target_rules_run() -> None:
    columns = clean(200)
    columns["target"] = [""] * 200
    report = validate(make_profile(columns))
    assert all(
        f.code not in ("target_missing", "target_constant", "unknown_target")
        for f in report.findings
    )
    assert report.target is None


def test_task_suitability_is_not_checked_here() -> None:
    """Class sizes and task type are Step 5's job: a target with a one-row class passes."""
    columns = clean(200)
    columns["target"] = ["common"] * 199 + ["rare"]
    report = validate(make_profile(columns), target="target")
    assert report.ok
    assert [f for f in report.findings if f.column == "target"] == []


def test_target_findings_do_not_stop_column_findings() -> None:
    columns = clean(200) | {"k": ["same"] * 200}
    report = validate(make_profile(columns), target="nope")
    assert ("constant_column", "k") in codes(report) and ("unknown_target", "nope") in codes(report)


# --- ordering, determinism, JSON, purity ---------------------------------------------------


def test_findings_are_ordered_errors_first_then_by_column_position() -> None:
    columns = {
        "a": ["same"] * 40,
        "b": [""] * 40,
        "t": [""] * 40,
    }
    report = validate(make_profile(columns), target="t")
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key=lambda s: {"error": 0, "warning": 1, "info": 2}[s])
    warnings = [(f.column, f.code) for f in report.findings if f.severity == "warning"]
    assert warnings[0] == (None, "few_rows")  # dataset-level first
    order = {"a": 0, "b": 1, "t": 2}
    column_positions = [order[c] for c, _ in warnings[1:] if c is not None]
    assert column_positions == sorted(column_positions)


def test_validation_is_deterministic_and_does_not_change_the_profile() -> None:
    profile = make_profile(clean() | {"k": ["same"] * 200, "z": ["007", "5"] * 100})
    before = json.dumps(profile.to_dict(), sort_keys=True)
    first, second = validate(profile, target="target"), validate(profile, target="target")
    assert first == second
    assert json.dumps(profile.to_dict(), sort_keys=True) == before
    a = json.dumps(first.to_dict(), sort_keys=True, allow_nan=False)
    assert a == json.dumps(second.to_dict(), sort_keys=True, allow_nan=False)


def test_report_json_shape() -> None:
    report = validate(make_profile(clean() | {"k": ["same"] * 200}), target="target")
    d = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert set(d) == {"ok", "target", "errors", "warnings", "infos", "thresholds", "findings"}
    assert d["ok"] is True and d["target"] == "target"
    assert set(d["findings"][0]) == {"code", "severity", "column", "message", "details"}
