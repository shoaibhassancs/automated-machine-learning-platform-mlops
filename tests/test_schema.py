"""Schema tests with small invented CSVs.

The key distinction under test: **structural failures** (make `ok` false) versus **observations**
about the training data (info only, never change `ok`). Unseen categories, out-of-range values, new
blanks, extra columns and non-whole numbers must be observations, never failures.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys

import pytest

from automl_ops.profile import profile_frame, read_text_frame
from automl_ops.schema import (
    SchemaError,
    SchemaReport,
    SchemaSpec,
    build_schema,
    check_csv_bytes,
    check_frame,
    profile_for_schema,
)
from automl_ops.task import TaskConfig


def csv_bytes(columns: dict[str, list[str]]) -> bytes:
    names = list(columns)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    for i in range(len(columns[names[0]])):
        writer.writerow([columns[name][i] for name in names])
    return out.getvalue().encode("utf-8")


def spec_from(
    columns: dict[str, list[str]], target: str = "target", excluded: tuple[str, ...] = ()
) -> SchemaSpec:
    frame = read_text_frame(csv_bytes(columns))
    return build_schema(profile_for_schema(frame), TaskConfig(target, "classification", excluded))


def check(spec: SchemaSpec, columns: dict[str, list[str]]) -> SchemaReport:
    return check_csv_bytes(spec, csv_bytes(columns))


def fail_codes(report: SchemaReport) -> list[tuple[str, str | None]]:
    return [(f.code, f.column) for f in report.failures]


def obs_codes(report: SchemaReport) -> list[tuple[str, str | None]]:
    return [(f.code, f.column) for f in report.observations]


def training(n: int = 100) -> dict[str, list[str]]:
    return {
        "age": [str(20 + i % 40) for i in range(n)],  # integer 20..59
        "score": [f"{(i % 10) / 4:.2f}" for i in range(n)],  # float 0.00..2.25
        "color": [("red", "green", "blue")[i % 3] for i in range(n)],  # categorical
        "target": [str(i % 2) for i in range(n)],
    }


# --- building the spec ---------------------------------------------------------------------


def test_spec_covers_feature_columns_only_and_records_what_was_observed() -> None:
    spec = spec_from(training(), excluded=("score",))
    assert [c.name for c in spec.columns] == ["age", "color"]  # no target, no excluded column
    assert (spec.target, spec.excluded, spec.training_rows) == ("target", ("score",), 100)
    age, color = spec.columns
    assert (age.kind, age.observed_min, age.observed_max) == ("integer", 20.0, 59.0)
    assert color.observed_categories == ("blue", "green", "red") and color.observed_min is None
    assert not age.had_missing


def test_spec_with_no_feature_columns_is_refused() -> None:
    columns = {"a": ["1", "2"], "target": ["x", "y"]}
    with pytest.raises(SchemaError, match="no feature columns"):
        spec_from(columns, excluded=("a",))


def test_categories_are_recorded_only_when_complete() -> None:
    columns = {"c": [f"v{i % 30}" for i in range(90)], "target": ["a", "b", "c"] * 30}
    frame = read_text_frame(csv_bytes(columns))
    config = TaskConfig("target", "classification")
    incomplete = build_schema(profile_frame(frame), config)  # default top_n = 20 < 30 categories
    assert incomplete.columns[0].observed_categories is None
    complete = build_schema(profile_for_schema(frame), config)
    assert complete.columns[0].observed_categories is not None
    assert len(complete.columns[0].observed_categories) == 30


def test_text_and_code_and_mixed_columns_get_no_category_set_or_range() -> None:
    columns = {
        "free": [f"text number {i}" for i in range(80)],  # > 50 distinct: kind text
        "code": ["007", "012", "5", "9"] * 20,
        "mixed": ["1", "2", "3", "x"] * 20,
        "target": ["a", "b"] * 40,
    }
    spec = spec_from(columns)
    for column in spec.columns:
        assert column.observed_categories is None and column.observed_min is None


def test_spec_round_trips_through_strict_json() -> None:
    spec = spec_from(training() | {"gap": [""] * 50 + ["x"] * 50}, excluded=("score",))
    text = json.dumps(spec.to_dict(), sort_keys=True, allow_nan=False)
    assert SchemaSpec.from_dict(json.loads(text)) == spec
    assert json.loads(text)["schema_format"] == 1


def test_unknown_schema_format_is_refused() -> None:
    raw = spec_from(training()).to_dict() | {"schema_format": 99}
    with pytest.raises(SchemaError, match="schema_format"):
        SchemaSpec.from_dict(raw)


# --- the training data checks cleanly against its own spec ---------------------------------


def test_training_data_passes_its_own_spec_with_no_findings() -> None:
    columns = training() | {
        "blank": ["", "a"] * 50,
        "long": [repr(0.1234567890123456 * (i + 1)) for i in range(100)],  # many digits
        "big": [str(10**15 + i) for i in range(100)],
    }
    spec = spec_from(columns)
    report = check(spec, columns)
    assert report.ok and report.failures == () and report.observations == ()


# --- structural failures -------------------------------------------------------------------


def test_missing_required_column_is_a_failure_but_target_and_excluded_are_not_required() -> None:
    spec = spec_from(training(), excluded=("score",))
    data = training()
    del data["target"], data["score"]
    assert check(spec, data).ok  # neither the target nor an excluded column is required
    del data["age"]
    report = check(spec, data)
    assert not report.ok and fail_codes(report) == [("missing_required_column", "age")]


def test_several_missing_columns_are_each_reported() -> None:
    spec = spec_from(training())
    report = check(spec, {"color": ["red"], "target": ["1"]})
    assert sorted(c for c, _ in fail_codes(report)) == ["missing_required_column"] * 2
    assert {col for _, col in fail_codes(report)} == {"age", "score"}


def test_duplicate_column_names_are_a_failure() -> None:
    spec = spec_from(training())
    data = b"age,age,score,color\n30,31,1.0,red\n"
    report = check_csv_bytes(spec, data)
    assert not report.ok and "duplicate_column_names" in [c for c, _ in fail_codes(report)]
    assert report.failures[0].details["names"] == ["age"]


@pytest.mark.parametrize("bad", ["abc", "1_000", "nan", "inf", "1e999", " 5", "1,5", "0x1F", "--5"])
def test_malformed_numbers_fail_in_numeric_columns(bad: str) -> None:
    spec = spec_from(training())
    data = training()
    data["age"][3] = bad
    report = check(spec, data)
    assert not report.ok
    (failure,) = [f for f in report.failures if f.code == "malformed_numeric"]
    assert failure.column == "age" and failure.severity == "error"
    assert failure.details["cells"] == 1 and failure.details["first_data_row"] == 4
    assert failure.details["examples"] == [{"value": bad, "count": 1}]


def test_malformed_numbers_are_counted_with_examples() -> None:
    spec = spec_from(training())
    data = training()
    for i in (0, 1, 2):
        data["score"][i] = "n/a?"
    data["score"][50] = "abc"
    (failure,) = check(spec, data).failures
    assert failure.column == "score" and failure.details["cells"] == 4
    assert failure.details["examples"][0] == {"value": "n/a?", "count": 3}
    assert failure.details["first_data_row"] == 1


@pytest.mark.parametrize("good", ["-3.5", ".5", "1e5", "+7", "7.", "0", "-0.0", "2E-3"])
def test_valid_number_spellings_pass(good: str) -> None:
    spec = spec_from(training())
    data = training()
    data["score"][5] = good
    assert check(spec, data).ok


def test_blank_cells_in_a_numeric_column_are_not_malformed() -> None:
    spec = spec_from(training())
    data = training()
    data["age"][2] = ""
    assert check(spec, data).ok


def test_text_columns_accept_any_text_and_the_target_is_not_checked() -> None:
    spec = spec_from(training())
    data = training()
    data["color"][0] = "nan"
    data["target"][0] = "abc"
    assert check(spec, data).ok


def test_failures_in_several_columns_are_all_reported_in_column_order() -> None:
    spec = spec_from(training())
    data = training()
    data["score"][0] = "x"
    data["age"][0] = "y"
    assert fail_codes(check(spec, data)) == [
        ("malformed_numeric", "age"),
        ("malformed_numeric", "score"),
    ]


# --- observations: never failures ----------------------------------------------------------


def test_values_outside_the_training_range_are_observations_not_failures() -> None:
    spec = spec_from(training())
    data = training()
    data["age"][0], data["age"][1], data["age"][2] = "19", "60", "250"
    report = check(spec, data)
    assert report.ok and report.failures == ()
    (obs,) = [f for f in report.observations if f.code == "out_of_observed_range"]
    assert obs.severity == "info" and obs.column == "age"
    assert (obs.details["below"], obs.details["above"]) == (1, 2)
    assert (obs.details["observed_min"], obs.details["observed_max"]) == (20.0, 59.0)
    assert (obs.details["data_min"], obs.details["data_max"]) == (19.0, 250.0)


def test_range_boundaries_are_inclusive_values_equal_to_min_or_max_are_not_flagged() -> None:
    spec = spec_from(training())
    data = training()
    data["age"][0], data["age"][1] = "20", "59"  # exactly the observed min and max
    assert check(spec, data).observations == ()


def test_numbers_outside_the_range_in_other_spellings_are_found() -> None:
    spec = spec_from(training())
    data = training()
    data["score"][0], data["score"][1] = "-0.5", "1e3"
    (obs,) = check(spec, data).observations
    assert obs.code == "out_of_observed_range" and (obs.details["below"], obs.details["above"]) == (
        1,
        1,
    )


def test_unseen_categories_are_observations_not_failures() -> None:
    spec = spec_from(training())
    data = training()
    data["color"][0], data["color"][1], data["color"][2] = "purple", "purple", "teal"
    report = check(spec, data)
    assert report.ok
    (obs,) = [f for f in report.observations if f.code == "unseen_categories"]
    assert obs.column == "color" and obs.details["cells"] == 3 and obs.details["distinct"] == 2
    assert obs.details["examples"][0] == {"value": "purple", "count": 2}


def test_an_entirely_new_set_of_categories_still_passes() -> None:
    spec = spec_from(training())
    data = training()
    data["color"] = [f"new{i}" for i in range(100)]
    report = check(spec, data)
    assert report.ok and ("unseen_categories", "color") in obs_codes(report)


def test_new_blanks_are_observed_only_when_training_had_none() -> None:
    columns = training() | {"gappy": ["", "a", "b", "a"] * 25}
    spec = spec_from(columns)
    data = columns | {"age": ["", *columns["age"][1:]], "gappy": ["", *columns["gappy"][1:]]}
    report = check(spec, data)
    assert report.ok
    assert ("new_blanks", "age") in obs_codes(report)
    assert ("new_blanks", "gappy") not in obs_codes(report)  # training had blanks here


def test_non_whole_numbers_in_an_integer_column_are_observed() -> None:
    spec = spec_from(training())
    data = training()
    data["age"][0], data["age"][1] = "30.5", "31.0"
    report = check(spec, data)
    assert report.ok
    (obs,) = [f for f in report.observations if f.code == "non_integer_values"]
    assert obs.column == "age" and obs.details["cells"] == 2


def test_extra_columns_are_observed_and_target_and_excluded_are_not_extra() -> None:
    spec = spec_from(training(), excluded=("score",))
    data = training() | {"surprise": ["x"] * 100}
    report = check(spec, data)
    assert report.ok
    (obs,) = [f for f in report.observations if f.code == "extra_columns"]
    assert obs.details["columns"] == ["surprise"]  # not 'target', not 'score'


def test_high_cardinality_text_has_no_category_observation() -> None:
    columns = {"free": [f"t{i}" for i in range(80)], "target": ["a", "b"] * 40}
    spec = spec_from(columns)
    report = check(spec, {"free": [f"other{i}" for i in range(80)], "target": ["a", "b"] * 40})
    assert report.ok and report.observations == ()


def test_observations_never_make_a_report_fail_however_many_there_are() -> None:
    spec = spec_from(training())
    data = {
        "age": ["500"] * 100,
        "score": ["-9"] * 100,
        "color": ["zzz"] * 100,
        "extra": ["q"] * 100,
        "target": ["1"] * 100,
    }
    report = check(spec, data)
    assert report.ok and len(report.observations) == 4 and report.failures == ()


def test_a_shifted_copy_passes_structurally_and_reports_its_shift() -> None:
    """The shifted-copy pattern: well-formed values moved upward, same categories."""
    columns = training()
    spec = spec_from(columns)
    shifted = {
        "age": [str(int(v) + 30) for v in columns["age"]],
        "score": [f"{float(v) + 1.5:.2f}" for v in columns["score"]],
        "color": ["red"] * 60 + columns["color"][60:],  # proportions changed, categories not
        "target": columns["target"],
    }
    report = check(spec, shifted)
    assert report.ok and report.failures == ()
    assert {c for c, _ in obs_codes(report)} == {"out_of_observed_range"}
    assert {col for _, col in obs_codes(report)} == {"age", "score"}


# --- result shape, purity, errors ----------------------------------------------------------


def test_report_json_shape_and_determinism() -> None:
    spec = spec_from(training())
    data = training()
    data["age"][0] = "x"
    data["color"][0] = "zzz"
    first, second = check(spec, data), check(spec, data)
    assert first == second
    d = json.loads(json.dumps(first.to_dict(), sort_keys=True, allow_nan=False))
    assert set(d) == {"ok", "rows", "structural_failures", "observations"}
    assert d["ok"] is False and d["rows"] == 100
    assert {f["severity"] for f in d["structural_failures"]} == {"error"}
    assert {f["severity"] for f in d["observations"]} == {"info"}


def test_checking_does_not_change_the_frame() -> None:
    spec = spec_from(training())
    data = training()
    data["age"][0] = "x"
    frame = read_text_frame(csv_bytes(data))
    before = frame.copy()
    check_frame(spec, frame)
    assert frame.equals(before)
    assert frame["age"].iloc[0] == "x"  # original text, nothing converted


@pytest.mark.parametrize("data", [b"", b"\xff\xfe\x00bad", b"a,b\n1,2,3,4,5\n6\n"])
def test_unreadable_data_raises_schema_error(data: bytes) -> None:
    spec = spec_from(training())
    with pytest.raises(SchemaError, match="could not be read as CSV"):
        check_csv_bytes(spec, data)


def test_header_only_data_has_no_rows_and_only_missing_column_failures_if_any() -> None:
    spec = spec_from(training())
    ok = check_csv_bytes(spec, b"age,score,color\n")
    assert ok.ok and ok.rows == 0
    bad = check_csv_bytes(spec, b"age\n")
    assert {c for _, c in fail_codes(bad)} == {"score", "color"}


def test_pandera_is_only_imported_when_a_check_runs() -> None:
    code = (
        "import sys\n"
        "import automl_ops.schema as s\n"
        "assert 'pandera' not in sys.modules, 'imported too early'\n"
        "spec = s.SchemaSpec(columns=(), target=None, excluded=(), training_rows=0)\n"
        "s.check_csv_bytes(spec, b'a\\n1\\n')\n"
        "assert 'pandera' in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
