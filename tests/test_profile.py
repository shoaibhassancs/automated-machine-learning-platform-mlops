"""Profiling tests with small invented CSVs. Statistics are checked against the standard
`statistics` module (an independent reference), not against pandas."""

from __future__ import annotations

import csv
import io
import json
import statistics
from pathlib import Path

import pytest

from automl_ops.dataset_store import DatasetStoreError, FileDatasetStore
from automl_ops.profile import (
    ColumnProfile,
    DatasetProfile,
    ProfileSettings,
    profile_csv_bytes,
    profile_version,
    read_text_frame,
)


def csv_bytes(columns: dict[str, list[str]]) -> bytes:
    """Build canonical CSV bytes; an empty string is an empty (missing) cell."""
    names = list(columns)
    n = len(columns[names[0]])
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    for i in range(n):
        writer.writerow([columns[name][i] for name in names])
    return out.getvalue().encode("utf-8")


def profile_of(
    columns: dict[str, list[str]], settings: ProfileSettings | None = None
) -> DatasetProfile:
    data = csv_bytes(columns)
    return profile_csv_bytes(data) if settings is None else profile_csv_bytes(data, settings)


def col(profile: DatasetProfile, name: str) -> ColumnProfile:
    found = profile.column(name)
    assert found is not None
    return found


# --- reading policy: only empty cells are missing; text is preserved ----------------------


def test_only_empty_cells_are_missing_and_text_is_preserved() -> None:
    frame = read_text_frame(b"a,b\n1,NA\n,N/A\nnull,007\n")
    assert frame["a"].isna().tolist() == [False, True, False]
    assert frame["a"].tolist()[0] == "1" and frame["a"].tolist()[2] == "null"
    assert frame["b"].tolist() == ["NA", "N/A", "007"]  # not missing, not converted
    assert not frame["b"].isna().any()


def test_profiling_does_not_change_the_frame() -> None:
    frame = read_text_frame(b"a,b\n1,x\n,007\n")
    before = frame.copy()
    from automl_ops.profile import profile_frame

    profile_frame(frame)
    assert frame.equals(before)


# --- kinds ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "kind"),
    [
        (["1", "2", "3", "-4", "+5"], "integer"),
        (["1", "2.5", "3"], "float"),
        (["1e5", "2E-3", ".5", "7."], "float"),
        (["a", "b", "a"], "categorical"),
        (["1", "2", "x"], "mixed"),  # 2 of 3 look numeric
        (["007", "012", "5"], "code"),  # leading zeros: kept as text, not numbers
        (["", "", ""], "empty"),
        (["x", "x", "x"], "categorical"),
        (["1", "1", "1"], "integer"),
    ],
)
def test_kind_inference(values: list[str], kind: str) -> None:
    assert col(profile_of({"x": values}), "x").kind == kind


@pytest.mark.parametrize(
    "token",
    [
        "1_000",
        "nan",
        "inf",
        "-inf",
        "Infinity",
        "1e999",
        " 5",
        "5 ",
        "1,5",
        "٣",
        "0x1F",
        "--5",
        "1e",
    ],
)
def test_things_python_float_accepts_or_that_are_not_clean_numbers_are_not_numbers(
    token: str,
) -> None:
    c = col(profile_of({"x": ["1", "2", token]}), "x")
    assert c.kind == "mixed"
    assert token in [v for v, _ in c.non_numeric_examples]


def test_text_versus_categorical_boundary_is_max_categories() -> None:
    settings = ProfileSettings(max_categories=5)
    five = col(profile_of({"x": list("abcde")}, settings), "x")
    six = col(profile_of({"x": list("abcdef")}, settings), "x")
    assert five.kind == "categorical" and six.kind == "text"
    assert col(profile_of({"x": [f"v{i}" for i in range(50)]}), "x").kind == "categorical"
    assert col(profile_of({"x": [f"v{i}" for i in range(51)]}), "x").kind == "text"


def test_mixed_threshold_boundary() -> None:
    settings = ProfileSettings(mixed_min_numeric_share=0.5)
    half = col(profile_of({"x": ["1", "2", "a", "b"]}, settings), "x")
    below = col(profile_of({"x": ["1", "a", "b", "c"]}, settings), "x")
    assert half.kind == "mixed" and below.kind == "categorical"


# --- leading zeros and NA-like text --------------------------------------------------------


def test_leading_zero_values_are_detected_and_kept_as_text() -> None:
    values = ["007", "00", "-012", "0", "0.5", "7", "10", "102"]
    c = col(profile_of({"x": values}), "x")
    assert c.leading_zero_count == 3
    assert [v for v, _ in c.leading_zero_examples] == ["-012", "00", "007"]
    assert c.kind == "code"
    assert c.numeric is None  # not treated as numbers, no statistics
    assert dict(c.top_values)["007"] == 1  # original text preserved, never "7"


def test_leading_zero_in_a_mixed_column_is_still_reported() -> None:
    c = col(profile_of({"x": ["007", "5", "abc"]}), "x")
    assert c.kind == "mixed" and c.leading_zero_count == 1


def test_na_like_text_is_kept_counted_and_not_missing() -> None:
    c = col(profile_of({"x": ["NA", "na", "N/A", " NULL ", "?", "a", ""]}), "x")
    assert c.missing == 1  # only the empty cell is missing
    assert c.count == 6  # NA-like text still counts as a value
    assert c.na_like_count == 5
    assert dict(c.na_like_tokens) == {"NA": 1, "na": 1, "N/A": 1, " NULL ": 1, "?": 1}
    assert dict(c.top_values)["NA"] == 1  # original spelling preserved


def test_every_na_like_spelling_is_recognised_ignoring_case_and_spaces() -> None:
    tokens = ["NA", "na", "N/A", " NULL ", "None", "nil", "nan", "?", "-", "--"]
    c = col(profile_of({"x": tokens + ["a"]}), "x")
    assert c.na_like_count == len(tokens)  # the total is complete even though examples are limited
    assert len(c.na_like_tokens) == 5  # at most 5 spellings are listed as examples


def test_ordinary_values_that_merely_contain_na_are_not_flagged() -> None:
    c = col(profile_of({"x": ["Namibia", "banana", "Nan Smith", "n"]}), "x")
    assert c.na_like_tokens == ()


# --- statistics checked against the standard library ---------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        [1.5, 2.5, 10.0],
        [3.0, 1.0, 2.0, 2.0, 2.0],
        [-5.0, 0.0, 5.0, 10.0, 1000.0],
    ],
)
def test_numeric_statistics_match_the_statistics_module(values: list[float]) -> None:
    c = col(profile_of({"x": [repr(v) for v in values]}), "x")
    assert c.numeric is not None
    q = statistics.quantiles(values, n=4, method="inclusive")
    assert c.numeric.minimum == min(values) and c.numeric.maximum == max(values)
    assert c.numeric.mean == pytest.approx(statistics.mean(values), rel=1e-11)
    assert c.numeric.std == pytest.approx(statistics.stdev(values), rel=1e-11)
    assert c.numeric.median == pytest.approx(statistics.median(values), rel=1e-11)
    assert c.numeric.p25 == pytest.approx(q[0], rel=1e-11)
    assert c.numeric.p75 == pytest.approx(q[2], rel=1e-11)


def test_single_value_has_no_standard_deviation() -> None:
    c = col(profile_of({"x": ["4", "", ""]}), "x")
    assert c.numeric is not None and c.numeric.std is None and c.numeric.mean == 4.0


def test_integer_and_float_kinds_have_statistics_but_text_does_not() -> None:
    p = profile_of({"i": ["1", "2"], "f": ["1.5", "2"], "t": ["a", "b"]})
    assert col(p, "i").numeric is not None and col(p, "f").numeric is not None
    assert col(p, "t").numeric is None


# --- counts, ranking, dataset level --------------------------------------------------------


def test_missing_unique_and_top_values() -> None:
    p = profile_of({"x": ["b", "a", "b", "", "c", "b", "a", ""]})
    c = col(p, "x")
    assert (c.count, c.missing, c.unique) == (6, 2, 3)
    assert c.missing_pct == 25.0
    assert c.top_values == (("b", 3), ("a", 2), ("c", 1))  # by count, ties by value


def test_top_values_are_limited_and_ties_break_alphabetically() -> None:
    values = [f"v{i:02d}" for i in range(30)]
    c = col(profile_of({"x": values}, ProfileSettings(top_n=4)), "x")
    assert c.top_values == (("v00", 1), ("v01", 1), ("v02", 1), ("v03", 1))


def test_dataset_level_counts() -> None:
    p = profile_of({"a": ["1", "1", "2", ""], "b": ["x", "x", "y", "z"]})
    assert (p.rows, p.column_count) == (4, 2)
    assert p.duplicate_rows == 1 and p.rows_in_duplicate_groups == 2
    assert p.rows_with_missing == 1 and p.missing_cells == 1


def test_duplicate_rows_treat_missing_values_as_equal() -> None:
    p = profile_of({"a": ["", "", "1"], "b": ["x", "x", "x"]})
    assert p.duplicate_rows == 1


def test_non_ascii_text_is_preserved() -> None:
    c = col(profile_of({"x": ["café", "café", "naïve"]}), "x")
    assert c.top_values == (("café", 2), ("naïve", 1))


# --- JSON and determinism ------------------------------------------------------------------


def test_profile_is_deterministic_and_json_safe() -> None:
    data = csv_bytes({"a": ["1", "2.5", ""], "b": ["1", "NA", "007"], "c": ["", "", ""]})
    first, second = profile_csv_bytes(data), profile_csv_bytes(data)
    assert first == second
    text1 = json.dumps(first.to_dict(), sort_keys=True, allow_nan=False)
    text2 = json.dumps(second.to_dict(), sort_keys=True, allow_nan=False)
    assert text1 == text2
    loaded = json.loads(text1)
    assert loaded["profile_format"] == 1 and loaded["rows"] == 3
    assert [c["kind"] for c in loaded["column_profiles"]] == ["float", "mixed", "empty"]


def test_empty_column_has_no_statistics_and_zero_unique() -> None:
    c = col(profile_of({"a": ["1", "2"], "b": ["", ""]}), "b")
    assert (c.kind, c.count, c.missing, c.unique, c.numeric, c.top_values) == (
        "empty",
        0,
        2,
        0,
        None,
        (),
    )


# --- profiling a stored version ------------------------------------------------------------


def test_profile_version_reads_the_stored_file(tmp_path: Path) -> None:
    store = FileDatasetStore(tmp_path / "store")
    store.ingest("demo", b"id,name\n1,Ana\n2,Bo\n")
    vp = profile_version(store, "demo@v1")
    assert vp.version.ref == "demo@v1" and vp.profile.rows == 2
    assert vp.to_dict()["sha256"] == vp.version.sha256


def test_profile_refuses_a_stored_file_that_no_longer_matches_its_record(tmp_path: Path) -> None:
    store = FileDatasetStore(tmp_path / "store")
    store.ingest("demo", b"id,name\n1,Ana\n2,Bo\n")
    path = store.version_path("demo@v1")
    path.chmod(0o644)
    path.write_bytes(b"id,name\n1,Ana\n2,Eve\n")
    with pytest.raises(DatasetStoreError, match="does not match its recorded SHA-256"):
        profile_version(store, "demo@v1")


def test_profiling_never_changes_the_stored_version(tmp_path: Path) -> None:
    store = FileDatasetStore(tmp_path / "store")
    store.ingest("demo", b"id,name\n1,Ana\n2,Bo\n")
    path = store.version_path("demo@v1")
    before = (path.read_bytes(), sorted(p.name for p in path.parent.iterdir()))
    profile_version(store, "demo@v1")
    assert (path.read_bytes(), sorted(p.name for p in path.parent.iterdir())) == before
