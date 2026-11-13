"""Profile and validate the prepared Adult and Abalone demo files (skipped until `make demo-data`).

These checks cross-check the profile against numbers recorded independently in Step 2 and against
plain standard-library calculations. The demo files are only read, never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from automl_ops.dataset_store import FileDatasetStore
from automl_ops.profile import (
    DatasetProfile,
    profile_csv_bytes,
    profile_frame,
    profile_version,
    read_text_frame,
)
from automl_ops.validate import validate

DATA = Path(__file__).resolve().parents[1] / "data"
DEMO = DATA / "demo"
MANIFEST = DEMO / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="demo data not prepared; run `make demo-data` first"
)

ADULT_INTEGERS = {
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
}
ABALONE_FLOATS = {
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
}
TARGETS = {"adult": "income", "abalone": "Rings"}
ROLES = ("training_pool", "monitoring_stream", "monitoring_shifted")


def manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data


def demo_path(key: str, role: str) -> Path:
    return DEMO / str(manifest()["datasets"][key]["files"][role]["file"])


def profile_of_file(key: str, role: str) -> DatasetProfile:
    return profile_csv_bytes(demo_path(key, role).read_bytes())


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def profiles() -> dict[tuple[str, str], DatasetProfile]:
    return {(k, r): profile_of_file(k, r) for k in TARGETS for r in ROLES}


# --- counts and kinds against the Step 2 manifest ------------------------------------------


@pytest.mark.parametrize("key", sorted(TARGETS))
@pytest.mark.parametrize("role", ROLES)
def test_rows_and_columns_match_the_manifest(
    profiles: dict[tuple[str, str], DatasetProfile], key: str, role: str
) -> None:
    meta = manifest()["datasets"][key]["files"][role]
    p = profiles[(key, role)]
    assert p.rows == meta["rows"]
    assert [c.name for c in p.columns] == meta["columns"]


@pytest.mark.parametrize("role", ROLES)
def test_adult_column_kinds(profiles: dict[tuple[str, str], DatasetProfile], role: str) -> None:
    kinds = {c.name: c.kind for c in profiles[("adult", role)].columns}
    assert {n for n, k in kinds.items() if k == "integer"} == ADULT_INTEGERS
    assert {k for n, k in kinds.items() if n not in ADULT_INTEGERS} == {"categorical"}


@pytest.mark.parametrize("role", ROLES)
def test_abalone_column_kinds(profiles: dict[tuple[str, str], DatasetProfile], role: str) -> None:
    kinds = {c.name: c.kind for c in profiles[("abalone", role)].columns}
    assert {n for n, k in kinds.items() if k == "float"} == ABALONE_FLOATS
    assert kinds["Rings"] == "integer" and kinds["Sex"] == "categorical"


def test_missing_values_agree_with_step_2_totals(
    profiles: dict[tuple[str, str], DatasetProfile],
) -> None:
    recorded = manifest()["datasets"]["adult"]["preparation_report"]["missing_by_column"]
    for column in ("workclass", "occupation", "native-country"):
        pool = profiles[("adult", "training_pool")].column(column)
        stream = profiles[("adult", "monitoring_stream")].column(column)
        assert pool is not None and stream is not None
        assert pool.missing + stream.missing == recorded[column]  # independent Step 2 number
    with_missing = {c.name for c in profiles[("adult", "training_pool")].columns if c.missing > 0}
    assert with_missing == {"workclass", "occupation", "native-country"}
    assert profiles[("abalone", "training_pool")].missing_cells == 0
    # the shifted copy changes values, never which cells are missing
    for column in ("workclass", "occupation", "native-country"):
        a = profiles[("adult", "monitoring_stream")].column(column)
        b = profiles[("adult", "monitoring_shifted")].column(column)
        assert a is not None and b is not None and a.missing == b.missing


def test_adult_target_class_counts_match_the_manifest(
    profiles: dict[tuple[str, str], DatasetProfile],
) -> None:
    counts = manifest()["datasets"]["adult"]["target_counts"]
    for role, name in (
        ("training_pool", "training_pool"),
        ("monitoring_stream", "monitoring_stream"),
    ):
        income = profiles[("adult", role)].column("income")
        assert income is not None and income.unique == 2
        assert dict(income.top_values) == counts[name]


def test_no_na_like_text_or_leading_zeros_in_the_demo_data(
    profiles: dict[tuple[str, str], DatasetProfile],
) -> None:
    for p in profiles.values():
        assert all(c.na_like_count == 0 and c.leading_zero_count == 0 for c in p.columns)


# --- statistics and duplicates against independent standard-library calculations -----------


def test_target_statistics_match_a_standard_library_calculation(
    profiles: dict[tuple[str, str], DatasetProfile],
) -> None:
    rings = [float(r["Rings"]) for r in csv_rows(demo_path("abalone", "training_pool"))]
    c = profiles[("abalone", "training_pool")].column("Rings")
    assert c is not None and c.numeric is not None
    q = statistics.quantiles(rings, n=4, method="inclusive")
    assert (c.numeric.minimum, c.numeric.maximum) == (min(rings), max(rings))
    assert c.numeric.mean == pytest.approx(statistics.mean(rings), rel=1e-11)
    assert c.numeric.std == pytest.approx(statistics.stdev(rings), rel=1e-11)
    assert c.numeric.median == pytest.approx(statistics.median(rings), rel=1e-11)
    assert c.numeric.p25 == pytest.approx(q[0])
    assert c.numeric.p75 == pytest.approx(q[2])


def test_duplicate_row_count_matches_a_standard_library_count(
    profiles: dict[tuple[str, str], DatasetProfile],
) -> None:
    for key in TARGETS:
        rows = [tuple(r.values()) for r in csv_rows(demo_path(key, "training_pool"))]
        extra = sum(n - 1 for n in Counter(rows).values())
        assert profiles[(key, "training_pool")].duplicate_rows == extra
    assert profiles[("adult", "training_pool")].duplicate_rows > 0  # known from Step 2


# --- the shifted copy is visible in the profile (not drift detection) ----------------------


def test_shifted_copy_profile_shows_the_documented_shift(
    profiles: dict[tuple[str, str], DatasetProfile],
) -> None:
    shift = {
        s["column"]: s["added_amount"] for s in manifest()["datasets"]["adult"]["shift"]["numeric"]
    }
    for column, amount in shift.items():
        a = profiles[("adult", "monitoring_stream")].column(column)
        b = profiles[("adult", "monitoring_shifted")].column(column)
        assert a is not None and b is not None and a.numeric is not None and b.numeric is not None
        assert b.numeric.mean - a.numeric.mean == pytest.approx(amount, abs=1e-6)
    untouched_a = profiles[("adult", "monitoring_stream")].column("fnlwgt")
    untouched_b = profiles[("adult", "monitoring_shifted")].column("fnlwgt")
    assert untouched_a == untouched_b


# --- validation on the demo files ----------------------------------------------------------


@pytest.mark.parametrize("key", sorted(TARGETS))
@pytest.mark.parametrize("role", ROLES)
def test_demo_files_have_no_errors_and_no_warnings(
    profiles: dict[tuple[str, str], DatasetProfile], key: str, role: str
) -> None:
    report = validate(profiles[(key, role)], target=TARGETS[key])
    assert report.ok and report.warnings == 0
    assert {f.code for f in report.findings} <= {"duplicate_rows"}  # the only info finding


# --- hand-broken copies (built in memory; the demo files are never modified) ---------------


def broken_frame(key: str) -> pd.DataFrame:
    return read_text_frame(demo_path(key, "training_pool").read_bytes())


def codes_of(p: DatasetProfile, target: str) -> set[tuple[str, str | None]]:
    return {(f.code, f.column) for f in validate(p, target=target).findings}


def test_blanked_target_values_are_an_error() -> None:
    frame = broken_frame("adult")
    frame.loc[frame.index[::20], "income"] = None
    report = validate(profile_frame(frame), target="income")
    f = next(x for x in report.findings if x.code == "target_missing")
    assert (
        f.severity == "error" and f.details["missing"] == len(frame.index[::20]) and not report.ok
    )


def test_added_constant_and_id_columns_are_warned_about() -> None:
    frame = broken_frame("adult")
    frame["constant"] = "x"
    frame["row_id"] = [str(i) for i in range(len(frame))]
    found = codes_of(profile_frame(frame), "income")
    assert ("constant_column", "constant") in found and ("id_like_integer", "row_id") in found


def test_na_text_injected_into_a_number_column() -> None:
    frame = broken_frame("adult")
    frame.loc[frame.index[:30], "age"] = "N/A"
    p = profile_frame(frame)
    found = codes_of(p, "income")
    age = p.column("age")
    assert age is not None and age.kind == "mixed" and age.na_like_count == 30
    assert ("na_like_values", "age") in found and ("mixed_types", "age") in found


def test_zero_padded_codes_are_flagged_and_kept_as_text() -> None:
    frame = broken_frame("adult")
    frame["education-num"] = frame["education-num"].str.zfill(3)  # "013": a code, not the number 13
    p = profile_frame(frame)
    c = p.column("education-num")
    assert c is not None and c.kind == "code" and c.numeric is None and c.leading_zero_count > 0
    assert ("leading_zero_values", "education-num") in codes_of(p, "income")
    assert "013" in dict(c.top_values) or "016" in dict(c.top_values)  # original text kept


def test_a_cut_down_file_has_too_few_rows() -> None:
    p = profile_frame(broken_frame("abalone").head(10))
    assert ("too_few_rows", None) in codes_of(p, "Rings")
    assert not validate(p, target="Rings").ok


def test_wrong_target_name_is_an_error() -> None:
    assert ("unknown_target", "Income") in codes_of(
        profile_of_file("adult", "training_pool"), "Income"
    )


# --- stored versions, JSON determinism, files untouched ------------------------------------


def test_profile_through_the_store_equals_profile_of_the_file(tmp_path: Path) -> None:
    store = FileDatasetStore(tmp_path / "store")
    for key in TARGETS:
        path = demo_path(key, "training_pool")
        store.ingest(f"{key}-pool", path.read_bytes())
        vp = profile_version(store, f"{key}-pool@v1")
        assert vp.profile == profile_of_file(key, "training_pool")
        assert (
            vp.to_dict()["sha256"]
            == manifest()["datasets"][key]["files"]["training_pool"]["sha256"]
        )


def test_json_output_is_identical_across_runs_and_strictly_valid() -> None:
    first = json.dumps(
        profile_of_file("adult", "training_pool").to_dict(), sort_keys=True, allow_nan=False
    )
    second = json.dumps(
        profile_of_file("adult", "training_pool").to_dict(), sort_keys=True, allow_nan=False
    )
    assert first == second and json.loads(first)["rows"] == 34190
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()


def test_profiling_and_validation_do_not_modify_the_demo_files() -> None:
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in DEMO.glob("*.csv")}
    for key in TARGETS:
        for role in ROLES:
            validate(profile_of_file(key, role), target=TARGETS[key])
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in DEMO.glob("*.csv")}
    assert before == after
    for key in TARGETS:
        for role in ROLES:
            meta = manifest()["datasets"][key]["files"][role]
            assert after[demo_path(key, role)] == meta["sha256"]
