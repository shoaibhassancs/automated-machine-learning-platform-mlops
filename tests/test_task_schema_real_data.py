"""Task suggestion, exploration and schema checks on the prepared Adult and Abalone demo files
(skipped until `make demo-data`). Expected numbers are recomputed here with the standard library
(`csv`, `statistics`) and NumPy, independently of the modules under test. The demo files are only
read, never modified."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from automl_ops.explore import explore_frame
from automl_ops.profile import profile_frame, read_text_frame
from automl_ops.schema import build_schema, check_csv_bytes, profile_for_schema
from automl_ops.task import TaskConfig, TaskType, check_task_config, class_counts, suggest_task

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"

pytestmark = pytest.mark.skipif(
    not (DEMO / "manifest.json").exists(),
    reason="demo data not prepared; run `make demo-data` first",
)


def read_rows(name: str) -> list[dict[str, str]]:
    with (DEMO / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def distinct(name: str, column: str) -> int:
    return len({row[column] for row in read_rows(name) if row[column] != ""})


def test_suggestions_match_independently_counted_targets() -> None:
    adult = read_rows("adult_train_pool.csv")
    abalone = read_rows("abalone_train_pool.csv")
    assert distinct("adult_train_pool.csv", "income") == 2
    rings = distinct("abalone_train_pool.csv", "Rings")
    assert rings > 20  # counted here, not recalled: more than the whole-number cut-off of 20

    adult_frame = read_text_frame((DEMO / "adult_train_pool.csv").read_bytes())
    abalone_frame = read_text_frame((DEMO / "abalone_train_pool.csv").read_bytes())
    s_adult = suggest_task(profile_frame(adult_frame), "income")
    s_abalone = suggest_task(profile_frame(abalone_frame), "Rings")
    assert s_adult is not None and s_adult.task == "classification"
    assert s_abalone is not None and s_abalone.task == "regression"
    assert len(adult) > 30000 and len(abalone) > 2000


@pytest.mark.parametrize(
    ("file", "target", "task"),
    [
        ("adult_train_pool.csv", "income", "classification"),
        ("abalone_train_pool.csv", "Rings", "regression"),
    ],
)
def test_demo_configurations_have_no_errors_or_warnings(
    file: str, target: str, task: TaskType
) -> None:
    frame = read_text_frame((DEMO / file).read_bytes())
    profile = profile_frame(frame)
    config = TaskConfig(target, task)
    counts = class_counts(frame, target) if task == "classification" else {}
    report = check_task_config(profile, counts, config)
    assert report.ok and report.warnings == 0


def test_class_counts_match_a_counter_over_the_raw_file() -> None:
    frame = read_text_frame((DEMO / "adult_train_pool.csv").read_bytes())
    expected = Counter(row["income"] for row in read_rows("adult_train_pool.csv"))
    assert class_counts(frame, "income") == dict(expected)


def test_abalone_exploration_matches_numpy_on_the_raw_values() -> None:
    rows = read_rows("abalone_train_pool.csv")
    frame = read_text_frame((DEMO / "abalone_train_pool.csv").read_bytes())
    result = explore_frame(frame, profile_frame(frame), TaskConfig("Rings", "regression"))
    length = np.array([float(r["Length"]) for r in rows])
    diameter = np.array([float(r["Diameter"]) for r in rows])
    counts, _ = np.histogram(length, bins=20)
    h = next(h for h in result["histograms"] if h["column"] == "Length")
    assert h["counts"] == counts.tolist() and sum(h["counts"]) == len(rows)
    columns = result["correlation"]["columns"]
    r = result["correlation"]["matrix"][columns.index("Length")][columns.index("Diameter")]
    assert r == pytest.approx(float(np.corrcoef(length, diameter)[0, 1]), abs=1e-10)
    assert "Sex" not in columns and result["class_balance"] is None


@pytest.mark.parametrize(
    ("prefix", "target", "shifted_numeric"),
    [
        ("adult", "income", {"age", "hours-per-week"}),
        ("abalone", "Rings", {"Whole_weight", "Shell_weight"}),
    ],
)
def test_pool_stream_and_shifted_copy_against_the_pool_schema(
    prefix: str, target: str, shifted_numeric: set[str]
) -> None:
    pool_bytes = (DEMO / f"{prefix}_train_pool.csv").read_bytes()
    frame = read_text_frame(pool_bytes)
    task: TaskType = "classification" if prefix == "adult" else "regression"
    spec = build_schema(profile_for_schema(frame), TaskConfig(target, task))

    own = check_csv_bytes(spec, pool_bytes)
    assert own.ok and own.observations == ()  # the training data fits its own schema exactly

    stream = check_csv_bytes(spec, (DEMO / f"{prefix}_monitor_stream.csv").read_bytes())
    assert stream.ok  # the stream is well-formed; any observations are information only

    shifted = check_csv_bytes(spec, (DEMO / f"{prefix}_monitor_shifted.csv").read_bytes())
    assert shifted.ok and shifted.failures == ()  # still passes structurally
    flagged = {f.column for f in shifted.observations if f.code == "out_of_observed_range"}
    assert shifted_numeric <= flagged  # the shifted columns are reported
    assert all(
        f.code != "unseen_categories" for f in shifted.observations
    )  # shift reuses categories


def test_adult_age_observation_matches_an_independent_count() -> None:
    pool_ages = [int(r["age"]) for r in read_rows("adult_train_pool.csv")]
    shifted_ages = [int(r["age"]) for r in read_rows("adult_monitor_shifted.csv")]
    expected_above = sum(a > max(pool_ages) for a in shifted_ages)
    expected_below = sum(a < min(pool_ages) for a in shifted_ages)

    frame = read_text_frame((DEMO / "adult_train_pool.csv").read_bytes())
    spec = build_schema(profile_for_schema(frame), TaskConfig("income", "classification"))
    report = check_csv_bytes(spec, (DEMO / "adult_monitor_shifted.csv").read_bytes())
    (obs,) = [f for f in report.observations if f.column == "age"]
    assert (obs.details["below"], obs.details["above"]) == (expected_below, expected_above)
    assert expected_above > 0


def test_demo_files_are_not_modified_by_any_of_this() -> None:
    def digest() -> dict[str, str]:
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEMO.glob("*.csv"))
        }

    before = digest()
    frame = read_text_frame((DEMO / "adult_train_pool.csv").read_bytes())
    spec = build_schema(profile_for_schema(frame), TaskConfig("income", "classification"))
    check_csv_bytes(spec, (DEMO / "adult_monitor_shifted.csv").read_bytes())
    explore_frame(frame, profile_frame(frame))
    assert digest() == before
