"""Tests for demo data preparation, using small invented data (no downloads, no real datasets)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_ops import demo_data as dd

# --- helpers -------------------------------------------------------------------------------


def make_frame(n: int = 300, *, seed: int = 0, with_duplicates: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "score": rng.normal(10.0, 2.0, n).round(3),
            "kind": pd.array(rng.choice(["a", "b", "c"], n), dtype="string"),
            "y": rng.choice(["no", "yes"], n, p=[0.7, 0.3]),
        }
    )
    frame.loc[::17, "kind"] = pd.NA
    if with_duplicates:
        frame = pd.concat([frame, frame.iloc[:20], frame.iloc[:5]], ignore_index=True)
    return frame


TINY_SPEC = dd.DatasetSpec(
    key="tiny",
    name="Tiny invented data",
    task="classification",
    target="y",
    page_url="https://example.invalid/tiny",
    zip_url="https://example.invalid/tiny.zip",
    zip_name="tiny.zip",
    licence_name="none (invented)",
    licence_url="https://example.invalid/licence",
    licence_statement="invented for tests",
    citation="invented",
    doi="none",
    documented_rows=300,
    numeric_shifts=(dd.NumericShift("age", 1.0), dd.NumericShift("score", 0.5)),
    category_shifts=(dd.CategoryShift("kind", "c", 0.5),),
    stratify=True,
)

# --- Adult parsing: the two original files use different label formats --------------------

ADULT_TRAIN = (
    "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, "
    "2174, 0, 40, United-States, <=50K\n"
    "38, ?, 215646, HS-grad, 9, Divorced, ?, Not-in-family, White, Male, 0, 0, 40, ?, >50K\n"
    "\n"
)
ADULT_TEST = (
    "|1x3 Cross validator\n"
    "25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, Own-child, Black, Male, "
    "0, 0, 40, United-States, <=50K.\n"
    "52, Self-emp-inc, 287927, HS-grad, 9, Married-civ-spouse, Exec-managerial, Wife, White, "
    "Female, 15024, 0, 40, United-States, >50K.\n"
    "\n"
)


def test_adult_files_merge_with_normalised_labels_and_missing_values() -> None:
    merged, report = dd.prepare_adult(ADULT_TRAIN.encode(), ADULT_TEST.encode())
    assert len(merged) == 4
    assert report["train_label_values_before"] == ["<=50K", ">50K"]
    assert report["test_label_values_before"] == ["<=50K.", ">50K."]  # the trailing period
    assert sorted(merged["income"].unique()) == ["<=50K", ">50K"]
    assert report["missing_by_column"] == {"workclass": 1, "occupation": 1, "native-country": 1}
    assert not (merged.astype(str) == "?").to_numpy().any()
    assert merged["age"].dtype == "int64"


def test_adult_unexpected_label_is_rejected() -> None:
    bad = ADULT_TRAIN.replace("<=50K", "maybe")
    with pytest.raises(ValueError, match="unexpected Adult label"):
        dd.prepare_adult(bad.encode(), ADULT_TEST.encode())


def test_adult_test_file_must_start_with_comment_line() -> None:
    with pytest.raises(ValueError, match="comment line"):
        dd.parse_adult_file(ADULT_TRAIN.encode(), skip_header_line=True)


def test_abalone_parses_without_header() -> None:
    raw = (
        b"M,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15\n"
        b"I,0.33,0.255,0.08,0.205,0.0895,0.0395,0.055,7\n\n"
    )
    frame, report = dd.prepare_abalone(raw)
    assert list(frame.columns) == dd.ABALONE_COLUMNS
    assert len(frame) == 2
    assert report["sex_values"] == ["I", "M"]


# --- duplicates ----------------------------------------------------------------------------


def test_duplicate_report_counts_without_removing() -> None:
    frame = make_frame(with_duplicates=True)
    before = len(frame)
    report = dd.duplicate_report(frame)
    assert len(frame) == before
    assert report["rows"] == before
    assert report["extra_copies"] == 25
    assert report["duplicate_groups"] >= 1


# --- partition -----------------------------------------------------------------------------


def test_partition_is_disjoint_complete_and_reproducible() -> None:
    frame = make_frame()
    pool, stream = dd.partition_rows(frame, "y", seed=7, stream_fraction=0.3, stratify=True)
    assert set(pool).isdisjoint(set(stream))
    assert len(pool) + len(stream) == len(frame)
    pool2, stream2 = dd.partition_rows(frame, "y", seed=7, stream_fraction=0.3, stratify=True)
    assert np.array_equal(pool, pool2) and np.array_equal(stream, stream2)
    _, stream3 = dd.partition_rows(frame, "y", seed=8, stream_fraction=0.3, stratify=True)
    assert not np.array_equal(stream, stream3)


def test_partition_keeps_classes_balanced_and_size_close_to_fraction() -> None:
    frame = make_frame(1000)
    pool, stream = dd.partition_rows(frame, "y", seed=1, stream_fraction=0.3, stratify=True)
    assert abs(len(stream) / len(frame) - 0.3) < 0.01
    share_pool = (frame.iloc[pool]["y"] == "yes").mean()
    share_stream = (frame.iloc[stream]["y"] == "yes").mean()
    assert abs(share_pool - share_stream) < 0.01


def test_partition_without_stratify_still_splits() -> None:
    frame = make_frame()
    pool, stream = dd.partition_rows(frame, "y", seed=3, stream_fraction=0.25, stratify=False)
    assert set(pool).isdisjoint(set(stream))
    assert abs(len(stream) / len(frame) - 0.25) < 0.02


def test_partition_keeps_exact_duplicates_on_the_same_side() -> None:
    frame = make_frame(with_duplicates=True)
    pool, stream = dd.partition_rows(frame, "y", seed=5, stream_fraction=0.3, stratify=True)
    assert len(pool) + len(stream) == len(frame)  # nothing removed
    pool_rows = {tuple(r) for r in frame.iloc[pool].astype(str).itertuples(index=False)}
    stream_rows = {tuple(r) for r in frame.iloc[stream].astype(str).itertuples(index=False)}
    assert pool_rows.isdisjoint(stream_rows)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_partition_rejects_bad_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="stream_fraction"):
        dd.partition_rows(make_frame(), "y", seed=0, stream_fraction=fraction, stratify=False)


# --- shifted copy --------------------------------------------------------------------------


def test_shifted_copy_changes_only_documented_columns_and_keeps_labels() -> None:
    stream = make_frame(400, seed=2)
    original = stream.copy()
    shifted, applied = dd.make_shifted_copy(
        stream, TINY_SPEC.numeric_shifts, TINY_SPEC.category_shifts, seed=9
    )
    pd.testing.assert_frame_equal(stream, original)  # the input stream is not modified
    assert list(shifted.columns) == list(stream.columns)
    assert len(shifted) == len(stream)
    assert (shifted["y"] == stream["y"]).all()
    # Whole-number column: exact whole-number shift
    age_amount = applied["numeric"][0]["added_amount"]
    assert float(age_amount).is_integer()
    assert ((shifted["age"] - stream["age"]) == int(age_amount)).all()
    # Float column: shifted by the documented amount
    score_amount = applied["numeric"][1]["added_amount"]
    assert np.allclose(shifted["score"] - stream["score"], score_amount)
    # Categorical: documented number of rows changed, missing values stay missing
    changed = int((shifted["kind"].fillna("NA") != stream["kind"].fillna("NA")).sum())
    assert changed == applied["categorical"][0]["rows_changed"] > 0
    assert shifted["kind"].isna().equals(stream["kind"].isna())


def test_shifted_copy_is_reproducible() -> None:
    stream = make_frame(200, seed=4)
    a, _ = dd.make_shifted_copy(stream, TINY_SPEC.numeric_shifts, TINY_SPEC.category_shifts, seed=1)
    b, _ = dd.make_shifted_copy(stream, TINY_SPEC.numeric_shifts, TINY_SPEC.category_shifts, seed=1)
    pd.testing.assert_frame_equal(a, b)


# --- download and zip safety (no network used) --------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://archive.ics.uci.edu/static/public/2/adult.zip",  # not https
        "https://example.com/adult.zip",  # other host
        "ftp://archive.ics.uci.edu/adult.zip",
    ],
)
def test_download_refuses_other_schemes_and_hosts(url: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        dd.download(url, tmp_path / "x.zip")
    assert not (tmp_path / "x.zip").exists()


def test_official_urls_pass_the_url_check() -> None:
    for spec in dd.SPECS.values():
        dd.check_download_url(spec.zip_url)


def test_zip_reader_returns_requested_members_and_rejects_unsafe_names(tmp_path: Path) -> None:
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("data.csv", "a,b\n1,2\n")
    assert dd.read_zip_members(good, ["data.csv"]) == {"data.csv": b"a,b\n1,2\n"}
    with pytest.raises(ValueError, match="not found"):
        dd.read_zip_members(good, ["missing.csv"])

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../escape.csv", "x")
    with pytest.raises(ValueError, match="unsafe"):
        dd.read_zip_members(bad, ["../escape.csv"])


# --- writing files, the manifest, and the verifier itself --------------------------------


def write_tiny(tmp_path: Path) -> Path:
    prepared = dd.build_dataset(
        TINY_SPEC, make_frame(with_duplicates=True), seed=11, stream_fraction=0.3
    )
    demo = tmp_path / "demo"
    entry = dd.write_dataset(prepared, demo)
    entry["source"]["documented_rows"] = len(prepared.source)  # tiny data has no UCI count
    manifest = dd.build_manifest({"tiny": entry}, {}, seed=11, stream_fraction=0.3)
    dd.write_manifest(manifest, demo / "manifest.json")
    return demo


def test_verifier_passes_on_correct_files(tmp_path: Path) -> None:
    demo = write_tiny(tmp_path)
    results = dd.verify_demo_data(tmp_path, demo)
    assert results
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_manifest_records_seed_shift_and_counts(tmp_path: Path) -> None:
    import json

    demo = write_tiny(tmp_path)
    manifest = json.loads((demo / "manifest.json").read_text())
    entry = manifest["datasets"]["tiny"]
    assert manifest["split_seed"] == 11 and manifest["shift_seed"] == 12
    assert manifest["notice"].startswith("Demo data")
    assert entry["duplicates_in_source"]["extra_copies"] == 25
    assert entry["partition"]["pool_rows"] + entry["partition"]["stream_rows"] == 325
    assert entry["shift"]["numeric"][0]["column"] == "age"
    assert set(entry["files"]) == {"training_pool", "monitoring_stream", "monitoring_shifted"}


def test_verifier_catches_a_stream_row_leaked_into_the_pool(tmp_path: Path) -> None:
    demo = write_tiny(tmp_path)
    pool_path = demo / "tiny_train_pool.csv"
    stream_lines = (demo / "tiny_monitor_stream.csv").read_text().splitlines()
    with pool_path.open("a") as handle:
        handle.write(stream_lines[1] + "\n")  # copy one stream row into the pool
    names = {r.name: r for r in dd.verify_demo_data(tmp_path, demo)}
    assert not names["[tiny] no row content shared by pool and stream"].passed
    assert not names["[tiny] file hashes match manifest"].passed


def test_verifier_catches_a_changed_label_in_the_shifted_copy(tmp_path: Path) -> None:
    demo = write_tiny(tmp_path)
    path = demo / "tiny_monitor_shifted.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[0, "y"] = "yes" if frame.loc[0, "y"] == "no" else "no"
    frame.to_csv(path, index=False, lineterminator="\n")
    names = {r.name: r for r in dd.verify_demo_data(tmp_path, demo)}
    assert not names[
        "[tiny] shifted copy has the stream's row count and unchanged target column"
    ].passed


# --- real prepared data (skipped if `make demo-data` has not been run) --------------------

REAL_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(
    not (REAL_DATA / "demo" / "manifest.json").exists(),
    reason="demo data not prepared; run `make demo-data` first",
)
def test_real_prepared_demo_data_passes_all_checks() -> None:
    results = dd.verify_demo_data(REAL_DATA)
    assert len(results) >= 20
    assert all(r.passed for r in results), [r for r in results if not r.passed]
