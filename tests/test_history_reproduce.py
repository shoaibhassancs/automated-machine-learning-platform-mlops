"""Experiment history, comparison and reproducibility (BUILD_PLAN Steps 15-16).

One module-scoped MLflow store holds a few small jobs (invented data, three folds); tests that
change a record work on a copy of it. Nothing here needs the demo data.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from mlflow import MlflowClient

import automl_ops.finalize as finalize_module
import automl_ops.train as train_module
from automl_ops.compare import CompareError, compare_jobs
from automl_ops.package import library_versions
from automl_ops.reproduce import (
    ReproductionError,
    _Walk,
    derive_config,
    reproduce_job,
    sealed_test_result,
)
from automl_ops.task import TaskConfig
from automl_ops.tracking import (
    JobNotFoundError,
    LoggedJob,
    TrackingStore,
    describe_job,
    list_jobs,
    log_job,
    read_job,
    record_from_package,
    record_from_training,
)
from automl_ops.train import TrainSettings, run_training
from conftest import FAST, FAST_REG, Run

SRC = Path(__file__).resolve().parents[1] / "src" / "automl_ops"


def digest_tree(*roots: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


@dataclasses.dataclass(frozen=True)
class Jobs:
    store: TrackingStore
    a: str  # classification, finalized, seed 42
    b: str  # classification, cross-validation only, seed 7
    reg_final: str
    reg_train: str  # regression, cross-validation only, seed 7
    clf: Run
    reg: Run


def train_job(run: Run, requests: Any, settings: TrainSettings, store: TrackingStore) -> str:
    training = run_training(run.frame, run.config, requests, settings)
    record = record_from_training(training.to_dict(), run.dataset.to_dict(), library_versions())
    log_job(record, store)
    return record.job_key


@pytest.fixture(scope="module")
def jobs(clf_run: Run, reg_run: Run, tmp_path_factory: pytest.TempPathFactory) -> Jobs:
    store = TrackingStore(tmp_path_factory.mktemp("history") / "mlflow")
    a = clf_run.result.package_id
    log_job(record_from_package(clf_run.root, a), store)
    log_job(record_from_package(reg_run.root, reg_run.result.package_id), store)
    b = train_job(clf_run, FAST, TrainSettings(folds=3, seed=7), store)
    reg_train = train_job(reg_run, FAST_REG, TrainSettings(folds=3, seed=7), store)
    return Jobs(store, a, b, reg_run.result.package_id, reg_train, clf_run, reg_run)


@pytest.fixture
def copy_store(jobs: Jobs, tmp_path: Path) -> TrackingStore:
    """A private copy of the module store that a test may tamper with."""
    target = tmp_path / "mlflow-copy"
    shutil.copytree(jobs.store.root, target)
    # artifact locations are stored as absolute URIs, so the copy reads the *original* artifacts;
    # point the copy's experiment at its own folder
    import sqlite3

    db = sqlite3.connect(target / "mlflow.db")
    old = jobs.store.artifact_path.as_uri()
    new = (target.resolve() / "artifacts").as_uri()
    db.execute(
        "update experiments set artifact_location = replace(artifact_location, ?, ?)", (old, new)
    )
    db.execute("update runs set artifact_uri = replace(artifact_uri, ?, ?)", (old, new))
    db.commit()
    db.close()
    return TrackingStore(target)


def client_of(store: TrackingStore) -> MlflowClient:
    return MlflowClient(tracking_uri=store.tracking_uri)


def parent_run_id(store: TrackingStore, job_key: str) -> str:
    runs = client_of(store).search_runs(
        ["1"], f"tags.`platform.job_key` = '{job_key}' and tags.`platform.role` = 'job'"
    )
    return str(runs[0].info.run_id)


def child_run_id(store: TrackingStore, job_key: str, algorithm: str) -> str:
    runs = client_of(store).search_runs(
        ["1"],
        f"tags.`platform.job_key` = '{job_key}' and tags.`platform.algorithm` = '{algorithm}'",
    )
    return str(runs[0].info.run_id)


def reproduce(jobs: Jobs, job: LoggedJob, run: Run, **extra: Any) -> Any:
    return reproduce_job(job, run.frame, run.dataset.sha256, library_versions(), **extra)


# --- Step 15: history ----------------------------------------------------------------------


def test_history_lists_every_job_newest_first_with_the_logged_facts(jobs: Jobs) -> None:
    listed = list_jobs(jobs.store)
    assert {j.job_key for j in listed} == {jobs.a, jobs.b, jobs.reg_final, jobs.reg_train}
    assert [j.start_ms for j in listed] == sorted((j.start_ms for j in listed), reverse=True)
    by_key = {j.job_key: j for j in listed}
    final, cv_only = by_key[jobs.a], by_key[jobs.b]
    assert final.kind == "final" and final.state == "complete" and final.has_test_result
    assert final.selected_algorithm == jobs.clf.frozen.data["algorithm"]
    assert final.package_id == jobs.a and final.task == "classification"
    assert cv_only.kind == "train" and not cv_only.has_test_result
    assert cv_only.selected_algorithm is None and cv_only.seed == "7"
    assert final.algorithms_ok == 2 and final.algorithms_failed == 0


def test_history_filters(jobs: Jobs) -> None:
    def keys(**kw: Any) -> set[str]:
        return {j.job_key for j in list_jobs(jobs.store, **kw)}

    assert keys(kind="final") == {jobs.a, jobs.reg_final}
    assert keys(task="regression") == {jobs.reg_final, jobs.reg_train}
    assert keys(dataset=jobs.clf.dataset.sha256[:10]) == keys(task="classification")
    assert keys(algorithm=jobs.clf.frozen.data["algorithm"]) == {jobs.a}
    assert keys(state="incomplete") == set()
    assert keys(kind="train", task="classification") == {jobs.b}
    assert len(list_jobs(jobs.store, limit=1)) == 1


def test_history_of_a_missing_store_is_empty_and_creates_nothing(tmp_path: Path) -> None:
    store = TrackingStore(tmp_path / "nothing-here")
    assert list_jobs(store) == []
    with pytest.raises(JobNotFoundError):
        read_job("m-x", store)
    assert not (tmp_path / "nothing-here").exists()


def test_job_detail_equals_what_mlflow_and_the_sealed_package_say(jobs: Jobs) -> None:
    job = read_job(jobs.a, jobs.store)
    raw = describe_job(jobs.a, jobs.store)
    assert raw is not None
    assert job.tags == raw["parent"]["tags"] and job.params == raw["parent"]["params"]
    assert job.metrics == raw["parent"]["metrics"]
    for name, algo in job.algorithms.items():
        child = raw["children"][name]
        assert algo.tags == child["tags"] and algo.params == child["params"]
        assert algo.metrics == child["metrics"] and algo.run_status == child["status"]
    # fold series: one value per fold, in fold order
    series = next(iter(job.algorithms.values())).fold_metrics
    assert all(len(v) == jobs.clf.training.folds_used for v in series.values())
    # the package's own files agree with the logged copies
    manifest = json.loads((jobs.clf.result.path / "manifest.json").read_text())
    assert job.tags["platform.selection_sha256"] == manifest["selection_sha256"]
    assert job.training == manifest["training"] and job.selection == manifest["selection"]
    sealed = json.loads((jobs.clf.result.path / "test_result.json").read_text())
    (logged_test,) = [a.test_result for a in job.algorithms.values() if a.test_result]
    assert logged_test == sealed
    # a CV-only job carries no test result anywhere
    cv_only = read_job(jobs.b, jobs.store)
    assert all(a.test_result is None for a in cv_only.algorithms.values())


def test_unknown_and_incomplete_jobs_are_reported(jobs: Jobs, copy_store: TrackingStore) -> None:
    with pytest.raises(JobNotFoundError, match="no tracked job"):
        read_job("m-nope", jobs.store)
    client_of(copy_store).set_tag(
        parent_run_id(copy_store, jobs.b), "platform.logging", "incomplete"
    )
    assert [j.state for j in list_jobs(copy_store, state="incomplete")] == ["incomplete"]
    job = read_job(jobs.b, copy_store)
    with pytest.raises(ReproductionError, match="not completely tracked"):
        reproduce(jobs, job, jobs.clf)


# --- Step 15: comparison (cross-validation only) -------------------------------------------


def test_compare_ranks_by_cv_in_the_metric_direction_and_reports_differences(jobs: Jobs) -> None:
    result = compare_jobs([read_job(jobs.a, jobs.store), read_job(jobs.b, jobs.store)])
    assert result["metric"] == "f1_macro" and result["direction"] == "higher"
    scores = [r["cv_mean"] for r in result["rows"]]
    assert scores == sorted(scores, reverse=True) and [r["rank"] for r in result["rows"]] == [
        1, 2, 3, 4
    ]  # fmt: skip
    assert (
        "seed differs" in result["split_note"] and "not directly comparable" in result["split_note"]
    )
    assert "dataset_sha256" not in " ".join(result["split_differences"])  # same dataset
    # each row equals what MLflow holds for that run
    raw = describe_job(jobs.a, jobs.store)
    assert raw is not None
    for row in (r for r in result["rows"] if r["job_key"] == jobs.a):
        assert row["cv_mean"] == raw["children"][row["algorithm"]]["metrics"]["cv_f1_macro_mean"]

    reg = compare_jobs(
        [read_job(jobs.reg_final, jobs.store), read_job(jobs.reg_train, jobs.store)], "mae"
    )
    assert reg["direction"] == "lower"
    maes = [r["cv_mean"] for r in reg["rows"]]
    assert maes == sorted(maes)


def test_compare_of_identical_configurations_has_no_split_note(jobs: Jobs, tmp_path: Path) -> None:
    store = TrackingStore(tmp_path / "m")
    twin = train_job(
        jobs.clf, FAST, TrainSettings(folds=3), store
    )  # same settings as the final job
    log_job(record_from_package(jobs.clf.root, jobs.a), store)
    result = compare_jobs([read_job(twin, store), read_job(jobs.a, store)])
    assert result["split_note"] is None and result["split_differences"] == []
    by_algorithm: dict[str, set[float]] = {}
    for row in result["rows"]:
        by_algorithm.setdefault(row["algorithm"], set()).add(row["cv_mean"])
    assert all(len(values) == 1 for values in by_algorithm.values())  # identical CV numbers


def test_compare_refuses_what_it_cannot_compare(jobs: Jobs) -> None:
    clf, reg = read_job(jobs.a, jobs.store), read_job(jobs.reg_final, jobs.store)
    with pytest.raises(CompareError, match="different task or target"):
        compare_jobs([clf, reg])
    with pytest.raises(CompareError, match="at least two"):
        compare_jobs([clf])
    with pytest.raises(CompareError, match="twice"):
        compare_jobs([clf, clf])
    with pytest.raises(CompareError, match="not a classification metric"):
        compare_jobs([clf, read_job(jobs.b, jobs.store)], "rmse")


def test_compare_never_uses_or_shows_test_results(jobs: Jobs, copy_store: TrackingStore) -> None:
    keys = [jobs.a, jobs.b]
    before = compare_jobs([read_job(k, jobs.store) for k in keys])
    text = json.dumps(before)
    test_hash = jobs.clf.training.split.fingerprints()["test_sha256"]
    assert test_hash not in text and "test_rows_sha256" not in text
    assert not any(
        k.startswith("test_") and k != "test_fraction" for r in before["rows"] for k in r
    )
    # rewriting every logged test metric (and the artifact) changes nothing in the comparison
    client = client_of(copy_store)
    run_id = child_run_id(copy_store, jobs.a, jobs.clf.frozen.data["algorithm"])
    for name in client.get_run(run_id).data.metrics:
        if name.startswith("test_"):
            client.log_metric(run_id, name, 123.0, timestamp=2**42)
    tampered = read_job(jobs.a, copy_store).algorithms[jobs.clf.frozen.data["algorithm"]]
    assert all(v == 123.0 for k, v in tampered.metrics.items() if k.startswith("test_"))
    after = compare_jobs([read_job(k, copy_store) for k in keys])
    assert after == before


# --- Step 16: reproduction -----------------------------------------------------------------


def test_a_finalized_job_is_reproduced_from_its_record_alone(jobs: Jobs) -> None:
    job = read_job(jobs.a, jobs.store)
    report = reproduce(
        jobs,
        job,
        jobs.clf,
        package_test_result=sealed_test_result(jobs.clf.root, jobs.a),
        package_selection_sha256=job.tags["platform.selection_sha256"],
        package_checked=True,
    )
    assert report.ok and report.verdict == "REPRODUCED", report.to_dict()
    assert report.bit_identical and report.max_deviation == 0 and report.environment_same
    names = [i.name for i in report.items]
    assert "selection (from cross-validation only)" in names
    assert "recorded test result vs sealed package" in names
    assert all(i.status == "pass" for i in report.items)
    assert json.dumps(report.to_dict(), allow_nan=False)


def test_a_cv_only_job_is_reproduced_and_has_no_final_checks(jobs: Jobs) -> None:
    report = reproduce(jobs, read_job(jobs.b, jobs.store), jobs.clf)
    assert report.ok and report.bit_identical
    assert not any("test" in i.name for i in report.items)


def test_regression_jobs_reproduce_too(jobs: Jobs) -> None:
    for key in (jobs.reg_final, jobs.reg_train):
        assert reproduce(jobs, read_job(key, jobs.store), jobs.reg).ok


def test_excluded_columns_are_derived_from_the_logged_plan(tmp_path: Path, clf_run: Run) -> None:
    config = TaskConfig("y", "classification", excluded=("z",))
    store = TrackingStore(tmp_path / "m")
    training = run_training(clf_run.frame, config, FAST, TrainSettings(folds=3))
    record = record_from_training(training.to_dict(), clf_run.dataset.to_dict(), library_versions())
    log_job(record, store)
    job = read_job(record.job_key, store)
    derived = derive_config(job, list(clf_run.frame.columns))
    assert derived.excluded == ("z",) and derived.target == "y"
    assert reproduce_job(job, clf_run.frame, clf_run.dataset.sha256, library_versions()).ok
    # A record whose plan was altered yields another (self-consistent) configuration; the plan
    # alone cannot expose that, but the recomputed numbers do.
    bad = copy.deepcopy(job.training)
    bad["columns"]["numeric"] = []
    report = reproduce_job(
        dataclasses.replace(job, training=bad),
        clf_run.frame,
        clf_run.dataset.sha256,
        library_versions(),
    )
    assert not report.ok and any(
        i.status == "fail" and i.name.startswith("algorithm") for i in report.items
    )


def test_a_failed_algorithm_is_reproduced_as_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clf_run: Run
) -> None:
    original = train_module._make_estimator

    def make(spec: Any, params: Any, seed: int) -> Any:
        if spec.key == "random_forest":
            raise RuntimeError("deliberate failure")
        return original(spec, params, seed)

    monkeypatch.setattr(train_module, "_make_estimator", make)
    store = TrackingStore(tmp_path / "m")
    key = train_job(clf_run, FAST, TrainSettings(folds=3), store)
    job = read_job(key, store)
    again = reproduce_job(job, clf_run.frame, clf_run.dataset.sha256, library_versions())
    assert again.ok and any("random_forest [failed (RuntimeError)]" in i.name for i in again.items)
    monkeypatch.setattr(train_module, "_make_estimator", original)  # the failure is gone: differs
    healed = reproduce_job(job, clf_run.frame, clf_run.dataset.sha256, library_versions())
    assert not healed.ok and "random_forest" in healed.verdict


def test_a_different_environment_is_reported_without_failing(jobs: Jobs) -> None:
    job = read_job(jobs.b, jobs.store)
    changed = {**library_versions(), "scikit-learn": "0.0.1"}
    report = reproduce_job(job, jobs.clf.frame, jobs.clf.dataset.sha256, changed)
    assert report.ok and not report.environment_same
    assert "different environment" in report.verdict
    assert any("scikit-learn" in line for line in report.environment_differences)


def test_changed_settings_are_not_reproduced_and_name_the_difference(jobs: Jobs) -> None:
    job = read_job(jobs.b, jobs.store)
    training = copy.deepcopy(job.training)
    # claim the job used another seed: the re-run cannot match the logged numbers
    training["settings"]["seed"] = 8
    report = reproduce(jobs, dataclasses.replace(job, training=training), jobs.clf)
    assert not report.ok and report.verdict.startswith("NOT REPRODUCED")
    failing = [i for i in report.items if i.status == "fail"]
    assert any("summary" in m or "fold_metrics" in m for i in failing for m in i.mismatches)


def test_different_data_under_the_same_hash_claim_is_not_reproduced(jobs: Jobs) -> None:
    job = read_job(jobs.b, jobs.store)
    frame = jobs.clf.frame.copy()
    frame.loc[:20, "y"] = frame.loc[:20, "y"].map({"pos": "neg", "neg": "pos"})
    report = reproduce_job(job, frame, jobs.clf.dataset.sha256, library_versions())
    assert not report.ok


def test_a_different_dataset_hash_cannot_be_reproduced(jobs: Jobs) -> None:
    with pytest.raises(ReproductionError, match="different data"):
        reproduce_job(read_job(jobs.b, jobs.store), jobs.clf.frame, "0" * 64, library_versions())


# --- tolerance ---------------------------------------------------------------------------


def test_tolerance_edges() -> None:
    inside = _Walk(rel=1e-6)
    inside.number("m", 1.0, 1.0 + 1e-9 + 1e-6 - 1e-12)
    assert not inside.mismatches and not inside.identical
    outside = _Walk(rel=1e-6)
    outside.number("m", 1.0, 1.0 + 1e-9 + 1e-6 + 1e-9)
    assert outside.mismatches
    exact = _Walk(rel=0.0)
    exact.number("m", 0.5, 0.5)
    assert not exact.mismatches and exact.identical
    tight = _Walk(rel=0.0)
    tight.number("m", 0.5, 0.5 + 1e-8)
    assert tight.mismatches
    structure = _Walk(rel=1e-6)
    structure.walk("x", {"a": [1, 2], "b": None, "c": "t"}, {"a": [1, 2, 3], "b": 0, "c": "u"})
    assert len(structure.mismatches) == 3
    structure = _Walk(rel=1e-6)
    structure.walk("x", {"t": True}, {"t": 1})  # a bool is not a number
    assert structure.mismatches


def test_a_looser_tolerance_accepts_what_a_tight_one_rejects(jobs: Jobs) -> None:
    job = read_job(jobs.b, jobs.store)
    training = copy.deepcopy(job.training)
    algo = training["results"][0]
    metric = next(iter(algo["summary"]))
    algo["summary"][metric]["mean"] *= 1 + 1e-4
    tampered = dataclasses.replace(job, training=training)
    # the MLflow metric no longer equals training.json either, so loosen both through the option
    assert not reproduce(jobs, tampered, jobs.clf).ok
    assert reproduce(jobs, tampered, jobs.clf, rel_tolerance=1e-3).ok


# --- tampering with the record -------------------------------------------------------------


def test_a_changed_mlflow_metric_is_named(jobs: Jobs, copy_store: TrackingStore) -> None:
    algorithm = jobs.clf.frozen.data["algorithm"]
    run_id = child_run_id(copy_store, jobs.b, algorithm)
    client_of(copy_store).log_metric(run_id, "cv_f1_macro_mean", 0.01, timestamp=2**42)
    report = reproduce(jobs, read_job(jobs.b, copy_store), jobs.clf)
    assert not report.ok
    (bad,) = [i for i in report.items if i.status == "fail"]
    assert bad.name == "record consistency"
    assert any(f"{algorithm}.cv_f1_macro_mean" in m for m in bad.mismatches)


def test_a_changed_dataset_hash_tag_cannot_be_reproduced(
    jobs: Jobs, copy_store: TrackingStore
) -> None:
    client_of(copy_store).set_tag(
        parent_run_id(copy_store, jobs.b), "platform.dataset_sha256", "f" * 64
    )
    with pytest.raises(ReproductionError, match="different data"):
        reproduce(jobs, read_job(jobs.b, copy_store), jobs.clf)


def test_a_changed_logged_test_result_is_caught_against_the_package(
    jobs: Jobs, copy_store: TrackingStore
) -> None:
    algorithm = jobs.clf.frozen.data["algorithm"]
    run = client_of(copy_store).get_run(child_run_id(copy_store, jobs.a, algorithm))
    path = Path(run.info.artifact_uri.removeprefix("file://")) / "test_result.json"
    assert path.is_file() and str(copy_store.root.resolve()) in str(path.resolve())
    data = json.loads(path.read_text())
    metric = next(iter(data["evaluation"]["metrics"]))
    data["evaluation"]["metrics"][metric] = 0.123456
    path.write_text(json.dumps(data))
    job = read_job(jobs.a, copy_store)
    report = reproduce(
        jobs,
        job,
        jobs.clf,
        package_test_result=sealed_test_result(jobs.clf.root, jobs.a),
        package_selection_sha256=job.tags["platform.selection_sha256"],
        package_checked=True,
    )
    assert not report.ok
    failed = {i.name for i in report.items if i.status == "fail"}
    assert "recorded test result vs sealed package" in failed


def test_a_missing_package_is_reported_as_not_checked_not_as_a_pass(jobs: Jobs) -> None:
    report = reproduce(jobs, read_job(jobs.a, jobs.store), jobs.clf, package_checked=False)
    skipped = [i for i in report.items if i.status == "skipped"]
    assert report.ok and len(skipped) == 1 and "NOT CHECKED" in skipped[0].detail


def test_a_tampered_package_is_refused_not_trusted(jobs: Jobs, tmp_path: Path) -> None:
    from conftest import writable_copy

    root = writable_copy(jobs.clf, tmp_path / "pkg")
    (root / jobs.a / "test_result.json").write_text("{}")
    with pytest.raises(ReproductionError, match="failed verification"):
        sealed_test_result(root, jobs.a)


# --- the sealed test set -------------------------------------------------------------------


def test_no_history_compare_or_reproduce_path_reads_the_test_part(
    jobs: Jobs, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the held-out test part must not be read")

    monkeypatch.setattr(finalize_module, "_read_test_part", boom)
    monkeypatch.setattr(finalize_module, "run_final_test", boom)
    import automl_ops.package as package_module

    monkeypatch.setattr(package_module, "load_package", boom)
    list_jobs(jobs.store)
    a, b = read_job(jobs.a, jobs.store), read_job(jobs.b, jobs.store)
    compare_jobs([a, b])
    assert reproduce(jobs, a, jobs.clf).ok
    assert reproduce(jobs, b, jobs.clf).ok


def imports_of(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("name", ["reproduce.py", "compare.py"])
def test_the_new_modules_cannot_reach_mlflow_models_or_the_test_part(name: str) -> None:
    path = SRC / name
    imported = imports_of(path)
    assert not {m for m in imported if m.split(".")[0] in {"mlflow", "joblib"}}
    assert "automl_ops.finalize" not in imported
    source = path.read_text()
    for forbidden in ("load_package", "_read_test_part", "joblib.load", "run_final_test", "open("):
        assert forbidden not in source


def test_reproduce_job_takes_no_test_positions_models_or_stores() -> None:
    names = set(inspect.signature(reproduce_job).parameters)
    assert names == {
        "job", "frame", "dataset_sha256", "current_libraries", "package_test_result",
        "package_selection_sha256", "package_checked", "rel_tolerance", "progress",
    }  # fmt: skip


def test_the_read_side_of_tracking_never_writes(jobs: Jobs) -> None:
    before = digest_tree(jobs.store.root, jobs.clf.root, jobs.reg.root)
    list_jobs(jobs.store)
    job = read_job(jobs.a, jobs.store)
    compare_jobs([job, read_job(jobs.b, jobs.store)])
    reproduce(
        jobs,
        job,
        jobs.clf,
        package_test_result=sealed_test_result(jobs.clf.root, jobs.a),
        package_checked=True,
    )
    assert digest_tree(jobs.store.root, jobs.clf.root, jobs.reg.root) == before


def test_the_read_side_leaves_no_new_run(jobs: Jobs) -> None:
    count = len(client_of(jobs.store).search_runs(["1"], max_results=1000))
    reproduce(jobs, read_job(jobs.b, jobs.store), jobs.clf)
    assert len(client_of(jobs.store).search_runs(["1"], max_results=1000)) == count
