"""P4-4: the shared runner and the job-tracking policy (no database needed)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from automl_ops import finalize as finalize_module
from automl_ops import runner
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.job_tracking import (
    JOBS_DIR_SETTING,
    TrackingNotAllowed,
    archive_root,
    job_tracking_store,
    tracking_problem,
)
from automl_ops.profile import read_version_frame
from automl_ops.runner import (
    JobCancelled,
    RunSpec,
    WorkerStopping,
    job_result,
    resolve_requests,
    run_pipeline,
    spec_from_config,
    spec_to_config,
)
from automl_ops.task import TaskConfig
from automl_ops.train import AlgorithmRequest, ProgressEvent, TrainSettings, run_training
from conftest import toy_bytes


def toy(tmp_path: Path) -> tuple[Any, Any]:
    store = FileDatasetStore(tmp_path / "store")
    store.ingest("toy", toy_bytes(), source_filename="toy.csv")
    return read_version_frame(store, "toy@v1")


def spec(kind: str = "train", **over: Any) -> RunSpec:
    base: dict[str, Any] = {
        "kind": kind,
        "config": TaskConfig("y", "classification"),
        "requests": (
            AlgorithmRequest("logistic_regression"),
            AlgorithmRequest("random_forest", {"n_estimators": 10}),
        ),
        "settings": TrainSettings(folds=3),
    }
    base.update(over)
    return RunSpec(**base)


def without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: without_timing(v) for k, v in value.items() if k not in ("seconds", "fold_seconds")
        }
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def test_a_cancel_raised_from_progress_is_not_swallowed_as_an_algorithm_failure(
    tmp_path: Path,
) -> None:
    """`run_training` catches `Exception` per algorithm; a stop signal must pass through it."""
    version, frame = toy(tmp_path)

    def cancel_on_first_fold(event: ProgressEvent) -> None:
        if event.kind == "fold_finished":
            raise JobCancelled()

    with pytest.raises(JobCancelled):
        run_pipeline(frame, version, spec(), progress=cancel_on_first_fold)

    def stop_early(event: ProgressEvent) -> None:
        raise WorkerStopping()

    with pytest.raises(WorkerStopping):
        run_pipeline(frame, version, spec(), progress=stop_early)
    assert issubclass(JobCancelled, BaseException) and not issubclass(JobCancelled, Exception)


def test_a_train_run_equals_run_training_and_never_touches_the_test_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version, frame = toy(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a train run must not read the held-out test part or finalize")

    monkeypatch.setattr(finalize_module, "_read_test_part", forbidden)
    monkeypatch.setattr(runner, "finalize", forbidden)
    outcome = run_pipeline(frame, version, spec(), select=True)
    direct = run_training(frame, spec().config, spec().requests, spec().settings)
    assert without_timing(outcome.training.to_dict()) == without_timing(direct.to_dict())
    assert outcome.selection is not None and outcome.final is None and outcome.frozen is None
    plain = run_pipeline(frame, version, spec())  # the CLI's `train`: no selection step
    assert plain.selection is None and plain.selection_note is None
    result = job_result(outcome)
    assert result["test_result"] is None and result["package_id"] is None
    assert result["dataset_ref"] == "toy@v1" and result["sha256"] == version.sha256


def test_a_final_run_reads_the_test_part_exactly_once_after_the_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version, frame = toy(tmp_path)
    calls: list[str] = []
    real = finalize_module._read_test_part

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append("test part read")
        return real(*args, **kwargs)

    monkeypatch.setattr(finalize_module, "_read_test_part", counting)
    outcome = run_pipeline(
        frame,
        version,
        spec("final"),
        tmp_path / "models",
        before_final=lambda: calls.append("hook"),
    )
    assert calls == ["hook", "test part read"]
    assert outcome.final is not None and outcome.frozen is not None
    assert job_result(outcome)["package_id"] == outcome.final.package_id


def test_a_hook_that_stops_the_run_keeps_the_test_part_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version, frame = toy(tmp_path)
    monkeypatch.setattr(
        finalize_module, "_read_test_part", lambda *a, **k: pytest.fail("test part was read")
    )

    def refuse() -> None:
        raise JobCancelled()

    with pytest.raises(JobCancelled):
        run_pipeline(frame, version, spec("final"), tmp_path / "models", before_final=refuse)
    assert not (tmp_path / "models").exists() or not list((tmp_path / "models").iterdir())


def test_job_config_round_trips_and_defaults_come_from_the_catalogue() -> None:
    original = spec("final", selection_metric="f1_macro")
    rebuilt = spec_from_config("final", spec_to_config(original))
    assert rebuilt == original
    task = TaskConfig("y", "classification")
    assert [r.key for r in resolve_requests(task.task, None)] == [
        r.key for r in resolve_requests(task.task, [])
    ]
    assert len(resolve_requests("regression", None)) >= 2
    assert [r.key for r in resolve_requests("classification", ["random_forest"])] == [
        "random_forest"
    ]
    with pytest.raises(ValueError):
        RunSpec("other", task, (), TrainSettings())


# -- job tracking policy (D6) --------------------------------------------------------------------


def test_sqlite_tracking_needs_its_own_directory(tmp_path: Path) -> None:
    with pytest.raises(TrackingNotAllowed, match=JOBS_DIR_SETTING):
        job_tracking_store({"AUTOML_MLFLOW_BACKEND": "sqlite"})
    assert JOBS_DIR_SETTING in (tracking_problem({}) or "")
    store = job_tracking_store({JOBS_DIR_SETTING: str(tmp_path / "jobs-mlflow")})
    assert store.backend == "sqlite" and store.root == (tmp_path / "jobs-mlflow").resolve()
    assert tracking_problem({JOBS_DIR_SETTING: str(tmp_path / "jobs-mlflow")}) is None


def test_the_original_archive_is_refused_however_it_is_named(tmp_path: Path) -> None:
    archive = archive_root()
    assert archive.name == "mlflow" and archive.parent.name == "data"
    candidates = [archive, archive / "artifacts", archive / "x" / "y", Path("data/mlflow")]
    for candidate in candidates:
        with pytest.raises(TrackingNotAllowed, match="archive"):
            job_tracking_store({JOBS_DIR_SETTING: str(candidate)})
    # a relative path is resolved under the project root, so this is the archive too
    with pytest.raises(TrackingNotAllowed, match="archive"):
        job_tracking_store({JOBS_DIR_SETTING: "data/mlflow/sub"})
    with pytest.raises(TrackingNotAllowed, match="archive"):
        job_tracking_store({JOBS_DIR_SETTING: "data/../data/mlflow"})
    # a symlink to the archive is the archive
    link = tmp_path / "innocent"
    os.symlink(archive, link)
    with pytest.raises(TrackingNotAllowed, match="archive"):
        job_tracking_store({JOBS_DIR_SETTING: str(link)})
    with pytest.raises(TrackingNotAllowed, match="archive"):
        job_tracking_store({JOBS_DIR_SETTING: str(link / "deeper")})
    # the archive's sibling is fine
    assert job_tracking_store(
        {JOBS_DIR_SETTING: str(archive.parent / "mlflow-jobs")}
    ).root.name == ("mlflow-jobs")


def test_postgresql_tracking_needs_no_directory_and_never_uses_the_archive() -> None:
    store = job_tracking_store({"AUTOML_MLFLOW_BACKEND": "postgresql"})
    assert store.backend == "postgresql"
    assert tracking_problem({"AUTOML_MLFLOW_BACKEND": "postgresql"}) is None
    assert "unknown tracking backend" in (
        tracking_problem({"AUTOML_MLFLOW_BACKEND": "oracle"}) or ""
    )
