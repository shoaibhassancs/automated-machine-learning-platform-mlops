"""The one training and finalization sequence, shared by the command line and the worker (P4-4).

`run_pipeline` does what `scripts/datasets.py train` and `finalize` always did, in the same order:
cross-validate, then (for `final`) select, freeze, refit, test once on the held-out part and seal.
It adds no logic of its own: every step is a call into `train`, `selection`, `finalize` and
`tracking`. The entry points differ only in how they read the data, report progress and print.

Stopping a run: `StopRun` derives from `BaseException`, not `Exception`, on purpose.
`run_training` catches `Exception` around each algorithm ("one algorithm failing must not stop the
others"); a cancel raised from the progress callback has to pass through that handler instead of
being recorded as an algorithm failure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl_ops.algorithms import algorithms_for
from automl_ops.dataset_store import DatasetVersion
from automl_ops.finalize import DatasetInfo, FinalizeError, FinalizeResult, finalize
from automl_ops.package import library_versions
from automl_ops.selection import FrozenSelection, SelectionError, SelectionReport
from automl_ops.selection import freeze_selection as _freeze
from automl_ops.selection import select_model as _select
from automl_ops.task import TaskConfig, TaskType
from automl_ops.tracking import JobRecord, record_from_package, record_from_training
from automl_ops.train import (
    AlgorithmRequest,
    ProgressCallback,
    TrainingResult,
    TrainSettings,
    run_training,
)

KINDS = ("train", "final")


class StopRun(BaseException):
    """A run was stopped on purpose (cancel, or the worker is shutting down)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class JobCancelled(StopRun):
    def __init__(self) -> None:
        super().__init__("cancelled", "the job was cancelled")


class WorkerStopping(StopRun):
    def __init__(self) -> None:
        super().__init__("worker_stopped", "the worker was stopped while the job was running")


@dataclass(frozen=True)
class RunSpec:
    kind: str  # "train": cross-validation only; "final": also select, test once and seal
    config: TaskConfig
    requests: tuple[AlgorithmRequest, ...]
    settings: TrainSettings
    selection_metric: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")


@dataclass(frozen=True, eq=False)
class RunOutcome:
    version: DatasetVersion
    training: TrainingResult
    selection: SelectionReport | None
    selection_note: str | None  # why a train run has no selection report (nothing finished)
    frozen: FrozenSelection | None
    final: FinalizeResult | None


def resolve_requests(task: TaskType, keys: Sequence[str] | None) -> list[AlgorithmRequest]:
    """The requested algorithms; with none named, every algorithm of the task (catalogue order)."""
    chosen = [k for k in keys or () if k]
    return [AlgorithmRequest(k) for k in (chosen or [a.key for a in algorithms_for(task)])]


def run_pipeline(
    frame: pd.DataFrame,
    version: DatasetVersion,
    spec: RunSpec,
    models_root: Path | None = None,
    *,
    progress: ProgressCallback | None = None,
    select: bool = False,
    before_final: Callable[[], None] | None = None,
) -> RunOutcome:
    """Run one job's computation. `frame` is a text frame of `version` (already hash-checked).

    `select` adds the selection report to a `train` run (cross-validation numbers only; the test
    part is never touched). A `final` run always selects. `before_final` is called after the
    selection is frozen and before the refit: the worker uses it to enter the "finalizing" phase
    atomically (it may raise `StopRun`). `finalize` itself creates the single test attempt marker.
    """
    training = run_training(frame, spec.config, spec.requests, spec.settings, progress=progress)
    if spec.kind == "train" and not select:
        return RunOutcome(version, training, None, None, None, None)
    try:
        report = _select(training, spec.selection_metric)
    except SelectionError as error:
        if spec.kind == "final":
            raise
        return RunOutcome(version, training, None, str(error), None, None)
    if spec.kind == "train":
        return RunOutcome(version, training, report, None, None, None)
    if models_root is None:
        raise FinalizeError("a final run needs the models folder")
    frozen = _freeze(training, report)
    if before_final is not None:
        before_final()
    result = finalize(
        frame,
        spec.config,
        training,
        frozen,
        DatasetInfo.from_version(version),
        models_root,
        confirm_final_test=True,  # the caller confirmed (--confirm / confirm_final_test)
    )
    return RunOutcome(version, training, report, None, frozen, result)


def tracking_record(outcome: RunOutcome, models_root: Path | None) -> JobRecord:
    """What to log to MLflow for this outcome (only a sealed package is tracked for `final`)."""
    if outcome.final is not None:
        assert models_root is not None
        return record_from_package(models_root, outcome.final.package_id)
    return record_from_training(
        outcome.training.to_dict(),
        DatasetInfo.from_version(outcome.version).to_dict(),
        library_versions(),
    )


def job_result(outcome: RunOutcome) -> dict[str, Any]:
    """The JSON stored on a job: the same payloads as `train --json` / `finalize --json`."""
    return {
        "dataset_ref": outcome.version.ref,
        "sha256": outcome.version.sha256,
        "training": outcome.training.to_dict(),
        "selection": None if outcome.selection is None else outcome.selection.to_dict(),
        "selection_sha256": None if outcome.frozen is None else outcome.frozen.sha256,
        "selection_note": outcome.selection_note,
        "test_result": None if outcome.final is None else outcome.final.test_result,
        "package_id": None if outcome.final is None else outcome.final.package_id,
    }


def spec_to_config(spec: RunSpec) -> dict[str, Any]:
    """The JSON stored on a job row: everything needed to rebuild the run, nothing else."""
    return {
        "target": spec.config.target,
        "task": spec.config.task,
        "excluded": list(spec.config.excluded),
        "algorithms": [{"key": r.key, "params": dict(r.params)} for r in spec.requests],
        "settings": spec.settings.to_dict(),
        "selection_metric": spec.selection_metric,
    }


def spec_from_config(kind: str, config: dict[str, Any]) -> RunSpec:
    """The inverse of `spec_to_config` (the worker calls it on a claimed job)."""
    settings = config["settings"]
    return RunSpec(
        kind,
        TaskConfig(config["target"], config["task"], tuple(config["excluded"])),
        tuple(AlgorithmRequest(a["key"], a["params"]) for a in config["algorithms"]),
        TrainSettings(
            test_fraction=settings["test_fraction"], seed=settings["seed"], folds=settings["folds"]
        ),
        config.get("selection_metric"),
    )
