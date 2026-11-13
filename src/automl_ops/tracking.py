"""Experiment tracking through MLflow (BUILD_PLAN Steps 13-14).

This is the **only** module that imports `mlflow` (a test enforces it). Everything crossing its
boundary is plain data: `JobRecord` in, `TrackedJob` / dictionaries out. No MLflow object leaks.

**What is logged.** One *parent* run per job and one *child* run per algorithm. Children hold the
parameters, per-fold and pooled cross-validation metrics, durations, status and the evaluation JSON;
a failed algorithm is a FAILED child run. For a finalized job the **selected** model's child run
also carries the sealed package files and the single final test result. Regular runs never carry
test metrics. Nothing here reads data rows, refits a model or touches the held-out test part: the
input is the sealed package's own manifest and test result (JSON), or a `TrainingResult.to_dict()`.

**Logging happens after sealing.** A tracking failure cannot unseal or spend a package. The sealed
package is self-sufficient: `record_from_package` rebuilds every run from `manifest.json` and
`test_result.json` alone (no unpickling), after the seal hashes are verified.

**Idempotent and resumable.** A job is identified by the tag `platform.job_key` (the package id for
a finalized job). `log_job` finds an existing incomplete parent and children and completes the
missing work instead of creating duplicates: metrics carry a fixed timestamp and step (re-logging
the same point is not a new record), params and tags are rewritten with the same values, artifacts
are re-copied only when missing or of the wrong size, and each run gets its
`platform.logging=complete` tag as its **last** write. The parent is marked complete only after
every child verifies. A completed job rejects another attempt (`AlreadyTrackedError`).

**Store.** SQLite metadata (`<root>/mlflow.db`) and a local artifact directory (`<root>/artifacts`),
both explicit. This is the Phase 3 development store, kept separate from the future application
database. Telemetry is disabled for this process before MLflow is imported.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from automl_ops.package import (
    ATTEMPT_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    PARITY_FILE,
    SEAL_FILE,
    TEST_RESULT_FILE,
    PackageError,
    package_path,
    package_state,
    verify_package,
)

TRACKING_FORMAT = 1
EXPERIMENT = "automl-ops"
DEFAULT_ROOT = Path("data/mlflow")
DEFAULT_UI_PORT = 5001  # 5000 is taken by macOS AirPlay Receiver on recent macOS versions
PACKAGE_ARTIFACT_DIR = "package"
PACKAGE_FILES = (MODEL_FILE, MANIFEST_FILE, PARITY_FILE, ATTEMPT_FILE, TEST_RESULT_FILE, SEAL_FILE)

# Tags that identify and order the platform's runs.
T_JOB = "platform.job_key"
T_ROLE = "platform.role"  # "job" (parent) or "algorithm" (child)
T_ALGORITHM = "platform.algorithm"
T_STATE = "platform.logging"  # "incomplete" until the run's last write, then "complete"
COMPLETE = "complete"
INCOMPLETE = "incomplete"
PARENT_TAG = "mlflow.parentRunId"

# Environment that keeps MLflow from sending anything out: documented switch plus the generic one,
# and the hint banner that MLflow prints when it detects a coding agent.
QUIET_ENVIRONMENT = {
    "MLFLOW_DISABLE_TELEMETRY": "true",
    "DO_NOT_TRACK": "true",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}


class TrackingError(RuntimeError):
    """Tracking could not be completed. The package (if any) is unaffected."""


class AlreadyTrackedError(TrackingError):
    """The job's runs are already complete; they are never logged twice."""


class JobNotFoundError(TrackingError):
    """No such job (or no tracking store yet) for a read-only query."""


def quiet_environment() -> dict[str, str]:
    """The environment variables every MLflow process of this project must run with."""
    return dict(QUIET_ENVIRONMENT)


def apply_quiet_environment() -> None:
    os.environ.update(QUIET_ENVIRONMENT)


@dataclass(frozen=True)
class TrackingStore:
    """Explicit SQLite database and artifact directory under one root (default `data/mlflow`)."""

    root: Path = DEFAULT_ROOT

    @property
    def db_path(self) -> Path:
        return self.root.resolve() / "mlflow.db"

    @property
    def artifact_path(self) -> Path:
        return self.root.resolve() / "artifacts"

    @property
    def tracking_uri(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    @property
    def artifact_uri(self) -> str:
        return self.artifact_path.as_uri()

    def ui_command(self, port: int = DEFAULT_UI_PORT) -> list[str]:
        """The explicit command that serves this store on localhost only."""
        return [
            "mlflow",
            "server",
            "--backend-store-uri",
            self.tracking_uri,
            "--default-artifact-root",
            self.artifact_uri,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]


@dataclass(frozen=True)
class JobRecord:
    """Everything needed to rebuild the runs of one job, as plain data."""

    job_key: str
    kind: str  # "train" (cross-validation only) or "final" (sealed package with a test result)
    created_ms: int
    dataset: dict[str, Any]
    training: dict[str, Any]  # TrainingResult.to_dict()
    libraries: dict[str, str]
    selection: dict[str, Any] | None = None
    selection_sha256: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] | None = None
    package_id: str | None = None
    package_dir: Path | None = None
    test_result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("train", "final"):
            raise TrackingError(f"unknown job kind {self.kind!r}")
        if self.kind == "train" and self.test_result is not None:
            raise TrackingError("a cross-validation-only job cannot carry a test result")
        if self.kind == "final" and (
            self.test_result is None or self.package_dir is None or self.selection is None
        ):
            raise TrackingError("a finalized job needs its package, selection and test result")


@dataclass(frozen=True)
class TrackedJob:
    job_key: str
    parent_run_id: str
    child_run_ids: dict[str, str]
    resumed: bool


def new_train_job_key() -> str:
    return f"t-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def _epoch_ms(text: str) -> int:
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def record_from_training(
    training: dict[str, Any],
    dataset: dict[str, Any],
    libraries: dict[str, str],
    *,
    job_key: str | None = None,
    created_ms: int | None = None,
) -> JobRecord:
    """A cross-validation-only record (`train --track`); `training` is a `TrainingResult` dict."""
    return JobRecord(
        job_key=job_key or new_train_job_key(),
        kind="train",
        created_ms=created_ms
        if created_ms is not None
        else int(datetime.now(UTC).timestamp() * 1000),
        dataset=dataset,
        training=training,
        libraries=libraries,
        settings=training["settings"],
    )


def record_from_package(models_root: Path, package_id: str) -> JobRecord:
    """Rebuild a job from a **sealed** package: manifest and test result only, no unpickling.

    Refuses an open, spent, unsealed or tampered package (seal hashes are checked first)."""
    try:
        path = package_path(models_root, package_id)
    except PackageError as error:
        raise TrackingError(str(error)) from error
    state = package_state(path)
    if state != "sealed":
        raise TrackingError(
            f"{package_id} is {state}, not sealed: only sealed packages are tracked"
        )
    check = verify_package(models_root, package_id, deep=False)
    if not check.ok:
        raise TrackingError(f"{package_id} failed verification: " + "; ".join(check.problems))
    try:
        manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
        test_result = json.loads((path / TEST_RESULT_FILE).read_text(encoding="utf-8"))
        return JobRecord(
            job_key=manifest["package_id"],
            kind="final",
            created_ms=_epoch_ms(manifest["created_at"]),
            dataset=manifest["dataset"],
            training=manifest["training"],
            libraries=manifest["libraries"],
            selection=manifest["selection"],
            selection_sha256=manifest["selection_sha256"],
            settings=manifest["settings"],
            split=manifest["split"],
            package_id=manifest["package_id"],
            package_dir=path,
            test_result=test_result,
        )
    except (KeyError, ValueError, OSError) as error:
        raise TrackingError(
            f"{package_id}: the manifest lacks what tracking needs ({error!r})"
        ) from error


# --------------------------------------------------------------------------------------------
# The MLflow boundary
# --------------------------------------------------------------------------------------------


class _Api:
    """Thin wrapper holding the MLflow client; the only place MLflow names appear."""

    def __init__(self, store: TrackingStore, *, read_only: bool = False) -> None:
        apply_quiet_environment()  # before MLflow is imported
        from mlflow import MlflowClient  # noqa: PLC0415 - lazy so non-tracking commands stay fast
        from mlflow.entities import Metric, Param, RunTag  # noqa: PLC0415

        self.read_only = read_only
        if read_only:
            if not store.db_path.exists():
                raise JobNotFoundError(f"no tracking store at {store.db_path}")
        else:
            store.root.mkdir(parents=True, exist_ok=True)
            store.artifact_path.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.client: Any = MlflowClient(tracking_uri=store.tracking_uri)
        self._metric, self._param, self._tag = Metric, Param, RunTag
        self.experiment_id = self._experiment()

    def _experiment(self) -> str:
        found = self.client.get_experiment_by_name(EXPERIMENT)
        if found is not None:
            if found.lifecycle_stage != "active":
                raise TrackingError(f"the MLflow experiment {EXPERIMENT!r} was deleted")
            return str(found.experiment_id)
        if self.read_only:
            raise JobNotFoundError(f"the tracking store has no {EXPERIMENT!r} experiment yet")
        return str(
            self.client.create_experiment(EXPERIMENT, artifact_location=self.store.artifact_uri)
        )

    def find(self, job_key: str, role: str, algorithm: str | None = None) -> Any | None:
        clauses = [f"tags.`{T_JOB}` = '{job_key}'", f"tags.`{T_ROLE}` = '{role}'"]
        if algorithm is not None:
            clauses.append(f"tags.`{T_ALGORITHM}` = '{algorithm}'")
        runs = self.client.search_runs([self.experiment_id], " and ".join(clauses), max_results=10)
        if len(runs) > 1:
            raise TrackingError(
                f"{len(runs)} runs match {job_key} / {role} / {algorithm}; the tracking store "
                "is inconsistent and needs a manual look"
            )
        return runs[0] if runs else None

    def create(self, name: str, start_ms: int, tags: dict[str, str]) -> str:
        run = self.client.create_run(
            self.experiment_id, start_time=start_ms, tags=tags, run_name=name
        )
        return str(run.info.run_id)

    def write(
        self,
        run_id: str,
        *,
        metrics: list[tuple[str, float, int]] = (),  # type: ignore[assignment]
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        timestamp: int,
    ) -> None:
        batch_metrics = [self._metric(k, float(v), timestamp, step) for k, v, step in metrics]
        batch_params = [self._param(k, _text(v)) for k, v in (params or {}).items()]
        batch_tags = [self._tag(k, v) for k, v in (tags or {}).items()]
        for start in range(0, max(len(batch_metrics), 1), 500):
            self.client.log_batch(
                run_id,
                metrics=batch_metrics[start : start + 500],
                params=batch_params if start == 0 else [],
                tags=batch_tags if start == 0 else [],
            )

    def text(self, run_id: str, text: str, artifact_file: str) -> None:
        self.client.log_text(run_id, text, artifact_file)

    def file(self, run_id: str, path: Path, artifact_dir: str) -> None:
        present = {a.path: a.file_size for a in self.client.list_artifacts(run_id, artifact_dir)}
        if present.get(f"{artifact_dir}/{path.name}") == path.stat().st_size:
            return  # already copied completely (a partial copy has the wrong size)
        self.client.log_artifact(run_id, str(path), artifact_dir)

    def finish(self, run_id: str, status: str, end_ms: int) -> None:
        self.client.set_terminated(run_id, status, end_time=end_ms)

    def mark(self, run_id: str, key: str, value: str) -> None:
        self.client.set_tag(run_id, key, value)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, allow_nan=False, default=str)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and value == value
        and (value not in (float("inf"), float("-inf")))
    )


# --------------------------------------------------------------------------------------------
# What each run holds
# --------------------------------------------------------------------------------------------


def _child_metrics(result: dict[str, Any]) -> list[tuple[str, float, int]]:
    """Cross-validation numbers only. Undefined (None) values are skipped, never logged as zero."""
    out: list[tuple[str, float, int]] = []
    for name, entry in sorted(result.get("summary", {}).items()):
        if _finite(entry.get("mean")):
            out.append((f"cv_{name}_mean", entry["mean"], 0))
        if _finite(entry.get("std")):
            out.append((f"cv_{name}_std", entry["std"], 0))
    oof = result.get("oof") or {}
    for name, value in sorted(oof.get("metrics", {}).items()):
        if _finite(value):
            out.append((f"oof_{name}", value, 0))
    for fold, scores in enumerate(result.get("fold_metrics", [])):
        for name, value in sorted(scores.items()):
            if _finite(value):
                out.append((f"fold_{name}", value, fold))
    if _finite(result.get("seconds")):
        out.append(("fit_seconds", result["seconds"], 0))
    for fold, seconds in enumerate(result.get("fold_seconds", [])):
        if _finite(seconds):
            out.append(("fold_fit_seconds", seconds, fold))
    return out


def _test_metrics(test_result: dict[str, Any]) -> list[tuple[str, float, int]]:
    metrics = test_result["evaluation"]["metrics"]
    return [(f"test_{k}", v, 0) for k, v in sorted(metrics.items()) if _finite(v)]


def _log_child(api: _Api, record: JobRecord, parent_id: str, result: dict[str, Any]) -> str:
    key = result["algorithm"]
    ts = record.created_ms
    existing = api.find(record.job_key, "algorithm", key)
    if existing is not None and existing.data.tags.get(T_STATE) == COMPLETE:
        return str(existing.info.run_id)
    selected = record.selection is not None and record.selection["selected"]["algorithm"] == key
    if existing is None:
        run_id = api.create(
            key,
            ts,
            {
                T_JOB: record.job_key,
                T_ROLE: "algorithm",
                T_ALGORITHM: key,
                T_STATE: INCOMPLETE,
                PARENT_TAG: parent_id,
            },
        )
    else:
        run_id = str(existing.info.run_id)
    tags = {
        "platform.status": result["status"],
        "platform.selected": str(selected).lower(),
        "platform.label": result["label"],
    }
    if result.get("error"):
        tags["platform.error_type"] = str(result["error"].get("type", ""))
        tags["platform.error_message"] = str(result["error"].get("message", ""))[:5000]
    params = {f"param.{k}": v for k, v in sorted(result.get("params", {}).items())}
    metrics = _child_metrics(result)
    if selected and record.test_result is not None:
        metrics += _test_metrics(record.test_result)
        tags["platform.final_test"] = "logged"
    api.write(run_id, metrics=metrics, params=params, tags=tags, timestamp=ts)
    api.text(run_id, _json(result), "evaluation.json")
    if selected and record.package_dir is not None and record.test_result is not None:
        api.text(run_id, _json(record.test_result), "test_result.json")
        for name in PACKAGE_FILES:
            api.file(run_id, record.package_dir / name, PACKAGE_ARTIFACT_DIR)
    end = ts + int(float(result.get("seconds") or 0) * 1000)
    api.finish(run_id, "FINISHED" if result["status"] == "ok" else "FAILED", end)
    api.mark(run_id, T_STATE, COMPLETE)  # the child's last write
    return run_id


def _parent_tags(record: JobRecord) -> dict[str, str]:
    dataset = record.dataset
    tags = {
        "platform.kind": record.kind,
        "platform.tracking_format": str(TRACKING_FORMAT),
        "platform.task": str(record.training["task"]),
        "platform.target": str(record.training["target"]),
        "platform.dataset_ref": str(dataset["ref"]),
        "platform.dataset_sha256": str(dataset["sha256"]),
        "platform.seed": str(record.settings.get("seed")),
    }
    if record.split and record.split.get("test_sha256"):
        tags["platform.test_rows_sha256"] = str(record.split["test_sha256"])
    if record.selection is not None:
        tags["platform.selected_algorithm"] = str(record.selection["selected"]["algorithm"])
        tags["platform.selection_metric"] = str(record.selection["metric"])
    if record.selection_sha256:
        tags["platform.selection_sha256"] = record.selection_sha256
    if record.package_id:
        tags["platform.package_id"] = record.package_id
    for name, value in sorted(record.libraries.items()):
        tags[f"lib.{name}"] = str(value)
    return tags


def log_job(record: JobRecord, store: TrackingStore) -> TrackedJob:
    """Log (or resume logging) one job. Raises `AlreadyTrackedError` if it is already complete and
    `TrackingError` for any other failure; the cause is chained."""
    try:
        return _log_job(record, store)
    except TrackingError:
        raise
    except Exception as error:
        raise TrackingError(f"{type(error).__name__}: {error}") from error


def _log_job(record: JobRecord, store: TrackingStore) -> TrackedJob:
    api = _Api(store)
    ts = record.created_ms
    parent = api.find(record.job_key, "job")
    if parent is not None and parent.data.tags.get(T_STATE) == COMPLETE:
        raise AlreadyTrackedError(f"{record.job_key} is already tracked; it is never logged twice")
    resumed = parent is not None
    if parent is None:
        parent_id = api.create(
            f"job {record.job_key}",
            ts,
            {T_JOB: record.job_key, T_ROLE: "job", T_STATE: INCOMPLETE, **_parent_tags(record)},
        )
    else:
        parent_id = str(parent.info.run_id)

    training = record.training
    parent_metrics: list[tuple[str, float, int]] = []
    if _finite(training.get("seconds")):
        parent_metrics.append(("training_seconds", training["seconds"], 0))
    if record.selection is not None:
        chosen = record.selection["ranking"][0]
        if _finite(chosen.get("metric_mean")):
            parent_metrics.append(
                (f"selected_cv_{record.selection['metric']}_mean", chosen["metric_mean"], 0)
            )
    params = {
        "seed": record.settings.get("seed"),
        "folds": record.settings.get("folds"),
        "test_fraction": record.settings.get("test_fraction"),
        "folds_used": training.get("folds", {}).get("used"),
        "algorithms": [r["algorithm"] for r in training["results"]],
    }
    api.write(
        parent_id, metrics=parent_metrics, params=params, tags=_parent_tags(record), timestamp=ts
    )
    api.text(parent_id, _json(training), "training.json")
    api.text(
        parent_id,
        _json({"dataset": record.dataset, "libraries": record.libraries}),
        "provenance.json",
    )
    if record.selection is not None:
        api.text(parent_id, _json(record.selection), "selection_report.json")

    children: dict[str, str] = {}
    for result in training["results"]:
        children[result["algorithm"]] = _log_child(api, record, parent_id, result)
    _verify_children(api, record, children)

    api.finish(parent_id, "FINISHED", ts + int(float(training.get("seconds") or 0) * 1000))
    api.mark(parent_id, T_STATE, COMPLETE)  # the parent's last write
    return TrackedJob(record.job_key, parent_id, children, resumed)


def _verify_children(api: _Api, record: JobRecord, children: dict[str, str]) -> None:
    """Read back: every expected child exists once and is complete, and the selected child has its
    test result. Only then may the parent be completed."""
    expected = [r["algorithm"] for r in record.training["results"]]
    for key in expected:
        run = api.find(record.job_key, "algorithm", key)
        if run is None or run.data.tags.get(T_STATE) != COMPLETE:
            raise TrackingError(f"the run for {key} is missing or incomplete")
    if record.selection is not None and record.test_result is not None:
        selected = record.selection["selected"]["algorithm"]
        run = api.find(record.job_key, "algorithm", selected)
        needed = {name for name, _, _ in _test_metrics(record.test_result)}
        if run is None or not needed <= set(run.data.metrics):
            raise TrackingError("the selected model's run does not hold the final test result")


# --------------------------------------------------------------------------------------------
# Reading back (plain dictionaries only)
# --------------------------------------------------------------------------------------------


def describe_job(job_key: str, store: TrackingStore) -> dict[str, Any] | None:
    """The job's runs as plain data, or None if the job is unknown. Used by tests and the CLI."""
    if not store.db_path.exists():
        return None
    api = _Api(store)
    parent = api.find(job_key, "job")
    if parent is None:
        return None
    runs = api.client.search_runs(
        [api.experiment_id], f"tags.`{T_JOB}` = '{job_key}' and tags.`{T_ROLE}` = 'algorithm'"
    )

    def shape(run: Any) -> dict[str, Any]:
        history = {
            name: len(api.client.get_metric_history(run.info.run_id, name))
            for name in run.data.metrics
        }
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "tags": dict(run.data.tags),
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "metric_points": history,
            "artifacts": sorted(_artifact_paths(api, run.info.run_id)),
        }

    return {
        "job_key": job_key,
        "parent": shape(parent),
        "children": {r.data.tags[T_ALGORITHM]: shape(r) for r in runs},
    }


def _artifact_paths(api: _Api, run_id: str, prefix: str | None = None) -> list[str]:
    paths: list[str] = []
    for item in api.client.list_artifacts(run_id, prefix):
        if item.is_dir:
            paths.extend(_artifact_paths(api, run_id, item.path))
        else:
            paths.append(str(item.path))
    return paths


def artifact_file(job_key: str, algorithm: str, relative: str, store: TrackingStore) -> Path:
    """Local path of a logged artifact of a child run (for verification)."""
    api = _Api(store)
    run = api.find(job_key, "algorithm", algorithm)
    if run is None:
        raise TrackingError(f"no run for {job_key} / {algorithm}")
    return Path(api.client.download_artifacts(run.info.run_id, relative))


# --------------------------------------------------------------------------------------------
# History and detail (Step 15): read-only, plain data. Nothing here writes to the store.
# --------------------------------------------------------------------------------------------

MAX_RUNS = 5000  # one search page; far above what a single-user project will hold


@dataclass(frozen=True)
class JobSummary:
    """One row of the job history (a parent run, plus a count of its algorithm runs)."""

    job_key: str
    kind: str
    state: str
    start_ms: int
    task: str | None
    target: str | None
    dataset_ref: str | None
    dataset_sha256: str | None
    seed: str | None
    folds_used: str | None
    selected_algorithm: str | None
    selection_metric: str | None
    selected_cv_mean: float | None
    algorithms_ok: int
    algorithms_failed: int
    has_test_result: bool
    package_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LoggedAlgorithm:
    """One algorithm's child run as stored: tags, params, metrics (latest value, and the per-fold
    series) and, for the selected model of a finalized job, the recorded test result."""

    algorithm: str
    run_status: str
    tags: dict[str, str]
    params: dict[str, str]
    metrics: dict[str, float]
    fold_metrics: dict[str, list[float]]
    test_result: dict[str, Any] | None


@dataclass(frozen=True)
class LoggedJob:
    """A whole job as stored in MLflow, in the platform's own shapes (no MLflow objects)."""

    job_key: str
    state: str
    kind: str
    start_ms: int
    tags: dict[str, str]
    params: dict[str, str]
    metrics: dict[str, float]
    dataset: dict[str, Any]
    libraries: dict[str, str]
    training: dict[str, Any]
    selection: dict[str, Any] | None
    algorithms: dict[str, LoggedAlgorithm]


def _all_runs(api: _Api) -> list[Any]:
    runs = api.client.search_runs([api.experiment_id], max_results=MAX_RUNS)
    if len(runs) >= MAX_RUNS:
        raise TrackingError(f"more than {MAX_RUNS - 1} runs; the history query is not paged")
    return list(runs)


def list_jobs(
    store: TrackingStore,
    *,
    kind: str | None = None,
    task: str | None = None,
    dataset: str | None = None,
    algorithm: str | None = None,
    state: str | None = None,
    limit: int | None = None,
) -> list[JobSummary]:
    """Jobs, newest first. `dataset` matches the dataset reference or the start of its SHA-256;
    `algorithm` matches the selected algorithm. A missing store gives an empty list."""
    try:
        api = _Api(store, read_only=True)
    except JobNotFoundError:
        return []
    runs = _all_runs(api)
    children: dict[str, list[Any]] = {}
    for run in runs:
        if run.data.tags.get(T_ROLE) == "algorithm":
            children.setdefault(run.data.tags.get(T_JOB, ""), []).append(run)
    out: list[JobSummary] = []
    for run in runs:
        tags = run.data.tags
        if tags.get(T_ROLE) != "job":
            continue
        key = tags[T_JOB]
        subs = children.get(key, [])
        metric = tags.get("platform.selection_metric")
        item = JobSummary(
            job_key=key,
            kind=tags.get("platform.kind", "?"),
            state=tags.get(T_STATE, INCOMPLETE),
            start_ms=int(run.info.start_time),
            task=tags.get("platform.task"),
            target=tags.get("platform.target"),
            dataset_ref=tags.get("platform.dataset_ref"),
            dataset_sha256=tags.get("platform.dataset_sha256"),
            seed=tags.get("platform.seed"),
            folds_used=run.data.params.get("folds_used"),
            selected_algorithm=tags.get("platform.selected_algorithm"),
            selection_metric=metric,
            selected_cv_mean=run.data.metrics.get(f"selected_cv_{metric}_mean") if metric else None,
            algorithms_ok=sum(1 for r in subs if r.data.tags.get("platform.status") == "ok"),
            algorithms_failed=sum(1 for r in subs if r.data.tags.get("platform.status") != "ok"),
            has_test_result=any(r.data.tags.get("platform.final_test") == "logged" for r in subs),
            package_id=tags.get("platform.package_id"),
        )
        if kind and item.kind != kind:
            continue
        if task and item.task != task:
            continue
        if state and item.state != state:
            continue
        if algorithm and item.selected_algorithm != algorithm:
            continue
        if dataset and not (
            item.dataset_ref == dataset or (item.dataset_sha256 or "").startswith(dataset)
        ):
            continue
        out.append(item)
    out.sort(key=lambda j: (-j.start_ms, j.job_key))
    return out[:limit] if limit else out


def _read_json_artifact(api: _Api, run_id: str, name: str) -> Any | None:
    """A JSON artifact of a run, or None if it does not exist. Downloaded into a temporary
    folder, so the store is never written."""
    present = {a.path for a in api.client.list_artifacts(run_id)}
    if name not in present:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(api.client.download_artifacts(run_id, name, dst_path=tmp))
        return json.loads(path.read_text(encoding="utf-8"))


def read_job(job_key: str, store: TrackingStore) -> LoggedJob:
    """The whole logged job as plain data. Raises `JobNotFoundError` for an unknown job, and
    `TrackingError` if the job's records are unreadable (for example logging never finished)."""
    api = _Api(store, read_only=True)
    parent = api.find(job_key, "job")
    if parent is None:
        raise JobNotFoundError(f"no tracked job {job_key!r}")
    run_id = str(parent.info.run_id)
    state = parent.data.tags.get(T_STATE, INCOMPLETE)
    training = _read_json_artifact(api, run_id, "training.json")
    provenance = _read_json_artifact(api, run_id, "provenance.json")
    if training is None or provenance is None:
        raise TrackingError(
            f"{job_key} has no complete record (logging state: {state}); finish it with "
            "`track-package` for a finalized job"
        )
    algorithms: dict[str, LoggedAlgorithm] = {}
    runs = api.client.search_runs(
        [api.experiment_id],
        f"tags.`{T_JOB}` = '{job_key}' and tags.`{T_ROLE}` = 'algorithm'",
        max_results=1000,
    )
    for run in runs:
        name = run.data.tags[T_ALGORITHM]
        series = {
            metric: [
                point.value
                for point in sorted(
                    api.client.get_metric_history(run.info.run_id, metric), key=lambda m: m.step
                )
            ]
            for metric in run.data.metrics
            if metric.startswith("fold_") and metric != "fold_fit_seconds"
        }
        test_result = (
            _read_json_artifact(api, str(run.info.run_id), "test_result.json")
            if run.data.tags.get("platform.final_test") == "logged"
            else None
        )
        algorithms[name] = LoggedAlgorithm(
            algorithm=name,
            run_status=str(run.info.status),
            tags=dict(run.data.tags),
            params=dict(run.data.params),
            metrics=dict(run.data.metrics),
            fold_metrics=series,
            test_result=test_result,
        )
    tags = dict(parent.data.tags)
    return LoggedJob(
        job_key=job_key,
        state=state,
        kind=tags.get("platform.kind", "?"),
        start_ms=int(parent.info.start_time),
        tags=tags,
        params=dict(parent.data.params),
        metrics=dict(parent.data.metrics),
        dataset=provenance["dataset"],
        libraries={k[4:]: v for k, v in tags.items() if k.startswith("lib.")},
        training=training,
        selection=_read_json_artifact(api, run_id, "selection_report.json"),
        algorithms=algorithms,
    )


def telemetry_disabled() -> bool:
    """True when MLflow's own check reports telemetry off in this process."""
    apply_quiet_environment()
    from mlflow.telemetry.utils import is_telemetry_disabled  # noqa: PLC0415

    return bool(is_telemetry_disabled())


def tracking_state(store: TrackingStore) -> dict[str, Any]:
    """Counts for a quick look: runs by role and logging state."""
    if not store.db_path.exists():
        return {"runs": 0}
    api = _Api(store)
    runs = api.client.search_runs([api.experiment_id], max_results=5000)
    counts: dict[str, int] = {}
    for run in runs:
        label = f"{run.data.tags.get(T_ROLE)}:{run.data.tags.get(T_STATE)}"
        counts[label] = counts.get(label, 0) + 1
    return {"runs": len(runs), "by_role_and_state": counts}
