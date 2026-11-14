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

**Store.** Two backends behind one `TrackingStore` (P4-2b): SQLite (`<root>/mlflow.db` plus
`<root>/artifacts`; the default, the tests and the historical archive) or PostgreSQL (MLflow's own
`mlflow` database on the platform's server, artifacts in `data/mlflow-pg/artifacts`). PostgreSQL is
set up only by an explicit `mlflow-init`; reads and status checks never create it. Telemetry is
disabled for this process before MLflow is imported.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
BACKENDS = ("sqlite", "postgresql")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PG_ARTIFACT_ROOT = PROJECT_ROOT / "data" / "mlflow-pg" / "artifacts"
DEFAULT_MLFLOW_DATABASE = "mlflow"
# MLflow's own schema: runs.artifact_uri is VARCHAR(200) and experiments.artifact_location is
# VARCHAR(256). PostgreSQL enforces them (SQLite does not). With an explicit experiment location a
# run's URI is `<location>/<32-character run id>/artifacts`.
RUN_URI_LIMIT = 200
EXPERIMENT_LOCATION_LIMIT = 256
RUN_URI_SUFFIX_LENGTH = len("/") + 32 + len("/artifacts")
_DATABASE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
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


class ArtifactLocationError(TrackingError):
    """An artifact location that PostgreSQL would reject (or that disagrees with the store)."""


def _dbconfig() -> Any:
    from automl_ops.db import config  # noqa: PLC0415 - imports SQLAlchemy; keep it lazy

    return config


def scrub(text: str) -> str:
    """Remove the database password (and its URL-encoded form) from a message."""
    try:
        secret = _dbconfig().database_url().password
    except Exception:  # no settings, nothing to hide
        return text
    if secret:
        for variant in {secret, quote(secret, safe="")}:
            text = text.replace(variant, "***")
    return text


def check_artifact_location(location: str) -> int:
    """Pre-flight for PostgreSQL: the longest run artifact URI this location would produce.

    Raises `ArtifactLocationError` if that exceeds MLflow's 200-character column (or the location
    its own 256). Runs nothing and touches nothing, so it can run before any experiment or run
    exists. Returns the longest run URI length."""
    base = location.rstrip("/")
    longest = len(base) + RUN_URI_SUFFIX_LENGTH
    if longest > RUN_URI_LIMIT or len(location) > EXPERIMENT_LOCATION_LIMIT:
        raise ArtifactLocationError(
            f"the artifact location is {len(base)} characters, so a run's artifact URI would be "
            f"{longest}; MLflow's PostgreSQL schema limits it to {RUN_URI_LIMIT} (the location to "
            f"at most {RUN_URI_LIMIT - RUN_URI_SUFFIX_LENGTH}). Set a shorter "
            f"AUTOML_MLFLOW_ARTIFACT_ROOT or use a shorter project path. Location: {base}"
        )
    return longest


def quiet_environment() -> dict[str, str]:
    """The environment variables every MLflow process of this project must run with."""
    return dict(QUIET_ENVIRONMENT)


def apply_quiet_environment() -> None:
    os.environ.update(QUIET_ENVIRONMENT)


@dataclass(frozen=True)
class TrackingStore:
    """Where MLflow keeps its records. Two backends, one implementation:

    * `sqlite` (default; tests and the historical archive): `<root>/mlflow.db` plus
      `<root>/artifacts`, exactly as in Phase 3.
    * `postgresql`: MLflow's own database (`mlflow` by default) on the platform's PostgreSQL server,
      same credentials as `automl_ops.db`; artifacts in `artifact_root` (default
      `data/mlflow-pg/artifacts`). The connection URL is built on demand and never stored, printed
      or placed on a command line.
    """

    root: Path = DEFAULT_ROOT
    backend: str = "sqlite"
    artifact_root: Path | None = None  # PostgreSQL only; an absolute path

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise TrackingError(f"unknown tracking backend {self.backend!r}; use one of {BACKENDS}")
        if self.artifact_root is not None and not self.artifact_root.is_absolute():
            raise TrackingError("the artifact root must be an absolute path")

    @staticmethod
    def from_settings(
        root: Path | None = None,
        backend: str | None = None,
        env: dict[str, str] | None = None,
    ) -> TrackingStore:
        """The store chosen by the command line, else `AUTOML_MLFLOW_BACKEND` (environment or
        `.env`), else SQLite. `AUTOML_MLFLOW_ARTIFACT_ROOT` overrides the PostgreSQL root."""
        settings = _dbconfig().setting
        chosen = backend or settings("AUTOML_MLFLOW_BACKEND", "sqlite", env)
        artifact = settings("AUTOML_MLFLOW_ARTIFACT_ROOT", None, env)
        return TrackingStore(
            root if root is not None else DEFAULT_ROOT,
            str(chosen),
            Path(artifact) if artifact and chosen == "postgresql" else None,
        )

    @property
    def database_name(self) -> str:
        name = str(_dbconfig().setting("AUTOML_MLFLOW_DB_NAME", DEFAULT_MLFLOW_DATABASE))
        if not _DATABASE_NAME.fullmatch(name):
            raise TrackingError(f"invalid MLflow database name {name!r}")
        return name

    @property
    def db_path(self) -> Path:
        if self.backend != "sqlite":
            raise TrackingError("only the SQLite backend has a database file")
        return self.root.resolve() / "mlflow.db"

    @property
    def artifact_path(self) -> Path:
        if self.backend == "postgresql":
            return (self.artifact_root or DEFAULT_PG_ARTIFACT_ROOT).resolve()
        return self.root.resolve() / "artifacts"

    @property
    def tracking_uri(self) -> str:
        """The backend URI. For PostgreSQL it contains the password: pass it to MLflow in this
        process only, never print it or put it in a command line."""
        if self.backend == "sqlite":
            return f"sqlite:///{self.db_path.as_posix()}"
        config = _dbconfig()
        try:
            url = config.database_url().set(database=self.database_name)
        except config.DatabaseConfigError as error:
            raise TrackingError(str(error)) from error
        return str(url.render_as_string(hide_password=False))

    @property
    def artifact_uri(self) -> str:
        return self.artifact_path.as_uri()

    def describe(self) -> str:
        """A printable description; never contains the password."""
        if self.backend == "sqlite":
            return str(self.db_path)
        config = _dbconfig()
        try:
            url = config.database_url().set(database=self.database_name)
        except config.DatabaseConfigError as error:
            raise TrackingError(str(error)) from error
        return f"PostgreSQL {config.describe(url)} (artifacts {self.artifact_path})"

    def exists(self) -> bool:
        """Is there an initialised store? Never creates anything."""
        if self.backend == "sqlite":
            return self.db_path.exists()
        state = _postgres_state(self)
        return state.database_exists and state.schema_present

    def ui_command(self, port: int = DEFAULT_UI_PORT) -> list[str]:
        """The explicit command that serves this store on localhost only. For PostgreSQL the
        backend and artifact locations are *not* arguments (they would expose the password in the
        process list); `ui_environment()` supplies them."""
        common = ["--host", "127.0.0.1", "--port", str(port)]
        if self.backend == "postgresql":
            return ["mlflow", "server", *common]
        return [
            "mlflow",
            "server",
            "--backend-store-uri",
            self.tracking_uri,
            "--default-artifact-root",
            self.artifact_uri,
            *common,
        ]

    def ui_environment(self) -> dict[str, str]:
        """Extra environment for the UI process (PostgreSQL only; contains the password)."""
        if self.backend != "postgresql":
            return {}
        return {
            "MLFLOW_BACKEND_STORE_URI": self.tracking_uri,
            "MLFLOW_DEFAULT_ARTIFACT_ROOT": self.artifact_uri,
        }


@dataclass(frozen=True)
class _PgState:
    database_exists: bool
    schema_present: bool
    revision: str | None


def _postgres_state(store: TrackingStore) -> _PgState:
    """Does the MLflow database exist, and has MLflow created its schema? Creates nothing."""
    from sqlalchemy import create_engine, text  # noqa: PLC0415
    from sqlalchemy.exc import SQLAlchemyError  # noqa: PLC0415

    config = _dbconfig()
    options: dict[str, Any] = {"connect_args": {"connect_timeout": config.CONNECT_TIMEOUT_SECONDS}}
    try:
        url = config.database_url()
        name = store.database_name
        admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT", **options)
        try:
            with admin.connect() as connection:
                found = connection.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
                ).first()
        finally:
            admin.dispose()
        if found is None:
            return _PgState(False, False, None)
        engine = create_engine(url.set(database=name), **options)
        try:
            with engine.connect() as connection:
                present = bool(
                    connection.execute(
                        text("SELECT to_regclass('public.alembic_version') IS NOT NULL")
                    ).scalar_one()
                )
                revision = (
                    connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
                    if present
                    else None
                )
        finally:
            engine.dispose()
        return _PgState(True, present, None if revision is None else str(revision))
    except (SQLAlchemyError, config.DatabaseConfigError) as error:
        reason = str(getattr(error, "orig", error)).strip().splitlines()[0]
        raise TrackingError(
            scrub(f"cannot reach the PostgreSQL tracking database: {reason}")
        ) from None


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
    extra_tags: dict[str, str] = field(default_factory=dict)  # provenance notes, `platform.*` only

    def __post_init__(self) -> None:
        if self.kind not in ("train", "final"):
            raise TrackingError(f"unknown job kind {self.kind!r}")
        for key, value in self.extra_tags.items():
            if not key.startswith("platform.") or not isinstance(value, str):
                raise TrackingError(
                    f"extra tags must be platform.* names with text values: {key!r}"
                )
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
    extra_tags: dict[str, str] | None = None,
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
        extra_tags=dict(extra_tags or {}),
    )


def record_from_package(
    models_root: Path, package_id: str, extra_tags: dict[str, str] | None = None
) -> JobRecord:
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
            extra_tags=dict(extra_tags or {}),
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

    def __init__(
        self,
        store: TrackingStore,
        *,
        read_only: bool = False,
        initialise: bool = False,
        experiment: bool = True,
    ) -> None:
        apply_quiet_environment()  # before MLflow is imported
        from mlflow import MlflowClient  # noqa: PLC0415 - lazy so non-tracking commands stay fast
        from mlflow.entities import Metric, Param, RunTag  # noqa: PLC0415

        self.read_only = read_only
        if store.backend == "postgresql":
            if (
                not initialise
            ):  # MLflow would create its schema on first use; only `mlflow-init` may
                state = _postgres_state(store)
                if not (state.database_exists and state.schema_present):
                    if read_only:
                        raise JobNotFoundError(
                            f"no tracking store: {store.describe()} is not initialised"
                        )
                    raise TrackingError(
                        f"the tracking store {store.describe()} is not initialised: run "
                        "`uv run python scripts/db.py mlflow-init` first"
                    )
            if not read_only:
                store.artifact_path.mkdir(parents=True, exist_ok=True)
        elif read_only:
            if not store.db_path.exists():
                raise JobNotFoundError(f"no tracking store at {store.db_path}")
        else:
            store.root.mkdir(parents=True, exist_ok=True)
            store.artifact_path.mkdir(parents=True, exist_ok=True)
        self.store = store
        try:
            self.client: Any = MlflowClient(tracking_uri=store.tracking_uri)
            self._metric, self._param, self._tag = Metric, Param, RunTag
            self.experiment_id = self._experiment() if experiment else ""
        except TrackingError:
            raise
        except Exception as error:
            if store.backend == "postgresql":
                raise TrackingError(scrub(f"{type(error).__name__}: {error}")) from None
            raise

    def _experiment(self) -> str:
        found = self.client.get_experiment_by_name(EXPERIMENT)
        if found is not None:
            if found.lifecycle_stage != "active":
                raise TrackingError(f"the MLflow experiment {EXPERIMENT!r} was deleted")
            if self.store.backend == "postgresql" and not self.read_only:
                self._check_stored_location(str(found.artifact_location))
            return str(found.experiment_id)
        if self.read_only:
            raise JobNotFoundError(f"the tracking store has no {EXPERIMENT!r} experiment yet")
        if self.store.backend == "postgresql":
            check_artifact_location(self.store.artifact_uri)  # before anything is created
        return str(
            self.client.create_experiment(EXPERIMENT, artifact_location=self.store.artifact_uri)
        )

    def _check_stored_location(self, stored: str) -> None:
        """Run URIs are derived from the experiment's *stored* location, so that is what must fit;
        and writes only go to the configured location (a moved project would otherwise log into a
        path that no longer exists)."""
        check_artifact_location(stored)
        if stored.rstrip("/") != self.store.artifact_uri.rstrip("/"):
            raise ArtifactLocationError(
                f"the store's experiment records the artifact location {stored}, but the "
                f"configured location is {self.store.artifact_uri} (was the project moved, or "
                "AUTOML_MLFLOW_ARTIFACT_ROOT changed?). Nothing was written."
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
    for name, value in sorted(record.extra_tags.items()):
        if name in tags:
            raise TrackingError(f"extra tag {name!r} would replace a tag the platform sets")
        tags[name] = value
    return tags


def log_job(record: JobRecord, store: TrackingStore) -> TrackedJob:
    """Log (or resume logging) one job. Raises `AlreadyTrackedError` if it is already complete and
    `TrackingError` for any other failure; the cause is chained."""
    try:
        return _log_job(record, store)
    except TrackingError:
        raise
    except Exception as error:
        raise TrackingError(scrub(f"{type(error).__name__}: {error}")) from error


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
    if not store.exists():
        return {"runs": 0}
    api = _Api(store)
    runs = api.client.search_runs([api.experiment_id], max_results=5000)
    counts: dict[str, int] = {}
    for run in runs:
        label = f"{run.data.tags.get(T_ROLE)}:{run.data.tags.get(T_STATE)}"
        counts[label] = counts.get(label, 0) + 1
    return {"runs": len(runs), "by_role_and_state": counts}


def _create_database_if_missing(store: TrackingStore) -> bool:
    """Create the (empty) MLflow database if it does not exist. Returns True if it was created."""
    from sqlalchemy import create_engine, text  # noqa: PLC0415
    from sqlalchemy.exc import SQLAlchemyError  # noqa: PLC0415

    config = _dbconfig()
    state = _postgres_state(store)
    if state.database_exists:
        return False
    try:
        admin = create_engine(
            config.database_url().set(database="postgres"),
            isolation_level="AUTOCOMMIT",
            connect_args={"connect_timeout": config.CONNECT_TIMEOUT_SECONDS},
        )
        try:
            with admin.connect() as connection:  # the name was validated against a strict pattern
                connection.execute(text(f'CREATE DATABASE "{store.database_name}"'))
        finally:
            admin.dispose()
    except SQLAlchemyError as error:
        reason = str(getattr(error, "orig", error)).strip().splitlines()[0]
        raise TrackingError(scrub(f"cannot create the MLflow database: {reason}")) from None
    return True


def initialise_store(store: TrackingStore, *, create_experiment: bool = True) -> dict[str, Any]:
    """`mlflow-init`: create the PostgreSQL database if needed, let MLflow create its schema, and
    create the experiment with its (pre-flight checked) artifact location. Idempotent. This is the
    only code path that may create the store; reads, status checks and `log_job` never do."""
    if store.backend != "postgresql":
        raise TrackingError("mlflow-init applies to the PostgreSQL backend only")
    check_artifact_location(store.artifact_uri)  # before anything is created
    created = _create_database_if_missing(store)
    store.artifact_path.mkdir(parents=True, exist_ok=True)
    _Api(store, initialise=True, experiment=create_experiment)
    status = store_status(store)
    status["database_created"] = created
    return status


def store_status(store: TrackingStore) -> dict[str, Any]:
    """Read-only state of a store: existence, schema revision, experiment, artifact location and
    the longest run artifact URI against the PostgreSQL limit. Creates nothing."""
    info: dict[str, Any] = {"backend": store.backend, "store": store.describe()}
    if store.backend == "sqlite":
        info["initialised"] = store.exists()
        return info
    state = _postgres_state(store)
    info.update(
        database=store.database_name,
        database_exists=state.database_exists,
        schema_present=state.schema_present,
        schema_revision=state.revision,
        initialised=state.database_exists and state.schema_present,
        configured_artifact_location=store.artifact_uri,
    )
    try:
        info["configured_location_longest_run_uri"] = check_artifact_location(store.artifact_uri)
        info["configured_location_ok"] = True
    except ArtifactLocationError as error:
        info["configured_location_ok"] = False
        info["configured_location_problem"] = str(error)
    if not info["initialised"]:
        return info
    api = _Api(store, read_only=True, experiment=False)
    found = api.client.get_experiment_by_name(EXPERIMENT)
    info["experiment_exists"] = found is not None
    if found is not None:
        stored = str(found.artifact_location)
        info["experiment_artifact_location"] = stored
        info["location_matches_configuration"] = stored.rstrip("/") == store.artifact_uri.rstrip(
            "/"
        )
        runs = api.client.search_runs([found.experiment_id], max_results=MAX_RUNS)
        info["runs"] = len(runs)
        longest = max((len(str(r.info.artifact_uri)) for r in runs), default=0)
        info["longest_run_uri"] = longest
        info["run_uri_margin"] = RUN_URI_LIMIT - longest
    return info


def run_ids(job_key: str, store: TrackingStore) -> dict[str, Any]:
    """The MLflow run ids of a job (for the migration mapping). Read-only."""
    api = _Api(store, read_only=True)
    parent = api.find(job_key, "job")
    if parent is None:
        raise JobNotFoundError(f"no tracked job {job_key!r}")
    children = api.client.search_runs(
        [api.experiment_id],
        f"tags.`{T_JOB}` = '{job_key}' and tags.`{T_ROLE}` = 'algorithm'",
        max_results=500,
    )
    return {
        "parent": str(parent.info.run_id),
        "state": parent.data.tags.get(T_STATE),
        "children": {str(c.data.tags.get(T_ALGORITHM)): str(c.info.run_id) for c in children},
    }


# --------------------------------------------------------------------------------------------
# P4-6: the registry mirror (additive; nothing above this line was changed)
#
# The platform registry in PostgreSQL is authoritative. These functions keep a *mirror* of it in
# MLflow's model registry and never decide anything: they read or write exactly what the caller
# (`automl_ops.registry_mirror`) tells them to. Rules (owner decisions, D-042):
#   * only platform-owned models (tag `platform.managed=true`) are ever written; a model of the
#     same name without that tag is a conflict and is never touched;
#   * the only alias the platform writes is `production`, and only on a platform-owned version;
#   * MLflow registered models, model versions and runs are never deleted, and no other alias is
#     removed or moved;
#   * a version's source is `runs:/<run>/package`, the sealed package logged by `log_job`;
#     nothing is unpickled and no test data is read.
# Every function is idempotent: running it again after an interruption finds what is there.
# --------------------------------------------------------------------------------------------

MIRROR_MANAGED_TAG = "platform.managed"  # on the registered model
MIRROR_PACKAGE_TAG = "platform.package_id"  # on each version: the sealed package it mirrors
MIRROR_VERSION_TAG = "platform.registry_version"
MIRROR_STATUS_TAG = "platform.status"
MIRROR_SELECTION_TAG = "platform.selection_sha256"
MIRROR_ALIAS = "production"
_MIRROR_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MIRROR_PACKAGE = re.compile(r"^m-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
_MIRROR_SOURCE = re.compile(r"^runs:/([0-9a-f]{32})/package$")
ALIAS_ATTEMPTS = 4  # MLflow's alias write is a read-then-insert; concurrent writers can collide


class MirrorError(TrackingError):
    """The mirror could not be read or written (MLflow unreachable, store not ready, ...)."""


class MirrorConflict(MirrorError):
    """Something in MLflow that the platform does not own, or that is ambiguous: never modified."""


@dataclass(frozen=True)
class MirrorVersion:
    """One MLflow model version, as read. `owned` means it carries the platform's package tag."""

    mlflow_version: str
    package_id: str | None
    registry_version: str | None
    status: str | None
    selection_sha256: str | None
    source: str
    source_ok: bool  # the source's run still has its `package` artifacts

    @property
    def owned(self) -> bool:
        return self.package_id is not None


@dataclass(frozen=True)
class MirrorSnapshot:
    exists: bool
    owned: bool  # the registered model carries `platform.managed=true`
    versions: tuple[MirrorVersion, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)  # every alias -> MLflow version


def _mirror_api(store: TrackingStore, *, read_only: bool) -> _Api:
    return _Api(store, read_only=read_only, experiment=False)


def _mirror_name(name: str) -> str:
    if not _MIRROR_NAME.fullmatch(name):
        raise MirrorError(f"{name!r} is not a valid registry model name")
    return name


def _mirror_guard(work: Any) -> Any:
    """Run one MLflow interaction; anything unexpected becomes a scrubbed `MirrorError`."""
    try:
        return work()
    except TrackingError:
        raise
    except Exception as error:
        raise MirrorError(scrub(f"{type(error).__name__}: {str(error)[:300]}")) from None


def _is_missing(error: Exception) -> bool:
    return getattr(error, "error_code", "") == "RESOURCE_DOES_NOT_EXIST"


def mirror_snapshot(store: TrackingStore, name: str) -> MirrorSnapshot:
    """Read-only: the registered model `name` as MLflow has it. Creates nothing."""
    _mirror_name(name)

    def work() -> MirrorSnapshot:
        if store.backend == "sqlite" and not store.db_path.exists():
            return MirrorSnapshot(False, False)  # nothing there yet; the first write creates it
        api = _mirror_api(store, read_only=True)
        client = api.client
        try:
            model = client.get_registered_model(name)
        except Exception as error:
            if _is_missing(error):
                return MirrorSnapshot(False, False)
            raise
        owned = dict(model.tags or {}).get(MIRROR_MANAGED_TAG) == "true"
        aliases = {str(k): str(v) for k, v in dict(model.aliases or {}).items()}
        versions: list[MirrorVersion] = []
        token: str | None = None
        while True:
            page = client.search_model_versions(f"name='{name}'", max_results=200, page_token=token)
            for mv in page:
                tags = dict(mv.tags or {})
                found = _MIRROR_SOURCE.fullmatch(str(mv.source))
                source_ok = False
                if found:
                    try:
                        source_ok = bool(
                            client.list_artifacts(found.group(1), PACKAGE_ARTIFACT_DIR)
                        )
                    except Exception:
                        source_ok = False
                versions.append(
                    MirrorVersion(
                        mlflow_version=str(mv.version),
                        package_id=tags.get(MIRROR_PACKAGE_TAG),
                        registry_version=tags.get(MIRROR_VERSION_TAG),
                        status=tags.get(MIRROR_STATUS_TAG),
                        selection_sha256=tags.get(MIRROR_SELECTION_TAG),
                        source=str(mv.source),
                        source_ok=source_ok,
                    )
                )
            token = getattr(page, "token", None)
            if not token or len(versions) >= 10_000:
                break
        versions.sort(key=lambda v: int(v.mlflow_version))
        return MirrorSnapshot(True, owned, tuple(versions), aliases)

    snapshot: MirrorSnapshot = _mirror_guard(work)
    return snapshot


def mirror_ensure_model(store: TrackingStore, name: str) -> bool:
    """Create the platform-owned registered model if it is missing; True if it was created. A model
    of that name that the platform does not own raises `MirrorConflict` and is not touched."""
    _mirror_name(name)

    def work() -> bool:
        client = _mirror_api(store, read_only=False).client
        try:
            client.create_registered_model(name, tags={MIRROR_MANAGED_TAG: "true"})
            return True
        except Exception as error:
            if getattr(error, "error_code", "") != "RESOURCE_ALREADY_EXISTS":
                raise
        model = client.get_registered_model(name)
        if dict(model.tags or {}).get(MIRROR_MANAGED_TAG) != "true":
            raise MirrorConflict(
                f"MLflow already has a registered model named {name!r} that the platform does not "
                "own; it was not touched"
            )
        return False

    created: bool = _mirror_guard(work)
    return created


def mirror_package_run(store: TrackingStore, models_root: Path, package_id: str) -> str:
    """The MLflow run that holds this sealed package's files, tracking the package first if it is
    not tracked yet (the existing resumable `log_job`; seal hashes are checked, nothing is
    unpickled). Returns the run id of the algorithm run that carries the `package` artifacts."""
    if not _MIRROR_PACKAGE.fullmatch(package_id):
        raise MirrorError(f"{package_id!r} is not a package id")

    def tracked() -> dict[str, Any] | None:
        try:
            found = run_ids(package_id, store)
        except JobNotFoundError:
            return None
        return found if found["state"] == COMPLETE else None

    def work() -> str:
        found = tracked()
        if found is None:
            try:
                log_job(record_from_package(models_root, package_id), store)
            except AlreadyTrackedError:
                pass
            found = tracked()
            if found is None:
                raise MirrorError(f"package {package_id} could not be tracked completely")
        client = _mirror_api(store, read_only=True).client
        holders = [
            run
            for run in found["children"].values()
            if any(a.path == PACKAGE_ARTIFACT_DIR for a in client.list_artifacts(run))
        ]
        if len(holders) != 1:
            raise MirrorError(
                f"package {package_id}: expected one tracked run holding its files, found "
                f"{len(holders)}"
            )
        return str(holders[0])

    run: str = _mirror_guard(work)
    return run


def mirror_ensure_version(
    store: TrackingStore,
    name: str,
    *,
    package_id: str,
    run_id: str,
    registry_version: int,
    status: str,
    selection_sha256: str,
) -> tuple[str, bool]:
    """The MLflow version that mirrors `package_id`, created if there is none. Found by its package
    tag, so a retry after an interruption never creates a second one. Two or more versions with
    that tag, or one whose source is not the expected run, raise `MirrorConflict` (nothing is
    deleted)."""
    _mirror_name(name)
    if not _MIRROR_PACKAGE.fullmatch(package_id):
        raise MirrorError(f"{package_id!r} is not a package id")
    source = f"runs:/{run_id}/{PACKAGE_ARTIFACT_DIR}"

    def work() -> tuple[str, bool]:
        client = _mirror_api(store, read_only=False).client
        found = list(
            client.search_model_versions(
                f"name='{name}' and tag.`{MIRROR_PACKAGE_TAG}`='{package_id}'"
            )
        )
        if len(found) > 1:
            raise MirrorConflict(
                f"MLflow has {len(found)} versions of {name!r} for package {package_id}; "
                "none was changed or deleted"
            )
        if found:
            if str(found[0].source) != source:
                raise MirrorConflict(
                    f"the MLflow version for package {package_id} has an unexpected source; "
                    "it was not changed"
                )
            return str(found[0].version), False
        created = client.create_model_version(
            name,
            source,
            run_id=run_id,
            tags={  # given at creation: the version is never visible without its package tag
                MIRROR_PACKAGE_TAG: package_id,
                MIRROR_VERSION_TAG: str(registry_version),
                MIRROR_STATUS_TAG: status,
                MIRROR_SELECTION_TAG: selection_sha256,
            },
        )
        return str(created.version), True

    result: tuple[str, bool] = _mirror_guard(work)
    return result


def mirror_set_tags(
    store: TrackingStore, name: str, mlflow_version: str, tags: dict[str, str]
) -> list[str]:
    """Set the given tags on a platform version where they differ; returns the keys changed."""
    _mirror_name(name)
    allowed = {MIRROR_VERSION_TAG, MIRROR_STATUS_TAG, MIRROR_SELECTION_TAG}
    if not set(tags) <= allowed:
        raise MirrorError("only the platform's own version tags can be set")

    def work() -> list[str]:
        client = _mirror_api(store, read_only=False).client
        current = dict(client.get_model_version(name, mlflow_version).tags or {})
        if MIRROR_PACKAGE_TAG not in current:
            raise MirrorConflict(f"version {mlflow_version} of {name!r} is not a platform version")
        changed = []
        for key, value in tags.items():
            if current.get(key) != value:
                client.set_model_version_tag(name, mlflow_version, key, value)
                changed.append(key)
        return changed

    result: list[str] = _mirror_guard(work)
    return result


def mirror_set_production_alias(
    store: TrackingStore, name: str, mlflow_version: str | None, *, current: str | None = None
) -> bool:
    """Point the `production` alias at `mlflow_version`, or remove it when None. This is the only
    alias the platform writes. `current` is the holder the caller last read: if the alias has since
    moved to a version the caller did not expect, nothing is changed (`MirrorConflict`). Returns
    True if something changed. MLflow's alias write can collide with a concurrent writer, so it is
    retried a few times and then read back."""
    _mirror_name(name)

    def holder(client: Any) -> str | None:
        value = dict(client.get_registered_model(name).aliases or {}).get(MIRROR_ALIAS)
        return None if value is None else str(value)

    def work() -> bool:
        client = _mirror_api(store, read_only=False).client
        now = holder(client)
        if now == mlflow_version:
            return False
        if now != current and current is not None:
            raise MirrorConflict(
                f"the {MIRROR_ALIAS!r} alias of {name!r} moved while it was being updated; "
                "it was not changed"
            )
        last: Exception | None = None
        for attempt in range(ALIAS_ATTEMPTS):
            try:
                if mlflow_version is None:
                    client.delete_registered_model_alias(name, MIRROR_ALIAS)
                else:
                    client.set_registered_model_alias(name, MIRROR_ALIAS, mlflow_version)
                last = None
                break
            except Exception as error:
                last = error
                time.sleep(0.1 * (attempt + 1))
        if holder(client) != mlflow_version:
            raise MirrorError(
                f"the {MIRROR_ALIAS!r} alias of {name!r} could not be set"
                + (f" ({type(last).__name__})" if last else "")
            )
        return True

    result: bool = _mirror_guard(work)
    return result
