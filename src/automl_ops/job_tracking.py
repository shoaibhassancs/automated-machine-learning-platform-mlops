"""Which MLflow store a job may log to (P4-4, D6): tracking is opt-in and never the archive.

A job logs only when its request says `track: true`. Then:

* **PostgreSQL** (`AUTOML_MLFLOW_BACKEND=postgresql`): the MLflow database on the platform server.
* **SQLite**: only in a directory named by `AUTOML_JOBS_MLFLOW_DIR` that is not the original
  Phase 3 archive `data/mlflow/` (nor inside it). Without that setting a SQLite job is refused.

The global default of `TrackingStore` (SQLite in `data/mlflow`) is not changed and not used here.
"""

from __future__ import annotations

import os
from pathlib import Path

from automl_ops.db.config import setting
from automl_ops.tracking import DEFAULT_ROOT, TrackingStore

JOBS_DIR_SETTING = "AUTOML_JOBS_MLFLOW_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TrackingNotAllowed(Exception):
    """Tracking is not configured for jobs, or points at the archive."""


def archive_root() -> Path:
    return (PROJECT_ROOT / DEFAULT_ROOT).resolve()


def _is_archive(path: Path, archive: Path) -> bool:
    candidate = path.resolve()
    if candidate == archive or archive in candidate.parents:
        return True
    if candidate.as_posix().casefold() == archive.as_posix().casefold():  # case-insensitive disks
        return True
    try:
        if candidate.exists() and archive.exists() and os.path.samefile(candidate, archive):
            return True
        marker = candidate / "mlflow.db"
        original = archive / "mlflow.db"
        return marker.exists() and original.exists() and os.path.samefile(marker, original)
    except OSError:
        return True  # cannot tell: refuse


def job_tracking_store(env: dict[str, str] | None = None) -> TrackingStore:
    """The store a tracked job logs to, or `TrackingNotAllowed` with the reason."""
    backend = setting("AUTOML_MLFLOW_BACKEND", "sqlite", env=env) or "sqlite"
    if backend not in ("sqlite", "postgresql"):
        raise TrackingNotAllowed(f"unknown tracking backend {backend!r}; use sqlite or postgresql")
    if backend == "postgresql":
        return TrackingStore.from_settings(None, "postgresql", env)
    directory = setting(JOBS_DIR_SETTING, None, env=env)
    if not directory:
        raise TrackingNotAllowed(
            f"tracking is not configured for jobs: set {JOBS_DIR_SETTING} to a separate "
            "directory, or select the PostgreSQL backend (AUTOML_MLFLOW_BACKEND=postgresql)"
        )
    root = Path(directory).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if _is_archive(root, archive_root()):
        raise TrackingNotAllowed(
            "jobs never write to the original tracking archive data/mlflow; "
            f"choose another directory for {JOBS_DIR_SETTING}"
        )
    return TrackingStore(root.resolve(), "sqlite")


def tracking_problem(env: dict[str, str] | None = None) -> str | None:
    """None if a tracked job could run now, otherwise the reason it could not."""
    try:
        job_tracking_store(env)
    except Exception as error:
        return str(error)
    return None
