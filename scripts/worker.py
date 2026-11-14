"""Run the job worker (P4-4): `uv run python scripts/worker.py` or `make worker`.

One worker at a time: a second one refuses to start. It needs PostgreSQL at the migration head, like
the API. `--once` processes the queued jobs and exits (tests, demos). Ctrl-C or SIGTERM stops the
running job at its next fold boundary (a job already in its final test finishes first); a second
signal exits at once and the job is then marked failed (`worker_lost`) by the next worker.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automl_ops.api_control.app import StartupError, require_current_schema  # noqa: E402
from automl_ops.dataset_store import FileDatasetStore  # noqa: E402
from automl_ops.db import migrate  # noqa: E402
from automl_ops.db.config import make_engine, setting  # noqa: E402
from automl_ops.ingest import DEFAULT_LIMITS  # noqa: E402
from automl_ops.jobs import WorkerLockBusy  # noqa: E402
from automl_ops.worker import (  # noqa: E402
    HEARTBEAT_SECONDS,
    POLL_SECONDS,
    Worker,
    WorkerSettings,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AutoML Ops job worker.")
    parser.add_argument("--store", type=Path, help="dataset store (default data/store)")
    parser.add_argument("--models", type=Path, help="model packages (default data/models)")
    parser.add_argument("--poll", type=float, default=POLL_SECONDS, help="seconds between polls")
    parser.add_argument("--heartbeat", type=float, default=HEARTBEAT_SECONDS)
    parser.add_argument("--once", action="store_true", help="exit when the queue is empty")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    store_root = args.store or Path(setting("AUTOML_STORE_DIR") or PROJECT_ROOT / "data" / "store")
    models_root = args.models or Path(
        setting("AUTOML_MODELS_DIR") or PROJECT_ROOT / "data" / "models"
    )
    engine = make_engine()
    try:
        require_current_schema(lambda: migrate.status(engine))
    except StartupError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    worker = Worker(
        engine,
        FileDatasetStore(store_root, DEFAULT_LIMITS),
        models_root,
        settings=WorkerSettings(poll_seconds=args.poll, heartbeat_seconds=args.heartbeat),
    )
    signalled = False

    def stop(signum: int, _frame: object) -> None:
        nonlocal signalled
        if signalled:
            raise SystemExit(130)  # second signal: leave now; recovery handles the job
        signalled = True
        worker.request_stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        done = worker.run(once=args.once)
    except WorkerLockBusy as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 3
    logging.getLogger("automl_ops.worker").info("worker finished after %d job(s)", done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
