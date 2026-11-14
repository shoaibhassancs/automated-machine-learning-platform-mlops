"""Start the MLflow UI on localhost for the platform's tracking store (BUILD_PLAN Step 13).

    uv run python scripts/mlflow_ui.py [--root data/mlflow] [--backend postgresql] [--port 5001]
        [--print-command]

Runs `mlflow server` (the UI plus REST API) in the foreground with **explicit** backend and artifact
locations, bound to 127.0.0.1 only, with MLflow telemetry disabled in the server's environment.
For the PostgreSQL backend the backend URI (which holds the password) and the artifact root are
passed in the server's *environment* (MLFLOW_BACKEND_STORE_URI, MLFLOW_DEFAULT_ARTIFACT_ROOT), never
as command-line arguments, so the password does not appear in the process list.
Stop it with Ctrl-C. It is never started automatically. Port 5001 is the default because macOS
AirPlay Receiver can occupy 5000.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from automl_ops.tracking import (
    DEFAULT_ROOT,
    DEFAULT_UI_PORT,
    TrackingError,
    TrackingStore,
    quiet_environment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="SQLite store folder")
    parser.add_argument(
        "--backend",
        choices=("sqlite", "postgresql"),
        default=None,
        help="default: AUTOML_MLFLOW_BACKEND (environment or .env), else sqlite",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--print-command", action="store_true", help="print the command and exit")
    args = parser.parse_args(argv)
    try:
        store = TrackingStore.from_settings(args.root, args.backend)
        command = [sys.executable, "-m", "mlflow", *store.ui_command(args.port)[1:]]
        extra = store.ui_environment()
        if args.print_command:
            shown = " ".join(f"{k}={v}" for k, v in quiet_environment().items())
            hidden = " ".join(
                f"{k}=<{'set, not shown' if v else 'empty'}>" for k, v in extra.items()
            )
            print(shown, hidden, shlex.join(command))
            return 0
        if store.backend == "postgresql":
            if not store.exists():
                print(
                    f"ERROR: {store.describe()} is not initialised: run scripts/db.py mlflow-init"
                )
                return 2
        else:
            store.root.mkdir(parents=True, exist_ok=True)
        store.artifact_path.mkdir(parents=True, exist_ok=True)
        print(
            f"MLflow UI on http://127.0.0.1:{args.port}  (store {store.describe()}; Ctrl-C to stop)"
        )
        return subprocess.call(command, env={**os.environ, **quiet_environment(), **extra})
    except TrackingError as error:
        print(f"ERROR: {error}")
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
