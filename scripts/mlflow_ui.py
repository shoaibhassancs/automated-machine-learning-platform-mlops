"""Start the MLflow UI on localhost for the platform's tracking store (BUILD_PLAN Step 13).

    uv run python scripts/mlflow_ui.py [--root data/mlflow] [--port 5001] [--print-command]

Runs `mlflow server` (the UI plus REST API) in the foreground with **explicit** SQLite and artifact
locations, bound to 127.0.0.1 only, with MLflow telemetry disabled in the server's environment.
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

from automl_ops.tracking import DEFAULT_ROOT, DEFAULT_UI_PORT, TrackingStore, quiet_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="tracking store folder")
    parser.add_argument("--port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--print-command", action="store_true", help="print the command and exit")
    args = parser.parse_args(argv)
    store = TrackingStore(args.root)
    command = [sys.executable, "-m", "mlflow", *store.ui_command(args.port)[1:]]
    if args.print_command:
        print(" ".join(f"{k}={v}" for k, v in quiet_environment().items()), shlex.join(command))
        return 0
    store.root.mkdir(parents=True, exist_ok=True)
    store.artifact_path.mkdir(parents=True, exist_ok=True)
    print(f"MLflow UI on http://127.0.0.1:{args.port}  (store {store.db_path}; Ctrl-C to stop)")
    try:
        return subprocess.call(command, env={**os.environ, **quiet_environment()})
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
