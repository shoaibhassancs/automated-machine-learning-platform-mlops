"""The backend lifecycle against an isolated stack (P4-10, backend half of BUILD_PLAN Step 35).

    make e2e        # creates, uses and removes a throwaway stack (recommended)
    uv run python scripts/e2e_demo.py abalone   # only a stack made by scripts/e2e_isolated.py

It uploads the dataset's *training pool*, trains and tracks a final job, registers, promotes and
deploys the model, then replays the monitoring stream (normal, shifted, normal) as simulated
traffic with feedback, monitoring after each phase, and reports every stage as passed or failed.

**It writes rows and model packages, so it refuses to run against anything but a stack made for
it.** Before any write it checks (read-only) that this script's own settings and both services
report a throwaway database, throwaway store, models and MLflow folders outside `data/`, the same
database and run, and an empty stack. There is no flag to skip the check.

Prints JSON; exit 0 = every stage passed, 1 = a stage failed (the story stops there), 2 = refused
or could not start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from automl_ops.db.config import DatabaseConfigError, database_url, setting
from automl_ops.e2e import run_story
from automl_ops.isolation import Wiring, assess, verify_stack
from automl_ops.simulator import LocalHttp, SimulatorError, load_stream

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "data" / "demo"


def _port(value: int | None, env: str, default: int) -> int:
    if value is not None:
        return value
    raw = os.environ.get(env)
    return int(raw) if raw and raw.isdigit() else default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("dataset", choices=["abalone", "adult"])
    parser.add_argument("--api-port", type=int)
    parser.add_argument("--serve-port", type=int)
    parser.add_argument("--rows", type=int, default=1000, help="rows per traffic phase")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the job")
    parser.add_argument("--model", help="registered model name (default: <dataset>-e2e)")
    args = parser.parse_args()
    manifest_path = DEMO_DIR / "manifest.json"
    try:
        info = json.loads(manifest_path.read_text(encoding="utf-8"))["datasets"][args.dataset]
        pool = DEMO_DIR / f"{args.dataset}_train_pool.csv"
        normal = load_stream(DEMO_DIR / f"{args.dataset}_monitor_stream.csv")
        shifted = load_stream(DEMO_DIR / f"{args.dataset}_monitor_shifted.csv")
        data = pool.read_bytes()
        control = LocalHttp(_port(args.api_port, "AUTOML_API_PORT", 8100))
        serving = LocalHttp(_port(args.serve_port, "AUTOML_SERVING_PORT", 8200))
    except (OSError, ValueError, KeyError, SimulatorError) as error:
        print(f"ERROR: cannot start ({error}); run `make demo-data` first", file=sys.stderr)
        return 2
    try:
        local = assess(
            Wiring(
                database_url(),
                Path(setting("AUTOML_STORE_DIR") or PROJECT_ROOT / "data" / "store"),
                Path(setting("AUTOML_MODELS_DIR") or PROJECT_ROOT / "data" / "models"),
            )
        )
    except DatabaseConfigError as error:
        print(f"ERROR: refusing to run: {error}", file=sys.stderr)
        return 2
    problems = verify_stack(local, control, serving)
    if problems:
        print(
            "REFUSED: this is not an isolated, empty stack made for the end-to-end story. "
            "Nothing was written. Use `make e2e`.",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    report = run_story(
        control,
        serving,
        train_csv=(pool.name, data),
        dataset=f"{args.dataset}-pool",
        target=info["target"],
        task=info["task"],
        model=args.model or f"{args.dataset}-e2e",
        normal=normal,
        shifted=shifted,
        rows=args.rows,
        seed=args.seed,
        timeout=args.timeout,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
