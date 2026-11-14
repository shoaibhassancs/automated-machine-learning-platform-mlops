"""The backend lifecycle against a running stack (P4-10, backend half of BUILD_PLAN Step 35).

    uv run python scripts/e2e_demo.py abalone

Needs `make db-up` with the migration head applied, then `make api`, `make serve` and `make worker`
running (the worker with tracking configured: AUTOML_JOBS_MLFLOW_DIR or the PostgreSQL backend), and
`make demo-data` done. It uploads the dataset's *training pool*, trains and tracks a final job,
registers, promotes and deploys the model, then replays the monitoring stream (normal, shifted,
normal) as simulated traffic with feedback, monitoring after each phase, and reports every stage as
passed or failed with what was observed.

**It writes real rows to the database the stack uses** (a dataset version, a job, a model package,
a model version, a deployment, predictions, labels, reports and alerts); nothing is ever deleted.
Prints JSON; exit 0 = every stage passed, 1 = a stage failed (the story stops there), 2 = could not
start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from automl_ops.e2e import run_story
from automl_ops.simulator import LocalHttp, SimulatorError, load_stream

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"


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
    health = control.request("GET", "/health")
    if health.status != 200:
        print("ERROR: the control-plane API is not answering (make api)", file=sys.stderr)
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
