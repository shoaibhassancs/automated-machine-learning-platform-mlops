"""The command-line side of the end-to-end story, shared by `scripts/e2e_demo.py` (a throwaway
stack made by `scripts/e2e_isolated.py`) and `scripts/e2e_compose.py` (the full Compose stack).

The two scripts differ only in how they state what stack they expect (`local`, an `Isolation`);
everything after that is the same: `verify_stack` must find both services isolated, on that database
and run, and the control plane empty, before any write. There is no flag to skip the check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from automl_ops.e2e import run_story
from automl_ops.isolation import Isolation, verify_stack
from automl_ops.simulator import LocalHttp, SimulatorError, load_stream

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "data" / "demo"


def build_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dataset", choices=["abalone", "adult"])
    parser.add_argument("--api-port", type=int)
    parser.add_argument("--serve-port", type=int)
    parser.add_argument("--rows", type=int, default=1000, help="rows per traffic phase")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the job")
    parser.add_argument("--model", help="registered model name (default: <dataset>-e2e)")
    return parser


def execute(
    args: argparse.Namespace,
    local: Isolation,
    api_port: int,
    serve_port: int,
    refused_hint: str,
) -> int:
    """Verify the stack, then run the story. Exit 0 passed, 1 a stage failed, 2 refused."""
    try:
        info = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))["datasets"][
            args.dataset
        ]
        pool = DEMO_DIR / f"{args.dataset}_train_pool.csv"
        normal = load_stream(DEMO_DIR / f"{args.dataset}_monitor_stream.csv")
        shifted = load_stream(DEMO_DIR / f"{args.dataset}_monitor_shifted.csv")
        data = pool.read_bytes()
        control = LocalHttp(args.api_port or api_port)
        serving = LocalHttp(args.serve_port or serve_port)
    except (OSError, ValueError, KeyError, SimulatorError) as error:
        print(f"ERROR: cannot start ({error}); run `make demo-data` first", file=sys.stderr)
        return 2
    problems = verify_stack(local, control, serving)
    if problems:
        print(
            f"REFUSED: this is not an isolated, empty stack made for the end-to-end story. "
            f"Nothing was written. {refused_hint}",
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
