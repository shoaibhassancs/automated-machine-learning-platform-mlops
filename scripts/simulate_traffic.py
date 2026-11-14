"""Replay the demo monitoring stream as simulated traffic (P4-10, BUILD_PLAN Step 25).

    uv run python scripts/simulate_traffic.py run data/demo/abalone_monitor_stream.csv --rows 200
    uv run python scripts/simulate_traffic.py run data/demo/abalone_monitor_shifted.csv --rows 500
    uv run python scripts/simulate_traffic.py demo abalone

Needs `make api` and `make serve` running, a deployed model, and (for `demo`) nothing else: the demo
creates the version's reference sample if it has none. Only the known demo monitoring files are
accepted, never a registered training dataset; every request is marked simulated; only 127.0.0.1 is
contacted. Ports: AUTOML_API_PORT (8100) and AUTOML_SERVING_PORT (8200), or --api-port/--serve-port.
Prints a JSON report; exit 0 = clean run, 1 = the run had failures, 2 = refused or cannot start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from automl_ops.simulator import LocalHttp, SimulatorError, demo, load_stream, run

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"


def _port(value: int | None, env: str, default: int) -> int:
    if value is not None:
        return value
    raw = os.environ.get(env)
    return int(raw) if raw and raw.isdigit() else default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--api-port", type=int)
    parser.add_argument("--serve-port", type=int)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="send one stream file as simulated traffic")
    one.add_argument("file", type=Path)
    one.add_argument("--rows", type=int, default=200)
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--batch-size", type=int, default=50)
    one.add_argument("--rate", type=float, help="rows per second (default: as fast as possible)")
    one.add_argument("--feedback", type=float, default=0.0, help="share of rows to label, 0 to 1")
    show = sub.add_parser("demo", help="normal, then shifted, then normal traffic, monitored")
    show.add_argument("dataset", choices=["abalone", "adult"])
    show.add_argument("--rows", type=int, default=1000)
    show.add_argument("--seed", type=int, default=0)
    show.add_argument("--feedback", type=float, default=0.3)
    args = parser.parse_args()
    try:
        control = LocalHttp(_port(args.api_port, "AUTOML_API_PORT", 8100))
        serving = LocalHttp(_port(args.serve_port, "AUTOML_SERVING_PORT", 8200))
        if args.command == "run":
            result = run(
                load_stream(args.file),
                serving,
                control,
                rows=args.rows,
                seed=args.seed,
                batch_size=args.batch_size,
                rate=args.rate,
                feedback_fraction=args.feedback,
            )
            print(json.dumps(result.to_dict(), indent=2))
            return 0 if result.clean else 1
        pair = (
            load_stream(DEMO_DIR / f"{args.dataset}_monitor_stream.csv"),
            load_stream(DEMO_DIR / f"{args.dataset}_monitor_shifted.csv"),
        )
        phases = demo(
            pair,
            serving,
            control,
            rows=args.rows,
            seed=args.seed,
            feedback_fraction=args.feedback,
        )
        print(
            json.dumps(
                [
                    {
                        "phase": p.name,
                        "traffic": p.traffic.to_dict(),
                        "monitoring": p.monitoring,
                        "alerts": p.alerts,
                    }
                    for p in phases
                ],
                indent=2,
            )
        )
        return 0 if all(p.traffic.clean for p in phases) else 1
    except SimulatorError as error:
        print(f"REFUSED ({error.code}): {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
