"""A throwaway, optionally seeded backend for the frontend (Phase 5).

Used by `make ui-demo` and the browser tests.

    uv run python scripts/ui_stack.py --seed demo --with-ui          # try the UI on demo data
    uv run python scripts/ui_stack.py --seed none --api-port 18100 --serve-port 18200 \\
        --ui-origin http://localhost:5301 --ready-file /tmp/stack.json   # used by Playwright

Everything is created by `automl_ops.e2e_stack.IsolatedStack` (new database `automl_test_e2e_*`,
temporary folders outside the project, SQLite MLflow store, free or given local ports) and removed
on Ctrl-C or SIGTERM after ownership is proved; the real `automl_ops` / `mlflow` databases and
`data/` are never used. `--seed demo` runs the end-to-end story on the Abalone demo data and then
leaves one open drift alert and one resolved alert, so every Monitoring screen has data.
`--with-ui` also starts the Vite dev server on the given UI origin.

Exit codes: 0 stopped cleanly, 2 could not start, 3 cleanup left resources behind.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automl_ops import simulator as sim  # noqa: E402
from automl_ops.db.config import DatabaseConfigError  # noqa: E402
from automl_ops.e2e import run_story  # noqa: E402
from automl_ops.e2e_stack import IsolatedStack, StackError  # noqa: E402

DEMO_DIR = PROJECT_ROOT / "data" / "demo"


def seed_demo(stack: IsolatedStack) -> None:
    info = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))["datasets"][
        "abalone"
    ]
    pool = DEMO_DIR / "abalone_train_pool.csv"
    normal = sim.load_stream(DEMO_DIR / "abalone_monitor_stream.csv")
    shifted = sim.load_stream(DEMO_DIR / "abalone_monitor_shifted.csv")
    control, serving = sim.LocalHttp(stack.api_port), sim.LocalHttp(stack.serve_port)
    report = run_story(
        control,
        serving,
        train_csv=(pool.name, pool.read_bytes()),
        dataset="abalone-pool",
        target=info["target"],
        task=info["task"],
        model="abalone",
        normal=normal,
        shifted=shifted,
        rows=1000,
        seed=0,
        timeout=900.0,
    )
    if not report.passed:
        raise StackError(f"the seeding story failed: {json.dumps(report.to_dict())[:400]}")
    active = control.request("GET", "/deployments/active")
    deployment = int((active.body or {})["id"])
    # a second, open drift episode so the alert screens show a live alert and a resolved one
    sim.run(shifted, serving, control, rows=400, seed=5, feedback_fraction=0.3)
    control.request("POST", f"/deployments/{deployment}/monitor", {"scope": "simulated"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--seed", choices=["none", "demo"], default="none")
    parser.add_argument("--api-port", type=int, default=0)
    parser.add_argument("--serve-port", type=int, default=0)
    parser.add_argument("--ui-origin", default="http://localhost:5173")
    parser.add_argument("--ready-file", type=Path, help="write the ports here when ready (JSON)")
    parser.add_argument("--with-ui", action="store_true", help="also start the Vite dev server")
    args = parser.parse_args()
    origin = urlparse(args.ui_origin)
    stack = IsolatedStack(
        api_port=args.api_port, serve_port=args.serve_port, ui_origin=args.ui_origin
    )
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    ui: subprocess.Popen[bytes] | None = None
    code = 0
    try:
        stack.create()
        stack.start()
        if args.seed == "demo":
            print("seeding the demo story (about 30 seconds)...", file=sys.stderr)
            seed_demo(stack)
        urls = {
            "api": f"http://127.0.0.1:{stack.api_port}",
            "serving": f"http://127.0.0.1:{stack.serve_port}",
            "ui": args.ui_origin,
        }
        if args.with_ui:
            env = {
                **os.environ,
                "VITE_API_URL": urls["api"],
                "VITE_SERVING_URL": urls["serving"],
            }
            ui = subprocess.Popen(
                ["npm", "--prefix", "frontend", "run", "dev", "--", "--port",
                 str(origin.port or 5173), "--strictPort"],
                cwd=PROJECT_ROOT,
                env=env,
            )  # fmt: skip
        if args.ready_file:
            args.ready_file.write_text(json.dumps(urls), encoding="utf-8")
        print(json.dumps(urls), flush=True)
        print("throwaway stack ready; Ctrl-C removes it", file=sys.stderr)
        stop.wait()
    except (StackError, DatabaseConfigError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        code = 2
    finally:
        if ui is not None and ui.poll() is None:
            ui.terminate()
            try:
                ui.wait(10)
            except subprocess.TimeoutExpired:
                ui.kill()
        left = stack.cleanup()
        if left:
            print("CLEANUP INCOMPLETE: left in place:", file=sys.stderr)
            for item in left:
                print(f"  - {item}", file=sys.stderr)
            code = 3
        else:
            print("throwaway stack removed", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
