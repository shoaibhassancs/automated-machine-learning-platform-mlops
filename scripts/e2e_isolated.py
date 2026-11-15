"""Run the end-to-end story on a throwaway stack made just for it: `make e2e`.

    uv run python scripts/e2e_isolated.py abalone|adult [--rows N] [--seed N] [--timeout S]

Creates a new database `automl_test_e2e_<random>` (migrated), a temporary folder outside the project
for the dataset store, model packages and a SQLite MLflow store, and starts the API, serving app and
worker on free local ports. `scripts/e2e_demo.py` then verifies the isolation before it writes
anything. Afterwards it stops the processes and removes only what this run created, after proving
ownership (database comment, owner marker, location). If ownership cannot be proved, nothing is
deleted and the remaining resources are listed. The real `automl_ops` and `mlflow` databases and
everything under `data/` are never connected to or changed.

Exit codes: the story's (0 passed, 1 a stage failed, 2 refused or could not start), or 3 if
cleanup left resources behind.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automl_ops.db.config import DatabaseConfigError  # noqa: E402
from automl_ops.e2e_stack import IsolatedStack, StackError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("dataset", choices=["abalone", "adult"])
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    stack = IsolatedStack()
    code = 2
    try:
        stack.create()
        print(f"isolated stack created (database {stack.database})", file=sys.stderr)
        stack.start()
        done = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "e2e_demo.py"),
                args.dataset,
                "--api-port",
                str(stack.api_port),
                "--serve-port",
                str(stack.serve_port),
                "--rows",
                str(args.rows),
                "--seed",
                str(args.seed),
                "--timeout",
                str(args.timeout),
            ],
            env=stack.environment(),
            cwd=PROJECT_ROOT,
        )
        code = done.returncode
    except (StackError, DatabaseConfigError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    finally:
        left = stack.cleanup()
        if left:
            print("CLEANUP INCOMPLETE: these resources were left in place:", file=sys.stderr)
            for item in left:
                print(f"  - {item}", file=sys.stderr)
            code = 3
        else:
            print("isolated stack removed", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
