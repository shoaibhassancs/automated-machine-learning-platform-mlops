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

import os
import sys
from pathlib import Path

from automl_ops.db.config import DatabaseConfigError, database_url, setting
from automl_ops.e2e_cli import PROJECT_ROOT, build_parser, execute
from automl_ops.isolation import Wiring, assess


def _port(env: str, default: int) -> int:
    raw = os.environ.get(env)
    return int(raw) if raw and raw.isdigit() else default


def main() -> int:
    args = build_parser(__doc__.splitlines()[0] if __doc__ else None).parse_args()
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
    return execute(
        args,
        local,
        _port("AUTOML_API_PORT", 8100),
        _port("AUTOML_SERVING_PORT", 8200),
        "Use `make e2e`.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
