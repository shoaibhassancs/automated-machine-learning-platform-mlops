"""The backend lifecycle against the full Compose stack (Phase 6, Step 35).

    make full-up && make e2e-compose

Runs the same story as `make e2e` (upload the training pool, train, register, promote, deploy,
replay simulated traffic with feedback, monitor), but against the running `compose.full.yaml` stack
on its own ports (API 28100, serving 28200). It still refuses, before any write, unless both
services report isolation, on the database and run the Compose file declares (the run token is the
one in the git-ignored `.env.full`), and the stack is empty. It never uses the development database.

Prints JSON; exit 0 = every stage passed, 1 = a stage failed, 2 = refused or could not start.
"""

from __future__ import annotations

import sys
from pathlib import Path

from automl_ops.db.config import read_env_file
from automl_ops.e2e_cli import PROJECT_ROOT, build_parser, execute
from automl_ops.isolation import compose_expectation

API_PORT = 28100
SERVING_PORT = 28200


def main() -> int:
    args = build_parser(__doc__.splitlines()[0] if __doc__ else None).parse_args()
    env_file = Path(PROJECT_ROOT, ".env.full")
    try:
        token = read_env_file(env_file).get("AUTOML_FULL_E2E_TOKEN", "")
    except OSError:
        token = ""
    if not token:
        print("ERROR: .env.full has no run token; run `make full-env` first", file=sys.stderr)
        return 2
    return execute(
        args,
        compose_expectation(token),
        API_PORT,
        SERVING_PORT,
        "Start the stack with `make full-up` (it must be empty).",
    )


if __name__ == "__main__":
    raise SystemExit(main())
