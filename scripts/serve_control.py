"""Start the control-plane API on the loopback interface (P4-3).

    uv run python scripts/serve_control.py [--port 8100]

Refuses to bind to anything but 127.0.0.1, ::1 or localhost, and refuses to start unless the
application database is reachable and at the migration head (run `scripts/db.py upgrade` first;
migrations never run on their own). Settings: AUTOML_API_PORT, AUTOML_UI_ORIGIN, AUTOML_STORE_DIR
(environment or the git-ignored .env). No authentication: local, single-user only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import uvicorn

from automl_ops.api_control.app import StartupError, build_app, require_current_schema
from automl_ops.api_control.settings import ApiSettings, SettingsError
from automl_ops.bind import host_problem
from automl_ops.db import migrate
from automl_ops.db.config import DatabaseConfigError, make_engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="loopback only (0.0.0.0 only inside the project's image)",
    )
    parser.add_argument("--port", type=int, help="default: AUTOML_API_PORT or 8100")
    args = parser.parse_args()
    problem = host_problem(args.host)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 2
    try:
        settings = ApiSettings.from_environment()
        if args.port is not None:
            settings = replace(settings, port=args.port)
            settings.__post_init__()
        engine = make_engine()
        require_current_schema(lambda: migrate.status(engine))
        engine.dispose()
        app = build_app(settings)
    except (SettingsError, DatabaseConfigError, StartupError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    uvicorn.run(
        app,
        host=args.host,
        port=settings.port,
        proxy_headers=False,
        server_header=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
