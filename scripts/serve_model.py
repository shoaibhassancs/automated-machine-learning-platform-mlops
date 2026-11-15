"""Start the serving app on the loopback interface (P4-7).

    uv run python scripts/serve_model.py [--port 8200]

Serves the one deployed model version from PostgreSQL's active deployment (see
`docs/serving.md`). Refuses to bind to anything but 127.0.0.1, ::1 or localhost, and refuses to
start unless the application database is reachable and at the migration head. Settings:
AUTOML_SERVING_PORT, AUTOML_MODELS_DIR, AUTOML_UI_ORIGIN (environment or the git-ignored .env).
No authentication: local, single-user only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import uvicorn

from automl_ops.api_control.app import StartupError, require_current_schema
from automl_ops.api_control.settings import SettingsError
from automl_ops.api_serving.app import build_app
from automl_ops.api_serving.settings import ServingSettings
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
    parser.add_argument("--port", type=int, help="default: AUTOML_SERVING_PORT or 8200")
    args = parser.parse_args()
    problem = host_problem(args.host)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 2
    try:
        settings = ServingSettings.from_environment()
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
