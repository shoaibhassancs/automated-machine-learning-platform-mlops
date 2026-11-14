"""Application database commands (BUILD_PLAN Step 17, chunk P4-1).

    uv run python scripts/db.py upgrade            # apply migrations (explicit; nothing else does)
    uv run python scripts/db.py status [--json]    # revision applied, revision expected, pending
    uv run python scripts/db.py check [--json]     # at head, and structure equals the models
    uv run python scripts/db.py sync [--store data/store] [--json]   # catalog the file store
    uv run python scripts/db.py mlflow-init [--json]    # P4-2b: create MLflow's database + schema
    uv run python scripts/db.py mlflow-status [--json]  # P4-2b: read-only; never creates anything

`mlflow-init` and `mlflow-status` concern MLflow's separate `mlflow` database on the same server
(MLflow owns its schema). Only `mlflow-init` may create the database, its schema and the experiment
(idempotent); the artifact location is checked against PostgreSQL's 200-character run-URI limit
before anything is created. `mlflow-status` exits 0 if the store is initialised and consistent, 1
otherwise. Settings: AUTOML_MLFLOW_DB_NAME (default `mlflow`), AUTOML_MLFLOW_ARTIFACT_ROOT (default
`data/mlflow-pg/artifacts`), plus the same credentials as the application database.

Start the server first: `make db-up` (Docker Compose, PostgreSQL 17 on 127.0.0.1:55432). The
password comes from POSTGRES_PASSWORD (environment or the git-ignored .env). Exit codes: 0 ok,
1 `status` not up to date / `check` or `sync` found problems, 2 cannot connect or bad settings.
`sync` only adds missing catalog rows from the file store; it never deletes or changes a row and
never modifies the store. The file store stays authoritative for dataset bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import catalog, migrate
from automl_ops.db.config import DatabaseConfigError, database_url, describe, make_engine
from automl_ops.tracking import TrackingError, TrackingStore, initialise_store, store_status


def _emit(args: argparse.Namespace, payload: dict[str, Any], lines: list[str]) -> None:
    print(json.dumps(payload, indent=2) if args.json else "\n".join(lines))


def cmd_upgrade(engine: Engine, args: argparse.Namespace) -> int:
    before = migrate.status(engine)
    after = migrate.upgrade(engine)
    applied = list(before.pending)
    _emit(
        args,
        {"applied": applied, **after.to_dict()},
        [
            f"applied: {', '.join(applied)}" if applied else "already up to date",
            f"revision: {after.current}",
        ],
    )
    return 0


def cmd_status(engine: Engine, args: argparse.Namespace) -> int:
    state = migrate.status(engine)
    lines = [f"applied revision: {state.current or '(none: empty database)'}"]
    lines.append(f"expected (head):  {state.head}")
    if not state.known:
        lines.append("the database is at a revision this code does not know")
    elif state.pending:
        lines.append(f"pending: {', '.join(state.pending)} (run `db upgrade`)")
    else:
        lines.append("up to date")
    _emit(args, state.to_dict(), lines)
    return 0 if state.up_to_date else 1


def cmd_check(engine: Engine, args: argparse.Namespace) -> int:
    result = migrate.check(engine)
    lines = (
        ["OK: at head and structure matches the models"]
        if result.ok
        else [f"PROBLEM: {p}" for p in result.problems]
    )
    _emit(args, {"ok": result.ok, "problems": list(result.problems)}, lines)
    return 0 if result.ok else 1


def cmd_sync(engine: Engine, args: argparse.Namespace) -> int:
    state = migrate.status(engine)
    if not state.up_to_date:
        raise DatabaseConfigError("the database is not up to date: run `db upgrade` first")
    result = catalog.sync_catalog(engine, FileDatasetStore(args.store))
    lines = [f"added {len(result.added)}, already catalogued {result.present}"]
    lines += [f"  + {ref}" for ref in result.added]
    lines += [f"CONFLICT (not changed): {c}" for c in result.conflicts]
    lines += [f"ORPHAN (row without a file, not deleted): {o}" for o in result.orphans]
    lines += [f"UNVERIFIED (not catalogued): {u}" for u in result.unverified]
    _emit(args, result.to_dict(), lines)
    return 0 if result.ok else 1


def _mlflow_lines(info: dict[str, Any]) -> list[str]:
    lines = [f"store: {info['store']}"]
    if not info.get("initialised"):
        lines.append("not initialised: run `uv run python scripts/db.py mlflow-init`")
    for key in (
        "database_created",
        "schema_revision",
        "experiment_artifact_location",
        "location_matches_configuration",
        "runs",
        "longest_run_uri",
        "run_uri_margin",
        "configured_location_longest_run_uri",
        "configured_location_problem",
    ):
        if key in info:
            lines.append(f"{key}: {info[key]}")
    return lines


def cmd_mlflow_init(args: argparse.Namespace) -> int:
    info = initialise_store(TrackingStore.from_settings(backend="postgresql"))
    _emit(args, info, _mlflow_lines(info))
    return 0


def cmd_mlflow_status(args: argparse.Namespace) -> int:
    info = store_status(TrackingStore.from_settings(backend="postgresql"))
    _emit(args, info, _mlflow_lines(info))
    good = info.get("initialised") and info.get("location_matches_configuration", True)
    return 0 if good and info.get("configured_location_ok", True) else 1


MLFLOW_COMMANDS = {"mlflow-init": cmd_mlflow_init, "mlflow-status": cmd_mlflow_status}
COMMANDS: dict[str, Any] = {
    "upgrade": cmd_upgrade,
    "status": cmd_status,
    "check": cmd_check,
    "sync": cmd_sync,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("upgrade", "apply pending migrations"),
        ("status", "show applied and expected revisions"),
        ("check", "verify the revision and the structure"),
        ("sync", "catalog the file store's versions"),
        ("mlflow-init", "create MLflow's database, schema and experiment (explicit, idempotent)"),
        ("mlflow-status", "read-only state of MLflow's PostgreSQL store"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--json", action="store_true")
        if name == "sync":
            p.add_argument("--store", type=Path, default=Path("data/store"), help="file store")
    args = parser.parse_args(argv)
    if args.command in MLFLOW_COMMANDS:
        try:
            return int(MLFLOW_COMMANDS[args.command](args))
        except (TrackingError, DatabaseConfigError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
    try:
        url = database_url()
        engine = make_engine(url)
        try:
            return int(COMMANDS[args.command](engine, args))
        except SQLAlchemyError as error:
            reason = str(getattr(error, "orig", error)).strip().splitlines()[0]
            print(f"ERROR: cannot use the database {describe(url)}: {reason}", file=sys.stderr)
            return 2
        finally:
            engine.dispose()
    except DatabaseConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
