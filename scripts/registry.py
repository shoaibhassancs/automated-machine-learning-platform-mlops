"""Registry mirror commands (P4-6): `status`, `sync`, `check`, `reconcile`.

The registry in PostgreSQL is authoritative; MLflow only mirrors it, and only when
`AUTOML_REGISTRY_MIRROR=on`. `status` and `check` change nothing. `sync` makes MLflow agree with the
registry. `reconcile` is the explicit repair of MLflow drift: it needs `--apply` and a `--reason`
and writes an audit record. Nothing here deletes anything in MLflow. Exit codes: 0 fine, 1 a problem
was found or the action did not finish, 2 cannot run (mirror disabled, database not ready).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automl_ops.db import migrate  # noqa: E402
from automl_ops.db.config import DatabaseConfigError, make_engine, setting  # noqa: E402
from automl_ops.registry_mirror import (  # noqa: E402
    InvalidReason,
    Mirror,
    MirrorDisabled,
    MirrorModelNotFound,
    MirrorSettingError,
)


def _show(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _status(s: Any) -> dict[str, Any]:
    return {
        "model": s.model,
        "state": s.state,
        "desired_seq": s.desired_seq,
        "synced_seq": s.synced_seq,
        "pending_events": s.pending_events,
        "attempts": s.attempts,
        "last_error": s.last_error,
        "last_error_kind": s.last_error_kind,
        "last_success_at": s.last_success_at,
        "next_attempt_at": s.next_attempt_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", type=Path, help="model packages (default data/models)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="mirror standing of every model (read-only)")
    for name, text in (
        ("sync", "make MLflow agree with the registry"),
        ("check", "compare the registry with MLflow (read-only)"),
    ):
        one = sub.add_parser(name, help=text)
        one.add_argument("model")
    rec = sub.add_parser("reconcile", help="explicit repair of MLflow drift (audited)")
    rec.add_argument("model")
    rec.add_argument("--reason", required=True, help="why (1-500 characters)")
    rec.add_argument("--apply", action="store_true", help="actually change MLflow; else a dry run")
    args = parser.parse_args(argv)

    try:
        engine = make_engine()
        state = migrate.status(engine)
    except Exception as error:
        print(f"cannot reach the application database ({type(error).__name__})", file=sys.stderr)
        return 2
    if not state.up_to_date:
        print(
            f"the database is at revision {state.current or 'none'}, the code expects "
            f"{state.head}: run `uv run python scripts/db.py upgrade`",
            file=sys.stderr,
        )
        return 2
    models = args.models or Path(setting("AUTOML_MODELS_DIR") or PROJECT_ROOT / "data" / "models")
    try:
        mirror = Mirror(engine, models)
        _ = mirror.enabled
        if args.command == "status":
            _show(
                {
                    "enabled": mirror.enabled,
                    "problem": mirror.problem(),
                    "models": [_status(s) for s in mirror.overview()],
                }
            )
            return 0
        if args.command == "sync":
            outcome = mirror.sync(args.model)
            _show(
                {
                    "result": outcome.result,
                    "seq": outcome.seq,
                    "actions": outcome.actions,
                    "findings": [f.to_dict() for f in outcome.findings],
                    "error": outcome.error,
                }
            )
            return 0 if outcome.result in ("synced", "up_to_date") else 1
        if args.command == "check":
            report = mirror.check(args.model)
            _show(
                {
                    "model": report.model,
                    "desired_seq": report.desired_seq,
                    "synced_seq": report.synced_seq,
                    "findings": [f.to_dict() for f in report.findings],
                }
            )
            return 1 if any(f.stops_sync or f.category == "pending" for f in report.findings) else 0
        if not args.apply:  # a dry run: what a reconcile would do
            report = mirror.check(args.model)
            from automl_ops.registry_mirror import intended_repair  # noqa: PLC0415

            _show(
                {
                    "dry_run": True,
                    "findings": [f.to_dict() for f in report.findings],
                    "intended": [intended_repair(f) for f in report.findings],
                }
            )
            return 0
        done = mirror.reconcile(args.model, args.reason)
        _show(
            {
                "outcome": done.outcome,
                "seq": done.seq,
                "detected": [f.to_dict() for f in done.detected],
                "intended": done.intended,
                "actions": done.actions,
                "result": [f.to_dict() for f in done.result],
                "error": done.error,
            }
        )
        return 0 if done.outcome == "ok" else 1
    except (MirrorDisabled, MirrorSettingError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except MirrorModelNotFound as error:
        print(str(error), file=sys.stderr)
        return 2
    except InvalidReason as error:
        print(str(error), file=sys.stderr)
        return 2
    except DatabaseConfigError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
