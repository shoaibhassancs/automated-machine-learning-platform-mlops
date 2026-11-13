"""Smoke test for the MLflow setup (BUILD_PLAN Step 13): log one dummy job and read it back.

    uv run python scripts/mlflow_smoke.py [--root data/mlflow-smoke]

Uses the platform's own `automl_ops.tracking` (SQLite metadata + local artifact folder, telemetry
off) with a hand-made two-algorithm record, so it exercises the same code path as `--track`. The
default root is separate from the real store (`data/mlflow`); both are git-ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from automl_ops.tracking import (
    AlreadyTrackedError,
    TrackingStore,
    describe_job,
    log_job,
    record_from_training,
    telemetry_disabled,
)

FOLD = {"accuracy": 0.9, "f1_macro": 0.85}


def dummy_training() -> dict[str, object]:
    summary = {m: {"mean": v, "std": 0.01, "folds_defined": 2, "folds": 2} for m, v in FOLD.items()}
    ok = {
        "algorithm": "dummy_ok",
        "label": "Dummy",
        "params": {"c": 1.0},
        "status": "ok",
        "error": None,
        "seconds": 0.5,
        "fold_seconds": [0.2, 0.3],
        "fold_metrics": [dict(FOLD), dict(FOLD)],
        "summary": summary,
        "oof": {"metrics": dict(FOLD), "undefined": {}},
    }
    failed = {
        **ok,
        "algorithm": "dummy_failed",
        "status": "failed",
        "error": {"type": "Boom", "message": "smoke test failure"},
        "summary": {},
        "oof": None,
        "fold_metrics": [],
        "fold_seconds": [],
    }
    return {
        "task": "classification",
        "target": "y",
        "settings": {"seed": 1, "folds": 2, "test_fraction": 0.2},
        "folds": {"requested": 2, "used": 2},
        "results": [ok, failed],
        "seconds": 1.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/mlflow-smoke"))
    args = parser.parse_args(argv)
    store = TrackingStore(args.root)
    print(f"telemetry disabled in this process: {telemetry_disabled()}")
    print(f"tracking uri: {store.tracking_uri}\nartifacts:    {store.artifact_uri}")
    record = record_from_training(
        dummy_training(),
        {"ref": "smoke@v1", "sha256": "0" * 64},
        {"python": sys.version.split()[0]},
    )
    tracked = log_job(record, store)
    print(
        f"logged job {tracked.job_key}: parent {tracked.parent_run_id[:8]}, "
        f"children {sorted(tracked.child_run_ids)}"
    )
    info = describe_job(record.job_key, store)
    assert info is not None
    statuses = {k: v["status"] for k, v in info["children"].items()}
    print(f"read back: parent {info['parent']['status']}, children {statuses}")
    try:
        log_job(record, store)
    except AlreadyTrackedError as error:
        print(f"second attempt refused: {error}")
    else:
        print("FAIL: a complete job accepted a second attempt")
        return 1
    ok = (
        info["parent"]["status"] == "FINISHED"
        and statuses == {"dummy_ok": "FINISHED", "dummy_failed": "FAILED"}
        and telemetry_disabled()
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
