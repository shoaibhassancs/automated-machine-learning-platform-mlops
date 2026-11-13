"""`--track` and `track-package` on the command line (Steps 13-14), as real subprocesses."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from automl_ops.package import package_state, verify_package
from automl_ops.tracking import TrackingStore, describe_job, tracking_state
from test_datasets_cli import toy_csv

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "datasets.py"
ALGORITHMS = "logistic_regression,random_forest"


def cli(tmp: Path, mlflow_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--store",
            str(tmp / "store"),
            "--models",
            str(tmp / "models"),
            "--mlflow-dir",
            str(mlflow_dir),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def prepared(tmp: Path) -> None:
    csv = tmp / "toy.csv"
    toy_csv(csv)
    assert cli(tmp, tmp / "unused", "ingest", "toy", str(csv)).returncode == 0


def finalize_args(*extra: str) -> tuple[str, ...]:
    return ("finalize", "toy@v1", "--target", "y", "--algorithms", ALGORITHMS, "--confirm", *extra)


def only_package(tmp: Path) -> str:
    (package,) = [p.name for p in (tmp / "models").iterdir()]
    return package


def test_train_track_logs_cross_validation_only_and_reports_the_run(tmp_path: Path) -> None:
    prepared(tmp_path)
    mlflow_dir = tmp_path / "mlflow"
    done = cli(
        tmp_path, mlflow_dir, "train", "toy@v1", "--target", "y", "--algorithms", ALGORITHMS,
        "--track", "--json",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    tracking = json.loads(done.stdout)["tracking"]
    assert tracking["state"] == "complete" and len(tracking["child_run_ids"]) == 2
    info = describe_job(tracking["job_key"], TrackingStore(mlflow_dir))
    assert info is not None and info["parent"]["tags"]["platform.kind"] == "train"
    for run in (info["parent"], *info["children"].values()):
        assert not any(name.startswith("test_") for name in run["metrics"])
    assert not (tmp_path / "models").exists()  # train never creates a package


def test_without_track_nothing_is_written_to_the_tracking_store(tmp_path: Path) -> None:
    prepared(tmp_path)
    mlflow_dir = tmp_path / "mlflow"
    done = cli(tmp_path, mlflow_dir, "train", "toy@v1", "--target", "y", "--algorithms", ALGORITHMS)
    assert done.returncode == 0 and not mlflow_dir.exists()


def test_finalize_track_logs_after_sealing_and_a_second_track_is_refused(tmp_path: Path) -> None:
    prepared(tmp_path)
    mlflow_dir = tmp_path / "mlflow"
    done = cli(tmp_path, mlflow_dir, *finalize_args("--track"))
    assert done.returncode == 0, done.stderr
    assert (
        "tracking: logged job" in done.stdout and "not independent or leakage-proof" in done.stdout
    )
    package = only_package(tmp_path)
    assert package_state(tmp_path / "models" / package) == "sealed"
    store = TrackingStore(mlflow_dir)
    info = describe_job(package, store)
    assert info is not None and info["parent"]["status"] == "FINISHED"
    selected = info["parent"]["tags"]["platform.selected_algorithm"]
    assert info["children"][selected]["tags"]["platform.final_test"] == "logged"
    assert "package/model.joblib" in info["children"][selected]["artifacts"]

    again = cli(tmp_path, mlflow_dir, "track-package", package)
    assert again.returncode == 2 and "already tracked" in again.stderr
    assert "Traceback" not in again.stderr
    assert tracking_state(store)["by_role_and_state"]["job:complete"] == 1
    assert verify_package(tmp_path / "models", package).ok


def test_a_tracking_failure_leaves_a_sealed_package_that_track_package_then_logs(
    tmp_path: Path,
) -> None:
    prepared(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the store folder should be")
    done = cli(tmp_path, blocked, *finalize_args("--track"))
    assert done.returncode == 3 and "TRACKING PENDING" in done.stderr
    assert "Traceback" not in done.stderr
    package = only_package(tmp_path)
    assert f"track-package {package}" in done.stderr
    assert "Final test (held-out part" in done.stdout  # the finalization itself succeeded
    assert verify_package(tmp_path / "models", package).ok

    good = tmp_path / "mlflow"
    recovered = cli(tmp_path, good, "track-package", package)
    assert recovered.returncode == 0, recovered.stderr
    assert "tracking: logged job" in recovered.stdout
    info = describe_job(package, TrackingStore(good))
    assert info is not None and info["parent"]["tags"]["platform.logging"] == "complete"
    assert verify_package(tmp_path / "models", package).ok


def test_track_package_refuses_unknown_and_malformed_ids_without_a_traceback(
    tmp_path: Path,
) -> None:
    for name in ("m-20200101T000000Z-00000000", "../etc"):
        bad = cli(tmp_path, tmp_path / "mlflow", "track-package", name)
        assert bad.returncode == 2 and "Traceback" not in bad.stderr
    assert not (tmp_path / "mlflow" / "mlflow.db").exists()
