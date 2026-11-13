"""`jobs`, `job`, `compare` and `reproduce` on the command line (Steps 15-16), as real
subprocesses. The last test uses the prepared Abalone demo file and is skipped without it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from mlflow import MlflowClient

from test_tracking_cli import ALGORITHMS, SCRIPT, cli, finalize_args, only_package, prepared

REGRESSORS = "ridge,random_forest"
DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


def track_train(
    tmp: Path,
    mlflow_dir: Path,
    *extra: str,
    ref: str = "toy@v1",
    target: str = "y",
    algorithms: str = ALGORITHMS,
) -> str:
    done = cli(
        tmp, mlflow_dir, "train", ref, "--target", target, "--algorithms", algorithms, "--track",
        "--json", *extra,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    return str(json.loads(done.stdout)["tracking"]["job_key"])


def test_history_detail_compare_and_reproduce_end_to_end(tmp_path: Path) -> None:
    prepared(tmp_path)
    mlflow_dir = tmp_path / "mlflow"
    package = cli(tmp_path, mlflow_dir, *finalize_args("--track"))
    assert package.returncode == 0, package.stderr
    a = only_package(tmp_path)
    b = track_train(tmp_path, mlflow_dir, "--seed", "7", "--folds", "3")

    listing = json.loads(cli(tmp_path, mlflow_dir, "jobs", "--json").stdout)
    assert {j["job_key"] for j in listing} == {a, b}
    final = cli(tmp_path, mlflow_dir, "jobs", "--kind", "final")
    assert final.returncode == 0 and a in final.stdout and b not in final.stdout

    detail = cli(tmp_path, mlflow_dir, "job", a)
    assert detail.returncode == 0 and "Recorded single test result" in detail.stdout
    assert "not independent or leakage-proof" in detail.stdout
    assert "Recorded single test result" not in cli(tmp_path, mlflow_dir, "job", b).stdout

    compared = cli(tmp_path, mlflow_dir, "compare", a, b, "--json")
    assert compared.returncode == 0, compared.stderr
    result = json.loads(compared.stdout)
    assert result["metric"] == "f1_macro" and result["split_note"]
    assert "test" not in " ".join(result["rows"][0]).replace("test_fraction", "")
    text = cli(tmp_path, mlflow_dir, "compare", a, b)
    assert text.returncode == 0 and "NOTE:" in text.stdout and "(selected in job)" in text.stdout

    for key in (a, b):
        done = cli(tmp_path, mlflow_dir, "reproduce", key)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "REPRODUCED" in done.stdout and "bit-identical" in done.stdout
    detail_json = json.loads(cli(tmp_path, mlflow_dir, "reproduce", a, "--json").stdout)
    assert detail_json["ok"] and detail_json["bit_identical"]
    assert any(i["name"] == "recorded test result vs sealed package" for i in detail_json["items"])


def test_the_commands_report_problems_with_the_documented_exit_codes(tmp_path: Path) -> None:
    prepared(tmp_path)
    mlflow_dir = tmp_path / "mlflow"
    key = track_train(tmp_path, mlflow_dir)
    assert cli(tmp_path, mlflow_dir, "job", "m-nope").returncode == 2
    assert cli(tmp_path, mlflow_dir, "reproduce", "m-nope").returncode == 2
    assert cli(tmp_path, mlflow_dir, "compare", key).returncode == 2  # needs two jobs
    assert cli(tmp_path, mlflow_dir, "compare", key, key).returncode == 2
    empty = cli(tmp_path, tmp_path / "no-store", "jobs")
    assert empty.returncode == 0 and "no tracked jobs" in empty.stdout
    assert not (tmp_path / "no-store").exists()  # a query never creates a store

    # the dataset version is missing from the given store: cannot run (2), not "not reproduced"
    moved = subprocess.run(
        [sys.executable, str(SCRIPT), "--store", str(tmp_path / "other"), "--mlflow-dir",
         str(mlflow_dir), "reproduce", key],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert moved.returncode == 2 and "ERROR" in moved.stderr

    # a changed logged metric: not reproduced (1) and the metric is named
    client = MlflowClient(tracking_uri=f"sqlite:///{(mlflow_dir / 'mlflow.db').resolve()}")
    target = client.search_runs(
        ["1"],
        f"tags.`platform.job_key` = '{key}' and tags.`platform.algorithm` = 'logistic_regression'",
    )[0]
    client.log_metric(target.info.run_id, "cv_accuracy_mean", 0.01, timestamp=2**42)
    bad = cli(tmp_path, mlflow_dir, "reproduce", key)
    assert bad.returncode == 1 and "NOT REPRODUCED" in bad.stdout
    assert "logistic_regression.cv_accuracy_mean" in bad.stdout


def test_a_missing_package_is_not_checked_and_says_so(tmp_path: Path) -> None:
    prepared(tmp_path)
    mlflow_dir = tmp_path / "mlflow"
    assert cli(tmp_path, mlflow_dir, *finalize_args("--track")).returncode == 0
    key = only_package(tmp_path)
    (tmp_path / "other-models").mkdir()
    done = cli(tmp_path, mlflow_dir, "--models", str(tmp_path / "other-models"), "reproduce", key)
    assert done.returncode == 0 and "NOT CHECKED" in done.stdout


@pytest.mark.skipif(
    not (DEMO / "abalone_train_pool.csv").exists(),
    reason="demo data not prepared; run `make demo-data` first",
)
def test_abalone_real_data_two_jobs_compared_and_reproduced(tmp_path: Path) -> None:
    """The Gate 3 sequence on the real Abalone pool: a finalized job and a CV-only job with another
    seed and fewer folds are tracked, compared and both reproduced from their records."""
    assert (
        cli(
            tmp_path, tmp_path / "unused", "ingest", "abalone", str(DEMO / "abalone_train_pool.csv")
        ).returncode
        == 0
    )
    mlflow_dir = tmp_path / "mlflow"
    done = cli(
        tmp_path, mlflow_dir, "finalize", "abalone@v1", "--target", "Rings", "--algorithms",
        REGRESSORS, "--confirm", "--track",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    a = only_package(tmp_path)
    b = track_train(
        tmp_path, mlflow_dir, "--seed", "7", "--folds", "3",
        ref="abalone@v1", target="Rings", algorithms=REGRESSORS,
    )  # fmt: skip
    compared = json.loads(cli(tmp_path, mlflow_dir, "compare", a, b, "--json").stdout)
    assert compared["metric"] == "rmse" and compared["direction"] == "lower"
    assert {r["job_key"] for r in compared["rows"]} == {a, b}
    for key in (a, b):
        report = json.loads(cli(tmp_path, mlflow_dir, "reproduce", key, "--json").stdout)
        assert report["ok"] and report["bit_identical"] and report["environment_same"]
