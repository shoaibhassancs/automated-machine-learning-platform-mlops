"""MLflow tracking (BUILD_PLAN Steps 13-14): layout, isolation, recovery and telemetry.

The sealed packages come from the shared `clf_run` / `reg_run` fixtures (small invented data), and
every store lives in a temporary folder.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from mlflow import MlflowClient

import automl_ops.tracking as tracking
from automl_ops.hashing import sha256_file
from automl_ops.package import SEAL_FILE, package_state, verify_package
from automl_ops.tracking import (
    AlreadyTrackedError,
    JobRecord,
    TrackingError,
    TrackingStore,
    describe_job,
    log_job,
    record_from_package,
    record_from_training,
    tracking_state,
)
from conftest import Run

SRC = Path(__file__).resolve().parents[1] / "src" / "automl_ops"
WRITES = ("create_run", "log_batch", "log_text", "log_artifact", "set_terminated", "set_tag")
QUIET = {"MLFLOW_DISABLE_TELEMETRY": "true", "DO_NOT_TRACK": "true"}


def record_of(run: Run) -> JobRecord:
    return record_from_package(run.root, run.result.package_id)


def store_in(path: Path) -> TrackingStore:
    return TrackingStore(path / "mlflow")


# --- what a sealed package gives back (self-sufficiency) --------------------------------------


def test_a_sealed_package_holds_everything_needed_to_rebuild_the_runs(clf_run: Run) -> None:
    record = record_of(clf_run)
    manifest = json.loads((clf_run.result.path / "manifest.json").read_text())
    assert record.job_key == clf_run.result.package_id == record.package_id
    assert record.kind == "final" and record.test_result is not None
    # every algorithm of the training run is there with params, status, per-fold and pooled numbers
    keys = [r["algorithm"] for r in record.training["results"]]
    assert keys == [r.key for r in clf_run.training.results]
    for result in record.training["results"]:
        assert {"params", "status", "error", "seconds", "fold_seconds", "fold_metrics"} <= set(
            result
        )
        assert {"summary", "oof"} <= set(result)
        assert len(result["fold_metrics"]) == clf_run.training.folds_used
    # selection, provenance and the test-row fingerprint (for later reuse queries)
    assert record.selection is not None
    assert record.selection["selected"]["algorithm"] == clf_run.frozen.data["algorithm"]
    assert record.selection_sha256 == clf_run.frozen.sha256 == manifest["selection_sha256"]
    assert record.dataset["sha256"] == clf_run.dataset.sha256
    assert record.split is not None
    assert record.split["test_sha256"] == clf_run.training.split.fingerprints()["test_sha256"]
    assert record.libraries["scikit-learn"] and record.settings["seed"] == 42


def test_only_a_sealed_verified_package_can_be_tracked(clf_run: Run, tmp_path: Path) -> None:
    with pytest.raises(TrackingError, match="not a package id"):
        record_from_package(clf_run.root, "../escape")
    with pytest.raises(TrackingError, match="no package"):
        record_from_package(clf_run.root, "m-20200101T000000Z-00000000")
    # an open package (no attempt yet) and a tampered copy are both refused
    from conftest import writable_copy

    root = writable_copy(clf_run, tmp_path / "copy")
    package = root / clf_run.result.package_id
    (package / "manifest.json").write_text('{"tampered": true}')
    with pytest.raises(TrackingError, match="failed verification"):
        record_from_package(root, clf_run.result.package_id)
    (package / SEAL_FILE).unlink()
    (package / "test_result.json").unlink()
    (package / "attempt.json").unlink()
    assert package_state(package) == "open"
    with pytest.raises(TrackingError, match="open, not sealed"):
        record_from_package(root, clf_run.result.package_id)


# --- the layout of a logged job ----------------------------------------------------------------


def test_a_finalized_job_is_one_parent_and_one_child_per_algorithm(
    clf_run: Run, tmp_path: Path
) -> None:
    store = store_in(tmp_path)
    tracked = log_job(record_of(clf_run), store)
    assert not tracked.resumed
    info = describe_job(tracked.job_key, store)
    assert info is not None
    selected = clf_run.frozen.data["algorithm"]
    assert set(info["children"]) == {r.key for r in clf_run.training.results}
    parent = info["parent"]
    assert parent["status"] == "FINISHED" and parent["tags"][tracking.T_STATE] == "complete"
    assert parent["tags"]["platform.dataset_sha256"] == clf_run.dataset.sha256
    assert parent["tags"]["platform.package_id"] == clf_run.result.package_id
    assert parent["tags"]["platform.selected_algorithm"] == selected
    assert parent["tags"]["lib.scikit-learn"] and parent["params"]["seed"] == "42"
    assert not any(name.startswith("test_") for name in parent["metrics"])
    assert {"training.json", "provenance.json", "selection_report.json"} <= set(parent["artifacts"])
    for key, child in info["children"].items():
        assert child["status"] == "FINISHED" and child["tags"][tracking.T_STATE] == "complete"
        assert child["tags"][tracking.PARENT_TAG] == parent["run_id"]
        assert "evaluation.json" in child["artifacts"]
        source = next(r for r in clf_run.training.results if r.key == key)
        summary = source.summary["f1_macro"]
        assert child["metrics"]["cv_f1_macro_mean"] == pytest.approx(summary["mean"])
        assert child["metrics"]["fit_seconds"] == pytest.approx(source.seconds, abs=1e-3)
        # one point per fold for the per-fold series
        assert child["metric_points"]["fold_f1_macro"] == clf_run.training.folds_used
        for name, value in source.params.items():
            expected = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            assert child["params"][f"param.{name}"] == expected
        has_test = any(name.startswith("test_") for name in child["metrics"])
        assert has_test is (key == selected)
        assert ("package/model.joblib" in child["artifacts"]) is (key == selected)


def test_the_final_test_result_is_logged_once_on_the_selected_run_and_matches_the_package(
    clf_run: Run, tmp_path: Path
) -> None:
    store = store_in(tmp_path)
    tracked = log_job(record_of(clf_run), store)
    info = describe_job(tracked.job_key, store)
    assert info is not None
    selected = clf_run.frozen.data["algorithm"]
    child = info["children"][selected]
    package_metrics = clf_run.result.test_result["evaluation"]["metrics"]
    for name, value in package_metrics.items():
        if value is not None:
            assert child["metrics"][f"test_{name}"] == pytest.approx(value)
            assert child["metric_points"][f"test_{name}"] == 1
    assert child["tags"]["platform.final_test"] == "logged"
    # the logged package files are byte-identical to the sealed ones (seal hashes still hold)
    seal = json.loads((clf_run.result.path / SEAL_FILE).read_text())
    for name, digest in seal["files"].items():
        copy = tracking.artifact_file(tracked.job_key, selected, f"package/{name}", store)
        assert sha256_file(copy) == digest
    limitation = tracking.artifact_file(tracked.job_key, selected, "test_result.json", store)
    assert "not independent or leakage-proof" in limitation.read_text()
    assert verify_package(clf_run.root, clf_run.result.package_id).ok


def test_a_regression_job_logs_in_the_same_layout(reg_run: Run, tmp_path: Path) -> None:
    store = store_in(tmp_path)
    tracked = log_job(record_of(reg_run), store)
    info = describe_job(tracked.job_key, store)
    assert info is not None
    selected = reg_run.frozen.data["algorithm"]
    assert {"test_rmse", "test_mae", "test_r2"} <= set(info["children"][selected]["metrics"])
    assert info["parent"]["metrics"]["selected_cv_rmse_mean"] > 0


def toy_training(failed: bool = True) -> dict[str, Any]:
    scores = {"f1_macro": 0.8, "accuracy": 0.9}
    ok: dict[str, Any] = {
        "algorithm": "logistic_regression",
        "label": "Logistic regression",
        "params": {"C": 1.0, "max_iter": 200},
        "status": "ok",
        "error": None,
        "seconds": 0.4,
        "fold_seconds": [0.2, 0.2],
        "fold_metrics": [dict(scores, roc_auc=None), dict(scores, roc_auc=0.7)],
        "summary": {
            m: {"mean": v, "std": 0.01, "folds_defined": 2, "folds": 2} for m, v in scores.items()
        }
        | {"roc_auc": {"mean": None, "std": None, "folds_defined": 1, "folds": 2}},
        "oof": {"metrics": dict(scores, roc_auc=None), "undefined": {"roc_auc": "one class"}},
    }
    results = [ok]
    if failed:
        results.append(
            {
                **ok,
                "algorithm": "random_forest",
                "label": "Random forest",
                "status": "failed",
                "error": {"type": "ValueError", "message": "boom"},
                "summary": {},
                "oof": None,
                "fold_metrics": [],
                "fold_seconds": [],
            }
        )
    return {
        "task": "classification",
        "target": "y",
        "settings": {"seed": 7, "folds": 2, "test_fraction": 0.2},
        "folds": {"requested": 2, "used": 2},
        "results": results,
        "seconds": 1.2,
    }


def train_record() -> JobRecord:
    return record_from_training(
        toy_training(), {"ref": "toy@v1", "sha256": "ab" * 32}, {"python": "3.12"}
    )


def test_a_failed_algorithm_is_a_failed_child_run_and_undefined_metrics_are_not_zeros(
    tmp_path: Path,
) -> None:
    store = store_in(tmp_path)
    tracked = log_job(train_record(), store)
    info = describe_job(tracked.job_key, store)
    assert info is not None
    ok, failed = info["children"]["logistic_regression"], info["children"]["random_forest"]
    assert failed["status"] == "FAILED" and ok["status"] == "FINISHED"
    assert failed["tags"]["platform.error_type"] == "ValueError"
    assert failed["tags"]["platform.error_message"] == "boom"
    assert not any(k.startswith(("cv_", "oof_", "fold_")) for k in failed["metrics"])
    # roc_auc is undefined in one fold: that fold point and the mean/pooled values are absent
    assert "cv_roc_auc_mean" not in ok["metrics"] and "oof_roc_auc" not in ok["metrics"]
    assert ok["metric_points"]["fold_roc_auc"] == 1
    assert info["parent"]["status"] == "FINISHED"


def test_a_cross_validation_only_job_never_carries_test_metrics_or_a_package(
    tmp_path: Path,
) -> None:
    store = store_in(tmp_path)
    tracked = log_job(train_record(), store)
    info = describe_job(tracked.job_key, store)
    assert info is not None
    assert info["parent"]["tags"]["platform.kind"] == "train"
    for run in (info["parent"], *info["children"].values()):
        assert not any(name.startswith("test_") for name in run["metrics"])
        assert not any(
            a.startswith("package/") or a == "test_result.json" for a in run["artifacts"]
        )
        assert run["tags"].get("platform.final_test") is None
    with pytest.raises(TrackingError, match="cannot carry a test result"):
        JobRecord("k", "train", 0, {}, {}, {}, test_result={"evaluation": {}})


# --- recovery: idempotent, resumable, one completion -------------------------------------------


@contextmanager
def failing_write(fail_at: int | None, *, after: bool = False) -> Iterator[dict[str, int]]:
    """Make the `fail_at`-th MLflow write call raise (before it runs, or just after it ran)."""
    state = {"calls": 0, "depth": 0}
    originals = {name: getattr(MlflowClient, name) for name in WRITES}

    def wrap(name: str) -> Callable[..., Any]:
        original = originals[name]

        def call(self: Any, *args: Any, **kwargs: Any) -> Any:
            if state["depth"]:  # a write made by another write (log_text -> log_artifact)
                return original(self, *args, **kwargs)
            state["calls"] += 1
            number = state["calls"]
            if number == fail_at and not after:
                raise RuntimeError(f"injected failure before {name} #{fail_at}")
            state["depth"] += 1
            try:
                result = original(self, *args, **kwargs)
            finally:
                state["depth"] -= 1
            if number == fail_at:
                raise RuntimeError(f"injected failure after {name} #{fail_at}")
            return result

        return call

    for name in WRITES:
        setattr(MlflowClient, name, wrap(name))
    try:
        yield state
    finally:
        for name, original in originals.items():
            setattr(MlflowClient, name, original)


def comparable(info: dict[str, Any]) -> dict[str, Any]:
    """The job's content without run ids, for comparing a recovered job with a clean one."""

    def strip(run: dict[str, Any]) -> dict[str, Any]:
        tags = {k: v for k, v in run["tags"].items() if k != tracking.PARENT_TAG}
        return {**{k: v for k, v in run.items() if k not in ("run_id", "tags")}, "tags": tags}

    return {
        "parent": strip(info["parent"]),
        "children": {k: strip(v) for k, v in sorted(info["children"].items())},
    }


@pytest.fixture(scope="module")
def clean_job(clf_run: Run, tmp_path_factory: pytest.TempPathFactory) -> tuple[int, dict[str, Any]]:
    """Number of write calls of one clean run, and its content."""
    store = store_in(tmp_path_factory.mktemp("clean"))
    with failing_write(None) as state:
        tracked = log_job(record_of(clf_run), store)
    info = describe_job(tracked.job_key, store)
    assert info is not None
    return state["calls"], comparable(info)


def test_a_failure_at_any_logging_step_is_recovered_without_duplicates(
    clf_run: Run, tmp_path: Path, clean_job: tuple[int, dict[str, Any]]
) -> None:
    calls, clean = clean_job
    assert calls > 20  # the sweep really covers parent, children, metrics, artifacts and the end
    record = record_of(clf_run)
    for fail_at in range(1, calls + 1):
        for after in (False, True):
            store = store_in(tmp_path / f"s{fail_at}{'a' if after else 'b'}")
            with (
                failing_write(fail_at, after=after),
                pytest.raises(TrackingError, match="injected"),
            ):
                log_job(record, store)
            state = tracking_state(store)
            done = state["by_role_and_state"].get("job:complete", 0)
            if fail_at == calls and after:
                # the very last write (the parent's complete tag) took effect before the client
                # reported an error: the job is complete, so a retry is rightly refused
                assert done == 1
                with pytest.raises(AlreadyTrackedError):
                    log_job(record, store)
                info = describe_job(record.job_key, store)
                assert info is not None and comparable(info) == clean
                continue
            assert done == 0, (fail_at, after)
            tracked = log_job(record, store)  # the retry completes the missing work
            assert tracked.resumed or fail_at == 1 and not after, (fail_at, after)
            info = describe_job(record.job_key, store)
            assert info is not None
            assert comparable(info) == clean, f"differs after a failure at call {fail_at} {after}"
            count = tracking_state(store)
            assert count["runs"] == 1 + len(clf_run.training.results), (fail_at, after, count)
            with pytest.raises(AlreadyTrackedError):
                log_job(record, store)


def test_repeated_failures_then_success_still_give_the_clean_result(
    clf_run: Run, tmp_path: Path, clean_job: tuple[int, dict[str, Any]]
) -> None:
    _, clean = clean_job
    record, store = record_of(clf_run), store_in(tmp_path)
    failures = 0
    for fail_at in (7, 6, 9, 8, 12, 11):  # each retry fails again, at a later or equal point
        try:
            with failing_write(fail_at):
                log_job(record, store)
        except TrackingError:
            failures += 1
    assert failures >= 4  # the job really was interrupted several times
    try:
        log_job(record, store)
    except AlreadyTrackedError:
        pass  # a late retry already completed it
    info = describe_job(record.job_key, store)
    assert info is not None and comparable(info) == clean
    assert tracking_state(store)["runs"] == 1 + len(clf_run.training.results)


def test_a_complete_job_rejects_another_attempt_and_changes_nothing(
    clf_run: Run, tmp_path: Path
) -> None:
    store = store_in(tmp_path)
    record = record_of(clf_run)
    log_job(record, store)
    before = describe_job(record.job_key, store)
    with failing_write(None) as state, pytest.raises(AlreadyTrackedError, match="already tracked"):
        log_job(record, store)
    assert state["calls"] == 0  # not a single write was attempted
    assert describe_job(record.job_key, store) == before


def test_two_jobs_share_a_store_without_interfering(
    clf_run: Run, reg_run: Run, tmp_path: Path
) -> None:
    store = store_in(tmp_path)
    log_job(record_of(clf_run), store)
    log_job(record_of(reg_run), store)
    log_job(train_record(), store)
    assert tracking_state(store)["by_role_and_state"] == {
        "job:complete": 3,
        "algorithm:complete": len(clf_run.training.results) + len(reg_run.training.results) + 2,
    }


def test_an_unwritable_store_leaves_the_package_sealed_and_recoverable(
    clf_run: Run, tmp_path: Path
) -> None:
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("a file where the store folder should be")
    with pytest.raises(TrackingError):
        log_job(record_of(clf_run), TrackingStore(blocked))
    assert package_state(clf_run.result.path) == "sealed"
    assert verify_package(clf_run.root, clf_run.result.package_id).ok
    # the same package then tracks fine into a working store (the `track-package` path)
    store = store_in(tmp_path)
    assert not log_job(record_of(clf_run), store).resumed


# --- isolation: nothing but plain data crosses the module boundary ----------------------------


def imports_of(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_only_the_tracking_module_imports_mlflow() -> None:
    importers = {p.name for p in SRC.glob("*.py") if "mlflow" in imports_of(p)}
    assert importers == {"tracking.py"}
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    assert all("mlflow" not in imports_of(p) for p in scripts.glob("*.py"))


def test_tracking_cannot_reach_data_rows_models_or_the_test_part() -> None:
    path = SRC / "tracking.py"
    imported = imports_of(path)
    assert not imported & {"pandas", "numpy", "sklearn", "joblib"}
    source = path.read_text()
    for forbidden in ("load_package", "_read_test_part", "load_pipeline_for_test", "joblib.load"):
        assert forbidden not in source
    # the logged record is plain data: no data frame, no model object
    import inspect

    assert list(inspect.signature(log_job).parameters) == ["record", "store"]
    assert "DataFrame" not in inspect.getsource(JobRecord)


def test_tracking_a_package_never_reads_a_test_row(
    clf_run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import automl_ops.finalize as finalize_module

    def forbidden(*args: object, **kwargs: object) -> None:  # pragma: no cover
        raise AssertionError("the test part was read while tracking")

    monkeypatch.setattr(finalize_module, "_read_test_part", forbidden)
    log_job(record_of(clf_run), store_in(tmp_path))


# --- telemetry ---------------------------------------------------------------------------------


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    """An environment without anything that makes MLflow think it is in CI or a test run."""
    drop = (
        "CI",
        "GITHUB_ACTIONS",
        "PYTEST_CURRENT_TEST",
        "MLFLOW_DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
    )
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith("MLFLOW_")}
    return {**env, **extra, "MLFLOW_DISABLE_AGENT_HINT": "1"}


def run_python(code: str, env: dict[str, str]) -> str:
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return done.stdout.strip().splitlines()[-1]


def test_telemetry_is_disabled_by_the_tracking_module_and_would_not_be_otherwise() -> None:
    probe = "from mlflow.telemetry.utils import is_telemetry_disabled as d; print(d())"
    # Without the setting, MLflow in a plain process reports telemetry as enabled ...
    assert run_python(probe, clean_env({})) == "False"
    # ... and with the documented setting it is disabled.
    assert run_python(probe, clean_env(QUIET)) == "True"
    # The tracking module applies the setting itself, before MLflow is imported.
    code = "import automl_ops.tracking as t; print(t.telemetry_disabled())"
    assert run_python(code, clean_env({})) == "True"
    assert tracking.quiet_environment()["MLFLOW_DISABLE_TELEMETRY"] == "true"


def test_the_ui_command_is_explicit_and_local_only(tmp_path: Path) -> None:
    store = TrackingStore(tmp_path / "m")
    command = store.ui_command(5123)
    assert command[:2] == ["mlflow", "server"]
    assert command[command.index("--backend-store-uri") + 1].startswith("sqlite:////")
    assert command[command.index("--default-artifact-root") + 1].startswith("file:///")
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "5123"
