"""P4-4: the worker end to end, on a real PostgreSQL server and with real subprocesses.

Throwaway `automl_test_*` databases, scratch stores and model folders only. Tracking tests use
scratch MLflow directories and compare the original archive `data/mlflow` before and after.
NOT RUN when the server is unreachable.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from automl_ops import finalize as finalize_module
from automl_ops import worker as worker_module
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.finalize import finalize
from automl_ops.job_tracking import archive_root, job_tracking_store
from automl_ops.jobs import JobRepository, WorkerLock, WorkerLockBusy, worker_running
from automl_ops.package import package_state, verify_package
from automl_ops.runner import RunSpec, run_pipeline, spec_to_config
from automl_ops.task import TaskConfig
from automl_ops.tracking import TrackingStore, describe_job
from automl_ops.train import AlgorithmRequest, TrainSettings
from automl_ops.worker import Worker, WorkerSettings, safe_message
from conftest import JobEnv, toy_bytes

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[1]
TIMING = {"seconds", "fold_seconds"}


def stripped(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: stripped(v) for k, v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [stripped(v) for v in value]
    return value


def make_spec(kind: str = "train", dataset_task: str = "classification", **over: Any) -> RunSpec:
    algorithms = (
        (AlgorithmRequest("logistic_regression"), AlgorithmRequest("random_forest"))
        if dataset_task == "classification"
        else (AlgorithmRequest("ridge"), AlgorithmRequest("random_forest"))
    )
    base: dict[str, Any] = {
        "kind": kind,
        "config": TaskConfig("y", dataset_task),  # type: ignore[arg-type]
        "requests": algorithms,
        "settings": TrainSettings(folds=3),
    }
    base.update(over)
    return RunSpec(**base)


def enqueue(
    env: JobEnv, spec: RunSpec | None = None, *, dataset: str = "toy", track: bool = False
) -> int:
    spec = spec or make_spec()
    version = env.store.get_version(f"{dataset}@v1")
    return (
        JobRepository(env.engine)
        .enqueue(
            kind=spec.kind,
            dataset_name=dataset,
            number=1,
            sha256=version.sha256,
            config=spec_to_config(spec),
            track=track,
            confirm_final_test=spec.kind == "final",
        )
        .id
    )


def make_worker(env: JobEnv, **kwargs: Any) -> Worker:
    return Worker(
        env.engine,
        env.store,
        env.models,
        settings=WorkerSettings(poll_seconds=0.05, heartbeat_seconds=0.2),
        **kwargs,
    )


def wait_until(check: Callable[[], bool], timeout: float = 60.0, step: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(step)
    return False


def tree_state(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under `root` with size and modification time (nothing is read)."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def cli(env: JobEnv, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "datasets.py"),
            "--store",
            str(env.store_root),
            "--models",
            str(env.root / "models-cli"),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def forbid_test_part(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        finalize_module, "_read_test_part", lambda *a, **k: pytest.fail("test part was read")
    )


# -- train and final jobs against the existing CLI ---------------------------------------------


def test_a_train_job_runs_to_completion_and_equals_the_cli(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_test_part(monkeypatch)  # a train job never reads the held-out test part
    job_id = enqueue(job_env)
    assert make_worker(job_env).run(once=True) == 1
    job = JobRepository(job_env.engine).get(job_id)
    assert (job.status, job.phase, job.error_code) == ("succeeded", "done", None)
    assert job.started_at and job.finished_at and job.tracking_state == "off"
    assert job.package_id is None and not job_env.models.exists()  # nothing was packaged
    assert job.result is not None and job.result["test_result"] is None
    assert job.result["selection"] is not None  # a selection report from CV numbers only

    done = cli(
        job_env, "train", "toy@v1", "--target", "y", "--task", "classification",
        "--algorithms", "logistic_regression,random_forest", "--folds", "3", "--json",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert payload["dataset_ref"] == job.result["dataset_ref"] == "toy@v1"
    assert payload["sha256"] == job.result["sha256"]
    expected = {k: v for k, v in payload.items() if k not in ("dataset_ref", "sha256")}
    assert stripped(job.result["training"]) == stripped(expected)


def test_a_final_job_equals_the_cli_and_reads_the_test_part_once(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = finalize_module._read_test_part

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(finalize_module, "_read_test_part", counting)
    job_id = enqueue(job_env, make_spec("final"))
    make_worker(job_env).run(once=True)
    job = JobRepository(job_env.engine).get(job_id)
    assert job.status == "succeeded", job.error_message
    assert calls == [1]  # exactly one read of the held-out part, for the selected model
    assert job.package_id and package_state(job_env.models / job.package_id) == "sealed"
    assert verify_package(job_env.models, job.package_id).ok
    assert job.result is not None and job.result["package_id"] == job.package_id

    done = cli(
        job_env, "finalize", "toy@v1", "--target", "y", "--task", "classification",
        "--algorithms", "logistic_regression,random_forest", "--folds", "3", "--confirm", "--json",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert calls == [1]  # the CLI ran in its own process; this process read the test part once
    assert stripped(job.result["selection"]) == stripped(payload["selection"])
    assert job.result["selection_sha256"] == payload["selection_sha256"]
    ours, theirs = job.result["test_result"], payload["test_result"]
    assert ours["evaluation"]["metrics"] == theirs["evaluation"]["metrics"]
    assert ours["limitation"] == theirs["limitation"]  # the cross-job reuse note is carried as is


def test_a_regression_job_works_too(job_env: JobEnv) -> None:
    job_id = enqueue(job_env, make_spec(dataset_task="regression"), dataset="toyreg")
    make_worker(job_env).run(once=True)
    job = JobRepository(job_env.engine).get(job_id)
    assert job.status == "succeeded" and job.result is not None
    assert {r["algorithm"] for r in job.result["training"]["results"]} == {"ridge", "random_forest"}


def test_progress_events_and_phases_are_recorded_in_order(job_env: JobEnv) -> None:
    job_id = enqueue(job_env)
    make_worker(job_env).run(once=True)
    repo = JobRepository(job_env.engine)
    events = repo.events(job_id)
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    kinds = [e.kind for e in events]
    assert kinds[0] == "phase" and events[0].payload == {"phase": "training"}
    assert kinds.count("algorithm_started") == 2 and kinds.count("algorithm_finished") == 2
    assert kinds.count("fold_finished") == 6  # 2 algorithms x 3 folds
    assert kinds[-1] == "state" and events[-1].payload["status"] == "succeeded"
    assert repo.get(job_id).progress is not None


def test_the_second_job_queues_behind_the_first(job_env: JobEnv) -> None:
    first, second = enqueue(job_env), enqueue(job_env)
    repo = JobRepository(job_env.engine)
    assert make_worker(job_env).run(once=True) == 2
    a, b = repo.get(first), repo.get(second)
    assert a.status == b.status == "succeeded"
    assert a.finished_at is not None and b.started_at is not None
    assert a.finished_at <= b.started_at  # never overlapped


# -- failures are recorded, scrubbed and never retried ------------------------------------------


def test_failures_are_recorded_with_codes_and_no_paths(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automl_ops.finalize import FinalizeError
    from automl_ops.train import TrainingError

    repo = JobRepository(job_env.engine)
    errors: list[Exception] = [
        TrainingError("the target is odd"),
        FinalizeError("cannot write /Users/someone/secret/models/m-1/manifest.json now"),
        RuntimeError("password=hunter2 at /private/tmp/x/y/z"),
    ]
    codes = []
    for error in errors:
        job_id = enqueue(job_env)

        def boom(*args: Any, error: Exception = error, **kwargs: Any) -> Any:
            raise error

        monkeypatch.setattr(worker_module, "run_pipeline", boom)
        make_worker(job_env).run(once=True)
        job = repo.get(job_id)
        assert job.status == "failed" and job.phase == "done" and job.finished_at
        codes.append(job.error_code)
        text = f"{job.error_code} {job.error_message}"
        assert "/Users/" not in text and "/private/" not in text and "hunter2" not in text
    assert codes == ["training_error", "finalize_error", "internal_error"]
    assert "<path>" in (repo.list_jobs(limit=2)[1].error_message or "")
    assert repo.counts() == {"failed": 3}  # nothing was re-queued or retried


def test_safe_message_removes_paths_and_bounds_length() -> None:
    assert safe_message("x /Users/a/b/c.txt y") == "x <path> y"
    assert "<path>" in safe_message("bad file '/var/folders/zz/tmp/file.csv'")
    assert len(safe_message("a" * 5000)) == 500
    assert safe_message("line one\n  line two") == "line one line two"


def test_a_changed_or_missing_dataset_fails_the_job_before_training(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    changed = enqueue(job_env)
    with job_env.engine.begin() as connection:
        from sqlalchemy import text

        connection.execute(
            text("UPDATE training_jobs SET dataset_sha256 = :s WHERE id = :i"),
            {"s": "a" * 64, "i": changed},
        )
    make_worker(job_env).run(once=True)
    job = repo.get(changed)
    assert (job.status, job.error_code) == ("failed", "dataset_changed")
    assert repo.events(changed) == [] or all(
        e.kind != "fold_finished" for e in repo.events(changed)
    )

    missing = enqueue(job_env)
    (job_env.store_root / "toy").rename(job_env.store_root / "toy-moved")
    make_worker(job_env).run(once=True)
    job = repo.get(missing)
    assert job.status == "failed" and job.error_code in ("dataset_missing", "dataset_unreadable")
    assert str(job_env.root) not in (job.error_message or "")


def test_the_result_is_never_truncated_a_too_large_one_fails_the_job(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_module, "RESULT_MAX_BYTES", 100)
    job_id = enqueue(job_env, make_spec("final"))
    make_worker(job_env).run(once=True)
    job = JobRepository(job_env.engine).get(job_id)
    assert (job.status, job.error_code) == ("failed", "result_too_large")
    assert job.result is None  # not stored in part
    assert job.package_id and package_state(job_env.models / job.package_id) == "sealed"


# -- cancel and stop ----------------------------------------------------------------------------


def patch_pipeline(monkeypatch: pytest.MonkeyPatch, before: Callable[[], object]) -> None:
    real = run_pipeline

    def patched(*args: Any, **kwargs: Any) -> Any:
        before()
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_module, "run_pipeline", patched)


def test_a_cancel_stops_a_train_job_without_recording_algorithm_failures(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = JobRepository(job_env.engine)
    job_id = enqueue(job_env)
    patch_pipeline(monkeypatch, lambda: repo.request_cancel(job_id))
    make_worker(job_env).run(once=True)
    job = repo.get(job_id)
    assert (job.status, job.phase) == ("cancelled", "done")
    assert job.result is None and job.error_code is None  # no algorithm "failed", nothing stored


def test_a_cancel_before_the_final_test_leaves_the_test_part_unread_and_no_package(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_test_part(monkeypatch)
    repo = JobRepository(job_env.engine)
    job_id = enqueue(job_env, make_spec("final"))
    patch_pipeline(monkeypatch, lambda: repo.request_cancel(job_id))
    make_worker(job_env).run(once=True)
    job = repo.get(job_id)
    assert job.status == "cancelled" and job.package_id is None
    assert not job_env.models.exists() or not any(job_env.models.iterdir())


def test_a_cancel_while_finalizing_is_refused_and_the_job_completes(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = JobRepository(job_env.engine)
    job_id = enqueue(job_env, make_spec("final"))
    seen: list[Any] = []
    from automl_ops import runner as runner_module

    real = finalize

    def asking(*args: Any, **kwargs: Any) -> Any:
        seen.append(repo.request_cancel(job_id))  # the API asks for a cancel mid-finalization
        return real(*args, **kwargs)

    monkeypatch.setattr(runner_module, "finalize", asking)
    make_worker(job_env).run(once=True)
    assert [(o.result, o.phase) for o in seen] == [("refused", "finalizing")]
    job = repo.get(job_id)
    assert job.status == "succeeded" and job.cancel_requested is False
    assert job.package_id and verify_package(job_env.models, job.package_id).ok


def test_a_queued_job_cancelled_before_the_worker_starts_never_runs(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    keep, drop = enqueue(job_env), enqueue(job_env)
    repo.request_cancel(drop)
    assert make_worker(job_env).run(once=True) == 1
    assert repo.get(keep).status == "succeeded" and repo.get(drop).status == "cancelled"
    assert repo.events(drop) == []


def test_stopping_the_worker_fails_the_running_job_before_the_test_part(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_test_part(monkeypatch)
    worker = make_worker(job_env)
    job_id = enqueue(job_env, make_spec("final"))
    patch_pipeline(monkeypatch, worker.request_stop)
    worker.run(once=True)
    job = JobRepository(job_env.engine).get(job_id)
    assert (job.status, job.error_code) == ("failed", "worker_stopped")


def test_cancel_latency_on_a_longer_job(job_env: JobEnv) -> None:
    """A real worker thread runs a longer job; the cancel lands at the next fold boundary."""
    big = CatalogedDatasetStore(job_env.store, job_env.engine)
    big.ingest("big", toy_bytes("classification", 4000), source_filename="big.csv")
    spec = make_spec(
        requests=(AlgorithmRequest("random_forest", {"n_estimators": 200}),),
        settings=TrainSettings(folds=5),
    )
    repo = JobRepository(job_env.engine)
    job_id = enqueue(job_env, spec, dataset="big")
    worker = make_worker(job_env)
    thread = threading.Thread(target=lambda: worker.run(once=True))
    thread.start()
    try:
        assert wait_until(lambda: any(e.kind == "fold_finished" for e in repo.events(job_id)))
        asked = time.monotonic()
        outcome = repo.request_cancel(job_id)
        assert outcome.result == "requested"
        assert wait_until(lambda: repo.get(job_id).status == "cancelled", 60)
        latency = time.monotonic() - asked
    finally:
        thread.join(timeout=120)
    assert latency < 30, latency  # bounded by one fold
    folds = [e for e in repo.events(job_id) if e.kind == "fold_finished"]
    assert 1 <= len(folds) < 5  # it did stop part-way
    print(f"\ncancel latency: {latency:.2f}s after {len(folds)} of 5 folds")


# -- heartbeat ----------------------------------------------------------------------------------


def test_the_heartbeat_keeps_running_while_the_job_computes(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = JobRepository(job_env.engine)
    job_id = enqueue(job_env)
    ages: list[float] = []

    def slow() -> None:
        time.sleep(1.5)  # long enough for several 0.2 s heartbeats
        ages.append(repo.get(job_id).heartbeat_age or 999.0)

    patch_pipeline(monkeypatch, slow)
    make_worker(job_env).run(once=True)
    assert ages and ages[0] < 1.0  # fresh although the job sat idle for 1.5 s
    assert not repo.get(job_id).stale


# -- recovery -----------------------------------------------------------------------------------


def test_a_new_worker_fails_orphans_then_runs_the_queue(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    orphan, waiting = enqueue(job_env), enqueue(job_env)
    assert repo.claim("w-dead") is not None  # a worker that then died
    assert make_worker(job_env).run(once=True) == 1
    assert repo.get(orphan).status == "failed" and repo.get(orphan).error_code == "worker_lost"
    assert repo.get(waiting).status == "succeeded"


def test_recovery_waits_for_the_lock_so_a_busy_lock_changes_nothing(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    orphan = enqueue(job_env)
    repo.claim("w-alive")
    holder = WorkerLock(job_env.engine)
    holder.acquire()  # another worker is genuinely running
    try:
        with pytest.raises(WorkerLockBusy):
            make_worker(job_env).run(once=True)
        assert repo.get(orphan).status == "running"  # not touched by a worker without the lock
    finally:
        holder.release()


def test_a_worker_lost_after_the_test_attempt_started_is_never_retried(
    job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process death right after the attempt marker exists: the job is failed, not re-run, and
    the test part is not read a second time by the next worker."""
    reads: list[int] = []

    def die_after_marker(*args: Any, **kwargs: Any) -> Any:
        reads.append(1)
        raise SystemExit(
            "simulated process death"
        )  # BaseException: nothing in the worker catches it

    monkeypatch.setattr(finalize_module, "_read_test_part", die_after_marker)
    repo = JobRepository(job_env.engine)
    job_id = enqueue(job_env, make_spec("final"))
    with pytest.raises(SystemExit):
        make_worker(job_env).run(once=True)
    assert repo.get(job_id).status == "running" and reads == [1]  # the row was left behind
    (package,) = [p.name for p in job_env.models.iterdir()]
    assert package_state(job_env.models / package) != "sealed"  # the attempt marker exists: spent

    later = enqueue(job_env)  # an ordinary job queued afterwards
    monkeypatch.undo()
    forbid_test_part(monkeypatch)
    assert make_worker(job_env).run(once=True) == 1
    lost = repo.get(job_id)
    assert (lost.status, lost.error_code) == ("failed", "worker_lost")
    assert repo.get(later).status == "succeeded"
    assert [p.name for p in job_env.models.iterdir()] == [package]  # no second package, no retry


# -- tracking is opt-in and never the archive ---------------------------------------------------


def test_tracking_is_off_unless_asked_for(job_env: JobEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> TrackingStore:
        pytest.fail("a job without track=true must not even look up a tracking store")

    job_id = enqueue(job_env)
    make_worker(job_env, tracking_store=refuse).run(once=True)
    job = JobRepository(job_env.engine).get(job_id)
    assert (
        job.status == "succeeded" and job.tracking_state == "off" and job.tracking_job_key is None
    )


def test_a_tracked_job_logs_to_a_separate_sqlite_directory_and_leaves_the_archive_alone(
    job_env: JobEnv,
) -> None:
    archive = archive_root()
    before = tree_state(archive) if archive.exists() else {}
    jobs_dir = job_env.root / "mlflow-jobs"
    store = job_tracking_store({"AUTOML_JOBS_MLFLOW_DIR": str(jobs_dir)})
    train_id = enqueue(job_env, track=True)
    final_id = enqueue(job_env, make_spec("final"), track=True)
    make_worker(job_env, tracking_store=lambda: store).run(once=True)
    repo = JobRepository(job_env.engine)
    for job_id in (train_id, final_id):
        job = repo.get(job_id)
        assert job.status == "succeeded" and job.tracking_state == "logged", job.tracking_error
        assert job.tracking_job_key and describe_job(job.tracking_job_key, store) is not None
        assert "tracking" in [e.payload.get("phase") for e in repo.events(job_id)]
    assert (jobs_dir / "mlflow.db").exists()
    after = tree_state(archive) if archive.exists() else {}
    assert after == before  # not a file added, removed, resized or rewritten


def test_a_tracked_job_is_refused_before_any_training_when_only_the_archive_is_available(
    job_env: JobEnv,
) -> None:
    archive = archive_root()
    before = tree_state(archive) if archive.exists() else {}
    job_id = enqueue(job_env, track=True)

    def archive_store() -> TrackingStore:
        return job_tracking_store({"AUTOML_JOBS_MLFLOW_DIR": str(archive)})

    make_worker(job_env, tracking_store=archive_store).run(once=True)
    job = JobRepository(job_env.engine).get(job_id)
    assert (job.status, job.error_code) == ("failed", "tracking_not_allowed")
    assert all(e.kind != "fold_finished" for e in JobRepository(job_env.engine).events(job_id))
    assert (tree_state(archive) if archive.exists() else {}) == before


def test_a_tracking_failure_leaves_the_job_succeeded_with_tracking_pending(
    job_env: JobEnv,
) -> None:
    blocked = job_env.root / "not-a-directory"
    blocked.write_text("a file where the tracking directory should be")
    store = TrackingStore(blocked / "mlflow", "sqlite")
    job_id = enqueue(job_env, track=True)
    make_worker(job_env, tracking_store=lambda: store).run(once=True)
    job = JobRepository(job_env.engine).get(job_id)
    assert job.status == "succeeded" and job.result is not None  # the computation is kept
    assert job.tracking_state == "pending" and job.tracking_error
    assert str(job_env.root) not in job.tracking_error


# -- real subprocesses ---------------------------------------------------------------------------


def worker_process(env: JobEnv, *extra: str) -> subprocess.Popen[str]:
    child_env = dict(os.environ)
    child_env["AUTOML_DATABASE_URL"] = env.url.render_as_string(hide_password=False)
    return subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "worker.py"),
            "--store",
            str(env.store_root),
            "--models",
            str(env.models),
            "--poll",
            "0.1",
            "--heartbeat",
            "0.5",
            *extra,
        ],
        env=child_env,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def long_spec() -> RunSpec:
    return make_spec(
        requests=(AlgorithmRequest("random_forest", {"n_estimators": 150}),),
        settings=TrainSettings(folds=5),
    )


def ingest_big(env: JobEnv) -> None:
    CatalogedDatasetStore(env.store, env.engine).ingest(
        "big", toy_bytes("classification", 4000), source_filename="big.csv"
    )


def test_a_worker_process_runs_the_queue_and_exits(job_env: JobEnv) -> None:
    first, second = enqueue(job_env), enqueue(job_env)
    process = worker_process(job_env, "--once")
    out, err = process.communicate(timeout=180)
    assert process.returncode == 0, err
    repo = JobRepository(job_env.engine)
    assert [repo.get(j).status for j in (first, second)] == ["succeeded", "succeeded"]
    assert "hunter" not in err and "POSTGRES_PASSWORD" not in err
    assert worker_running(job_env.engine) is False  # the lock went with the process


def test_a_second_worker_process_refuses_to_start_while_one_runs(job_env: JobEnv) -> None:
    first = worker_process(job_env)
    try:
        assert wait_until(lambda: worker_running(job_env.engine), 60)
        second = worker_process(job_env, "--once")
        _, err = second.communicate(timeout=60)
        assert second.returncode == 3 and "another worker" in err
    finally:
        first.send_signal(signal.SIGTERM)
        first.communicate(timeout=60)
    assert first.returncode == 0
    assert worker_running(job_env.engine) is False


def test_sigterm_stops_a_running_job_at_a_fold_boundary(job_env: JobEnv) -> None:
    ingest_big(job_env)
    repo = JobRepository(job_env.engine)
    job_id = enqueue(job_env, long_spec(), dataset="big")
    process = worker_process(job_env)
    try:
        assert wait_until(
            lambda: any(e.kind == "fold_finished" for e in repo.events(job_id)), 120
        ), "the job never started"
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0
    job = repo.get(job_id)
    assert (job.status, job.error_code) == ("failed", "worker_stopped")
    assert worker_running(job_env.engine) is False


def test_a_killed_worker_leaves_the_job_running_until_the_next_worker_marks_it_failed(
    job_env: JobEnv,
) -> None:
    ingest_big(job_env)
    repo = JobRepository(job_env.engine)
    victim = enqueue(job_env, long_spec(), dataset="big")
    after = enqueue(job_env)
    process = worker_process(job_env)
    try:
        assert wait_until(
            lambda: any(e.kind == "fold_finished" for e in repo.events(victim)), 120
        ), "the job never started"
        process.kill()  # SIGKILL: no cleanup of any kind
        process.communicate()
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == -signal.SIGKILL
    assert repo.get(victim).status == "running"  # the row cannot know yet
    assert wait_until(lambda: not worker_running(job_env.engine), 30)  # the server freed the lock

    second = worker_process(job_env, "--once")
    _, err = second.communicate(timeout=180)
    assert second.returncode == 0, err
    lost = repo.get(victim)
    assert (lost.status, lost.error_code) == ("failed", "worker_lost")
    assert repo.get(after).status == "succeeded"  # the queue moved on
    assert repo.counts() == {"failed": 1, "succeeded": 1}
