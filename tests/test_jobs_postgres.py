"""P4-4: the job repository, its state machine, the one-worker lock and orphan recovery, against a
real PostgreSQL server (throwaway `automl_test_*` databases; NOT RUN when the server is down)."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError, IntegrityError

from automl_ops.jobs import (
    DatasetMismatch,
    DatasetNotCatalogued,
    InvalidTransition,
    JobNotFound,
    JobRepository,
    QueueFull,
    WorkerLock,
    WorkerLockBusy,
    WorkerLockLost,
    worker_running,
)
from conftest import JobEnv

pytestmark = pytest.mark.postgres

CONFIG: dict[str, Any] = {
    "target": "y",
    "task": "classification",
    "excluded": [],
    "algorithms": [{"key": "logistic_regression", "params": {}}],
    "settings": {"test_fraction": 0.2, "seed": 42, "folds": 3},
    "selection_metric": None,
}


def sha_of(engine: Engine, name: str = "toy") -> str:
    with engine.connect() as connection:
        return str(
            connection.execute(
                text(
                    "SELECT v.sha256 FROM dataset_versions v "
                    "JOIN datasets d ON d.id = v.dataset_id "
                    "WHERE d.name = :n"
                ),
                {"n": name},
            ).scalar_one()
        )


def add(env: JobEnv, kind: str = "train", **over: Any) -> int:
    repo = JobRepository(env.engine)
    args: dict[str, Any] = {
        "kind": kind,
        "dataset_name": "toy",
        "number": 1,
        "sha256": sha_of(env.engine),
        "config": CONFIG,
        "track": False,
        "confirm_final_test": kind == "final",
    }
    args.update(over)
    return repo.enqueue(**args).id


def sql(engine: Engine, statement: str, **params: Any) -> Any:
    with engine.begin() as connection:
        return connection.execute(text(statement), params)


def status_of(engine: Engine, job_id: int) -> tuple[str, str]:
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT status, phase FROM training_jobs WHERE id = :i"), {"i": job_id}
        ).one()
    return row.status, row.phase


# -- submit and read ---------------------------------------------------------------------------


def test_enqueue_stores_a_queued_job_and_get_reads_it(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    job_id = add(job_env, track=True)
    job = repo.get(job_id)
    assert (job.kind, job.status, job.phase, job.ref) == ("train", "queued", "queued", "toy@v1")
    assert job.track is True and job.cancel_requested is False and job.config == CONFIG
    assert job.started_at is None and job.finished_at is None and job.result is None
    assert job.tracking_state == "off" and not job.stale
    with pytest.raises(JobNotFound):
        repo.get(job_id + 99)


def test_enqueue_refuses_unknown_or_mismatched_dataset_versions(job_env: JobEnv) -> None:
    with pytest.raises(DatasetNotCatalogued):
        add(job_env, dataset_name="nope")
    with pytest.raises(DatasetNotCatalogued):
        add(job_env, number=7)
    with pytest.raises(DatasetMismatch):
        add(job_env, sha256="0" * 64)


def test_the_queue_cap_holds_even_when_many_submit_at_once(job_env: JobEnv) -> None:
    for _ in range(10):
        add(job_env)
    with pytest.raises(QueueFull):
        add(job_env)
    # cancelling frees a place
    JobRepository(job_env.engine).request_cancel(1)
    add(job_env)

    # a fresh, empty queue hit by 25 concurrent submits accepts exactly the cap
    sql(job_env.engine, "DELETE FROM job_events")
    sql(job_env.engine, "DELETE FROM training_jobs")
    outcomes: list[str] = []

    def submit() -> None:
        try:
            add(job_env)
            outcomes.append("ok")
        except QueueFull:
            outcomes.append("full")

    threads = [threading.Thread(target=submit) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 10 and outcomes.count("full") == 15


def test_list_filters_and_paginates_newest_first(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    ids = [add(job_env) for _ in range(3)] + [add(job_env, "final")]
    listed = repo.list_jobs()
    assert [j.id for j in listed] == sorted(ids, reverse=True)
    assert all(j.result is None for j in listed)
    assert [j.kind for j in repo.list_jobs(kind="final")] == ["final"]
    assert repo.list_jobs(dataset="nope") == []
    page = repo.list_jobs(limit=2)
    rest = repo.list_jobs(limit=2, before_id=page[-1].id)
    assert [j.id for j in page + rest] == sorted(ids, reverse=True)
    assert repo.counts() == {"queued": 4}


# -- claiming and the one-running-job rule -----------------------------------------------------


def test_claim_is_first_in_first_out_and_one_at_a_time(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    first, second = add(job_env), add(job_env)
    claimed = repo.claim("w-a")
    assert claimed is not None and claimed.id == first
    assert (claimed.status, claimed.phase, claimed.worker_id) == ("running", "loading", "w-a")
    assert claimed.started_at is not None and claimed.heartbeat_at is not None
    assert repo.claim("w-a") is None  # the database allows one running job: the second waits
    assert status_of(job_env.engine, second) == ("queued", "queued")
    repo.finish(first, "w-a", "succeeded")
    nxt = repo.claim("w-a")
    assert nxt is not None and nxt.id == second
    assert JobRepository(job_env.engine).claim("w-b") is None  # still one at a time


def test_nothing_to_claim_returns_none(job_env: JobEnv) -> None:
    assert JobRepository(job_env.engine).claim("w-a") is None


def test_concurrent_claimers_get_exactly_one_job(job_env: JobEnv) -> None:
    for _ in range(3):
        add(job_env)
    won: list[int] = []

    def go(name: str) -> None:
        job = JobRepository(job_env.engine).claim(name)
        if job is not None:
            won.append(job.id)

    threads = [threading.Thread(target=go, args=(f"w-{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(won) == 1
    assert job_env_counts(job_env) == {"queued": 2, "running": 1}


def job_env_counts(env: JobEnv) -> dict[str, int]:
    return JobRepository(env.engine).counts()


def test_the_database_itself_refuses_a_second_running_job(job_env: JobEnv) -> None:
    a, b = add(job_env), add(job_env)
    start = (
        "UPDATE training_jobs SET status = 'running', phase = 'loading', worker_id = 'x', "
        "started_at = now() WHERE id = :i"
    )
    sql(job_env.engine, start, i=a)
    with pytest.raises(IntegrityError, match="one_running"):
        sql(job_env.engine, start, i=b)


# -- state transitions -------------------------------------------------------------------------


def test_a_job_finishes_once_and_only_by_its_own_worker(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    job_id = add(job_env)
    repo.claim("w-a")
    with pytest.raises(InvalidTransition):
        repo.finish(job_id, "w-other", "succeeded")
    repo.finish(job_id, "w-a", "succeeded", result={"ok": True})
    done = repo.get(job_id)
    assert done.status == "succeeded" and done.phase == "done" and done.result == {"ok": True}
    assert done.finished_at is not None
    with pytest.raises(InvalidTransition):
        repo.finish(job_id, "w-a", "succeeded")  # not completed twice
    with pytest.raises(InvalidTransition):
        repo.finish(job_id, "w-a", "failed", error_code="late")
    with pytest.raises(InvalidTransition):
        repo.set_phase(job_id, "w-a", "training")
    assert repo.heartbeat(job_id, "w-a") is False  # no heartbeat for a finished job
    assert repo.claim("w-a") is None  # a finished job is never claimed again
    assert repo.get(job_id).status == "succeeded"


def test_a_queued_job_cannot_be_finished_or_phased(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    job_id = add(job_env)
    with pytest.raises(InvalidTransition):
        repo.finish(job_id, "w-a", "succeeded")
    with pytest.raises(InvalidTransition):
        repo.set_phase(job_id, "w-a", "training")
    with pytest.raises(InvalidTransition):
        repo.enter_uncancellable(job_id, "w-a", "finalizing")
    assert status_of(job_env.engine, job_id) == ("queued", "queued")


def test_a_failed_job_needs_an_error_code(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    job_id = add(job_env)
    repo.claim("w-a")
    with pytest.raises(IntegrityError, match="failed_has_error"):
        repo.finish(job_id, "w-a", "failed")
    repo.finish(job_id, "w-a", "failed", error_code="training_error", error_message="boom")
    failed = repo.get(job_id)
    assert (failed.status, failed.error_code, failed.error_message) == (
        "failed",
        "training_error",
        "boom",
    )


def test_the_database_trigger_refuses_forbidden_moves_and_changes_to_finished_jobs(
    job_env: JobEnv,
) -> None:
    repo = JobRepository(job_env.engine)
    queued, running = add(job_env), add(job_env)
    # queued -> succeeded skips running
    with pytest.raises(DBAPIError, match="cannot move"):
        sql(
            job_env.engine,
            "UPDATE training_jobs SET status = 'succeeded', finished_at = now() WHERE id = :i",
            i=queued,
        )
    first = repo.claim("w-a")
    assert first is not None
    repo.finish(first.id, "w-a", "succeeded")
    # a finished job cannot go back to running or change at all
    with pytest.raises(DBAPIError, match="finished"):
        sql(
            job_env.engine,
            "UPDATE training_jobs SET status = 'running', finished_at = NULL, worker_id = 'x', "
            "started_at = now() WHERE id = :i",
            i=first.id,
        )
    with pytest.raises(DBAPIError, match="finished"):
        sql(job_env.engine, "UPDATE training_jobs SET progress = '{}' WHERE id = :i", i=first.id)
    # running -> queued is not a move either
    again = repo.claim("w-a")
    assert again is not None and again.id == running
    with pytest.raises(DBAPIError, match="cannot move"):
        sql(
            job_env.engine,
            "UPDATE training_jobs SET status = 'queued', worker_id = NULL WHERE id = :i",
            i=running,
        )


def test_a_final_job_must_carry_the_confirmation(job_env: JobEnv) -> None:
    with pytest.raises(IntegrityError, match="final_is_confirmed"):
        add(job_env, "final", confirm_final_test=False)


def test_progress_events_are_numbered_and_paged(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    job_id = add(job_env)
    repo.claim("w-a")
    for i in range(5):
        assert (
            repo.record_progress(job_id, "w-a", "fold_finished", {"fold": i}, {"fold": i}) is False
        )
    events = repo.events(job_id)
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert [e.payload["fold"] for e in repo.events(job_id, after_seq=3)] == [3, 4]
    assert len(repo.events(job_id, limit=2)) == 2
    assert repo.get(job_id).progress == {"fold": 4}
    with pytest.raises(InvalidTransition):
        repo.record_progress(job_id, "w-other", "fold_finished", {}, {})


# -- cancel ------------------------------------------------------------------------------------


def test_cancelling_a_queued_job_is_immediate_and_it_is_never_claimed(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    job_id = add(job_env)
    outcome = repo.request_cancel(job_id)
    assert outcome.result == "cancelled"
    cancelled = repo.get(job_id)
    assert cancelled.status == "cancelled" and cancelled.finished_at is not None
    assert repo.claim("w-a") is None
    assert repo.request_cancel(job_id).result == "finished"
    with pytest.raises(JobNotFound):
        repo.request_cancel(job_id + 50)


def test_cancelling_a_running_job_is_a_request_that_blocks_success(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    job_id = add(job_env, "final")
    repo.claim("w-a")
    repo.set_phase(job_id, "w-a", "training")
    assert repo.request_cancel(job_id).result == "requested"
    assert repo.record_progress(job_id, "w-a", "fold_finished", {}, {}) is True
    # the worker may no longer enter the final test or succeed
    assert repo.enter_uncancellable(job_id, "w-a", "finalizing") is False
    assert repo.get(job_id).phase == "training"
    with pytest.raises(InvalidTransition, match="cancel"):
        repo.finish(job_id, "w-a", "succeeded")
    repo.finish(job_id, "w-a", "cancelled")
    assert repo.get(job_id).status == "cancelled"


def test_after_finalizing_starts_a_cancel_is_refused(job_env: JobEnv) -> None:
    repo = JobRepository(job_env.engine)
    job_id = add(job_env, "final")
    repo.claim("w-a")
    assert repo.enter_uncancellable(job_id, "w-a", "finalizing") is True
    outcome = repo.request_cancel(job_id)
    assert (outcome.result, outcome.phase) == ("refused", "finalizing")
    assert repo.get(job_id).cancel_requested is False
    assert repo.enter_uncancellable(job_id, "w-a", "tracking") is True
    assert repo.request_cancel(job_id).result == "refused"
    repo.finish(job_id, "w-a", "succeeded")  # still allowed: no cancel was ever accepted


def test_cancel_and_entering_finalizing_never_both_win(job_env: JobEnv) -> None:
    """50 races: the worker enters `finalizing` while the API cancels, released together. Exactly
    one side wins each time, and the final state agrees with the winner."""
    repo = JobRepository(job_env.engine)
    results = {"final_test_started": 0, "cancel_accepted": 0}
    for _ in range(50):
        job_id = add(job_env, "final")
        repo.claim("w-a")
        repo.set_phase(job_id, "w-a", "training")
        barrier = threading.Barrier(2)
        seen: dict[str, Any] = {}

        def worker(
            job_id: int = job_id, barrier: threading.Barrier = barrier, seen: dict[str, Any] = seen
        ) -> None:
            barrier.wait()
            seen["entered"] = repo.enter_uncancellable(job_id, "w-a", "finalizing")

        def api(
            job_id: int = job_id, barrier: threading.Barrier = barrier, seen: dict[str, Any] = seen
        ) -> None:
            barrier.wait()
            seen["cancel"] = repo.request_cancel(job_id)

        threads = [threading.Thread(target=worker), threading.Thread(target=api)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        job = repo.get(job_id)
        if seen["entered"]:
            assert seen["cancel"].result == "refused" and job.cancel_requested is False
            assert job.phase == "finalizing"
            results["final_test_started"] += 1
            repo.finish(job_id, "w-a", "succeeded")
        else:
            assert seen["cancel"].result == "requested" and job.cancel_requested is True
            assert job.phase == "training"
            results["cancel_accepted"] += 1
            repo.finish(job_id, "w-a", "cancelled")
    assert sum(results.values()) == 50


# -- the one-worker lock and orphan recovery ---------------------------------------------------


def test_only_one_worker_lock_can_be_held_and_health_can_see_it(job_env: JobEnv) -> None:
    first, second = WorkerLock(job_env.engine), WorkerLock(job_env.engine)
    assert worker_running(job_env.engine) is False
    first.acquire()
    try:
        assert first.held and worker_running(job_env.engine) is True
        with pytest.raises(WorkerLockBusy):
            second.acquire()
        assert not second.held
    finally:
        first.release()
    assert worker_running(job_env.engine) is False
    second.acquire()  # free again once the holder let go
    second.release()


def test_a_worker_on_another_database_does_not_make_this_one_look_busy(
    job_env: JobEnv, pg_server: URL
) -> None:
    """Regression: `worker_running` once read `pg_locks` for the whole server, so a worker in a
    separate demo or test database made every other database report a running worker."""
    import secrets

    from sqlalchemy import create_engine

    from automl_ops.db.config import make_engine

    other = f"automl_test_{secrets.token_hex(6)}"
    admin = create_engine(pg_server.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{other}"'))
    other_engine = make_engine(pg_server.set(database=other))
    foreign = WorkerLock(other_engine)
    try:
        foreign.acquire()
        assert worker_running(other_engine) is True  # it is busy in its own database
        assert worker_running(job_env.engine) is False  # and invisible from this one
        own = WorkerLock(job_env.engine)
        own.acquire()  # this database's lock is independent of the other's
        try:
            assert worker_running(job_env.engine) is True
        finally:
            own.release()
    finally:
        foreign.release()
        other_engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{other}" WITH (FORCE)'))
        admin.dispose()


def test_recovery_marks_only_running_jobs_failed_and_only_for_the_lock_holder(
    job_env: JobEnv,
) -> None:
    repo = JobRepository(job_env.engine)
    orphan, waiting, cancelled = add(job_env), add(job_env), add(job_env)
    repo.request_cancel(cancelled)
    repo.claim("w-dead")  # claims `orphan` and never finishes it
    # a process that does not hold the lock cannot recover anything
    outsider = WorkerLock(job_env.engine)
    with pytest.raises(WorkerLockLost):
        outsider.recover_orphans()
    assert status_of(job_env.engine, orphan)[0] == "running"
    # nor can one whose lock was released
    holder = WorkerLock(job_env.engine)
    holder.acquire()
    holder.release()
    with pytest.raises(WorkerLockLost):
        holder.recover_orphans()
    assert status_of(job_env.engine, orphan)[0] == "running"

    holder.acquire()
    try:
        assert holder.recover_orphans() == [orphan]
    finally:
        holder.release()
    failed = repo.get(orphan)
    assert (failed.status, failed.error_code, failed.phase) == ("failed", "worker_lost", "done")
    assert "not retried" in (failed.error_message or "")
    assert status_of(job_env.engine, waiting) == ("queued", "queued")
    assert status_of(job_env.engine, cancelled)[0] == "cancelled"
    assert [e.kind for e in repo.events(orphan)] == ["state"]
    # recovery is repeatable and changes nothing the second time
    holder.acquire()
    try:
        assert holder.recover_orphans() == []
    finally:
        holder.release()
    claimed = repo.claim("w-new")
    assert claimed is not None and claimed.id == waiting
