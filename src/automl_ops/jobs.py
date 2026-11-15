"""Training-job state in PostgreSQL (P4-4): the queue, the state machine and the single worker.

Everything that changes a job is one conditional SQL statement, so two processes (the API and the
worker) can never both win a race:

* **Claim.** `UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1)`. A partial unique
  index (migration 0002) lets the database hold at most one `running` job.
* **Moves.** queued -> running | cancelled; running -> succeeded | failed | cancelled. Nothing else.
  Each move is `UPDATE ... WHERE status = <expected>`; zero rows means the move was not allowed
  and `InvalidTransition` is raised. A trigger in the database refuses the same moves again, and
  refuses every change to a finished job.
* **Cancel versus the final test.** A cancel is accepted only in the phases loading, training and
  selecting. Entering `finalizing` (or `tracking`) is one UPDATE that requires `NOT
  cancel_requested`, and a cancel is one UPDATE that requires a cancellable phase: whichever runs
  first wins, and the other sees zero rows. A succeeded job also requires `NOT cancel_requested`,
  so an accepted cancel always ends as `cancelled`.
* **One worker.** The worker holds a PostgreSQL session advisory lock for its whole life. Only the
  holder can run orphan recovery (`WorkerLock.recover_orphans`), and it does so on the very
  connection that holds the lock.

No job is ever retried automatically. Nothing here reads data rows, models or the test part.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError, IntegrityError

WORKER_LOCK = (7265, 2)  # (classid, objid) of the session advisory lock: "one worker"
QUEUE_LOCK = (7265, 3)  # transaction advisory lock around "count the queue, then insert"
QUEUE_LIMIT = 10
STALE_AFTER_SECONDS = 30.0  # a running job whose heartbeat is older is shown as stale
EVENT_PAGE = 500
RESULT_MAX_BYTES = 8 * 1024 * 1024  # a larger result fails the job (never truncated)
CANCELLABLE_PHASES = ("loading", "training", "selecting")
FINAL_STATES = ("succeeded", "failed", "cancelled")

Status = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobError(Exception):
    """Base class of the job repository's errors."""


class JobNotFound(JobError):
    pass


class QueueFull(JobError):
    pass


class DatasetNotCatalogued(JobError):
    """The dataset version is not in the PostgreSQL catalog (run `scripts/db.py sync`)."""


class DatasetMismatch(JobError):
    """The catalog and the file store disagree about the version's SHA-256."""


class InvalidTransition(JobError):
    """The job is not in a state that allows this change (for example, it already finished)."""


class WorkerLockBusy(JobError):
    """Another worker holds the lock."""


class WorkerLockLost(JobError):
    """This worker no longer holds the lock; it must not touch jobs."""


@dataclass(frozen=True)
class Job:
    id: int
    kind: str
    status: str
    phase: str
    dataset_name: str
    dataset_number: int
    dataset_sha256: str
    config: dict[str, Any]
    track: bool
    confirm_final_test: bool
    cancel_requested: bool
    progress: dict[str, Any] | None
    worker_id: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    heartbeat_at: datetime | None
    heartbeat_age: float | None  # seconds since the last heartbeat, measured by the database
    error_code: str | None
    error_message: str | None
    result: dict[str, Any] | None
    package_id: str | None
    tracking_state: str
    tracking_job_key: str | None
    tracking_error: str | None

    @property
    def ref(self) -> str:
        return f"{self.dataset_name}@v{self.dataset_number}"

    @property
    def stale(self) -> bool:
        """Running, but the worker has not been heard from for a while."""
        return (
            self.status == "running"
            and self.heartbeat_age is not None
            and self.heartbeat_age > STALE_AFTER_SECONDS
        )


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class CancelOutcome:
    """`cancelled`: a queued job was cancelled now. `requested`: the worker will stop it at the
    next fold boundary. `refused`: too late (`phase` says where it is) or already finished."""

    result: Literal["cancelled", "requested", "refused", "finished"]
    status: str
    phase: str


_COLUMNS = """
    j.id, j.kind, j.status, j.phase, d.name AS dataset_name, v.number AS dataset_number,
    j.dataset_sha256, j.config, j.track, j.confirm_final_test, j.cancel_requested, j.progress,
    j.worker_id, j.created_at, j.started_at, j.finished_at, j.heartbeat_at,
    EXTRACT(EPOCH FROM (now() - j.heartbeat_at))::float AS heartbeat_age,
    j.error_code, j.error_message, {result} AS result, j.package_id, j.tracking_state,
    j.tracking_job_key, j.tracking_error
"""
_FROM = """
    FROM training_jobs j
    JOIN dataset_versions v ON v.id = j.dataset_version_id
    JOIN datasets d ON d.id = v.dataset_id
"""


def _select(with_result: bool) -> str:
    return "SELECT " + _COLUMNS.format(result="j.result" if with_result else "NULL") + _FROM


def _job(row: Any) -> Job:
    return Job(**dict(row._mapping))


def _dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True)


class JobRepository:
    """All reads and writes of `training_jobs` and `job_events`."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # -- submit and read ----------------------------------------------------------------------

    def enqueue(
        self,
        *,
        kind: str,
        dataset_name: str,
        number: int,
        sha256: str,
        config: dict[str, Any],
        track: bool,
        confirm_final_test: bool,
        queue_limit: int = QUEUE_LIMIT,
    ) -> Job:
        """Add a queued job. The queue cap is checked under a lock, so concurrent submits cannot
        exceed it."""
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:a, :b)"),
                {"a": QUEUE_LOCK[0], "b": QUEUE_LOCK[1]},
            )
            version = connection.execute(
                text(
                    "SELECT v.id, v.sha256 FROM dataset_versions v "
                    "JOIN datasets d ON d.id = v.dataset_id WHERE d.name = :n AND v.number = :v"
                ),
                {"n": dataset_name, "v": number},
            ).first()
            if version is None:
                raise DatasetNotCatalogued(f"{dataset_name}@v{number} is not in the catalog")
            if version.sha256 != sha256:
                raise DatasetMismatch(f"the catalog and the store disagree about {dataset_name}")
            queued = connection.execute(
                text("SELECT count(*) FROM training_jobs WHERE status = 'queued'")
            ).scalar_one()
            if queued >= queue_limit:
                raise QueueFull(f"{queued} jobs are already waiting (the limit is {queue_limit})")
            job_id = connection.execute(
                text(
                    "INSERT INTO training_jobs (kind, dataset_version_id, dataset_sha256, config, "
                    "track, confirm_final_test) VALUES (:kind, :version, :sha, "
                    "CAST(:config AS jsonb), "
                    ":track, :confirm) RETURNING id"
                ),
                {
                    "kind": kind,
                    "version": version.id,
                    "sha": sha256,
                    "config": _dump(config),
                    "track": track,
                    "confirm": confirm_final_test,
                },
            ).scalar_one()
        return self.get(job_id)

    def get(self, job_id: int, *, with_result: bool = True) -> Job:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(_select(with_result) + " WHERE j.id = :id"), {"id": job_id}
            ).first()
        if row is None:
            raise JobNotFound(f"no job {job_id}")
        return _job(row)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        dataset: str | None = None,
        limit: int = 20,
        before_id: int | None = None,
    ) -> list[Job]:
        """Newest first, without the (large) result."""
        clauses = ["TRUE"]
        params: dict[str, Any] = {"limit": limit}
        for column, value, name in (
            ("j.status", status, "status"),
            ("j.kind", kind, "kind"),
            ("d.name", dataset, "dataset"),
        ):
            if value is not None:
                clauses.append(f"{column} = :{name}")
                params[name] = value
        if before_id is not None:
            clauses.append("j.id < :before")
            params["before"] = before_id
        query = (
            _select(False) + " WHERE " + " AND ".join(clauses) + " ORDER BY j.id DESC LIMIT :limit"
        )
        with self.engine.connect() as connection:
            return [_job(row) for row in connection.execute(text(query), params)]

    def events(self, job_id: int, after_seq: int = 0, limit: int = EVENT_PAGE) -> list[Event]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT seq, kind, payload, created_at FROM job_events "
                    "WHERE job_id = :id AND seq > :after ORDER BY seq LIMIT :limit"
                ),
                {"id": job_id, "after": after_seq, "limit": limit},
            )
            return [Event(r.seq, r.kind, r.payload, r.created_at) for r in rows]

    def counts(self) -> dict[str, int]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                text("SELECT status, count(*) AS n FROM training_jobs GROUP BY status")
            )
            return {r.status: int(r.n) for r in rows}

    # -- cancel (API side) ---------------------------------------------------------------------

    def request_cancel(self, job_id: int) -> CancelOutcome:
        """Cancel a queued job now, or ask the worker to stop a running one. Never moves a job
        that is past its cancel point (finalizing, tracking) or finished."""
        with self.engine.begin() as connection:
            queued = connection.execute(
                text(
                    "UPDATE training_jobs SET status = 'cancelled', phase = 'done', "
                    "cancel_requested = true, finished_at = now() "
                    "WHERE id = :id AND status = 'queued' RETURNING id"
                ),
                {"id": job_id},
            ).first()
            if queued is not None:
                return CancelOutcome("cancelled", "cancelled", "done")
            running = connection.execute(
                text(
                    "UPDATE training_jobs SET cancel_requested = true "
                    "WHERE id = :id AND status = 'running' AND phase = ANY(:phases) RETURNING phase"
                ),
                {"id": job_id, "phases": list(CANCELLABLE_PHASES)},
            ).first()
            if running is not None:
                return CancelOutcome("requested", "running", running.phase)
            row = connection.execute(
                text("SELECT status, phase FROM training_jobs WHERE id = :id"), {"id": job_id}
            ).first()
        if row is None:
            raise JobNotFound(f"no job {job_id}")
        if row.status in FINAL_STATES:
            return CancelOutcome("finished", row.status, row.phase)
        return CancelOutcome("refused", row.status, row.phase)

    # -- the worker's side ---------------------------------------------------------------------

    def claim(self, worker_id: str) -> Job | None:
        """Take the oldest queued job and make it `running`; None if there is none, or if a job is
        already running (the one-running-job index refused the second)."""
        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    text(
                        "UPDATE training_jobs SET status = 'running', phase = 'loading', "
                        "worker_id = :w, started_at = now(), heartbeat_at = now() "
                        "WHERE id = (SELECT id FROM training_jobs WHERE status = 'queued' "
                        "ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING id"
                    ),
                    {"w": worker_id},
                ).first()
        except IntegrityError:
            return None
        return None if row is None else self.get(row.id, with_result=False)

    def heartbeat(self, job_id: int, worker_id: str) -> bool:
        """False if this worker no longer owns a running job `job_id`."""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    "UPDATE training_jobs SET heartbeat_at = now() "
                    "WHERE id = :id AND worker_id = :w AND status = 'running' RETURNING id"
                ),
                {"id": job_id, "w": worker_id},
            ).first()
        return row is not None

    def set_phase(self, job_id: int, worker_id: str, phase: str) -> None:
        """Move to a phase that can still be cancelled (loading, training, selecting)."""
        if phase not in CANCELLABLE_PHASES:
            raise ValueError(f"{phase!r} is entered with enter_uncancellable")
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    "UPDATE training_jobs SET phase = :p, heartbeat_at = now() "
                    "WHERE id = :id AND worker_id = :w AND status = 'running' RETURNING id"
                ),
                {"id": job_id, "w": worker_id, "p": phase},
            ).first()
            if row is None:
                raise InvalidTransition(f"job {job_id} is not running under this worker")
            self._event(connection, job_id, "phase", {"phase": phase})

    def enter_uncancellable(self, job_id: int, worker_id: str, phase: str) -> bool:
        """Atomically enter `finalizing` or `tracking`. False when a cancel was requested first
        (the caller must then stop and finish the job as cancelled). After True, `request_cancel`
        refuses for good."""
        if phase not in ("finalizing", "tracking"):
            raise ValueError("only finalizing and tracking are past the cancel point")
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    "UPDATE training_jobs SET phase = :p, heartbeat_at = now() "
                    "WHERE id = :id AND worker_id = :w AND status = 'running' "
                    "AND NOT cancel_requested RETURNING id"
                ),
                {"id": job_id, "w": worker_id, "p": phase},
            ).first()
            if row is not None:
                self._event(connection, job_id, "phase", {"phase": phase})
                return True
            state = connection.execute(
                text(
                    "SELECT status, worker_id, cancel_requested FROM training_jobs WHERE id = :id"
                ),
                {"id": job_id},
            ).first()
        if state is None or state.status != "running" or state.worker_id != worker_id:
            raise InvalidTransition(f"job {job_id} is not running under this worker")
        return False

    def record_progress(
        self,
        job_id: int,
        worker_id: str,
        kind: str,
        payload: dict[str, Any],
        summary: dict[str, Any],
    ) -> bool:
        """Store one progress event and the one-line summary; returns `cancel_requested`."""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    "UPDATE training_jobs SET progress = CAST(:summary AS jsonb), "
                    "heartbeat_at = now() "
                    "WHERE id = :id AND worker_id = :w AND status = 'running' "
                    "RETURNING cancel_requested"
                ),
                {"id": job_id, "w": worker_id, "summary": _dump(summary)},
            ).first()
            if row is None:
                raise InvalidTransition(f"job {job_id} is not running under this worker")
            self._event(connection, job_id, kind, payload)
        return bool(row.cancel_requested)

    def cancel_requested(self, job_id: int) -> bool:
        with self.engine.connect() as connection:
            value = connection.execute(
                text("SELECT cancel_requested FROM training_jobs WHERE id = :id"), {"id": job_id}
            ).scalar_one_or_none()
        return bool(value)

    def finish(
        self,
        job_id: int,
        worker_id: str,
        status: Literal["succeeded", "failed", "cancelled"],
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        result: dict[str, Any] | None = None,
        package_id: str | None = None,
        tracking_state: str = "off",
        tracking_job_key: str | None = None,
        tracking_error: str | None = None,
    ) -> None:
        """running -> a final state, once. A `succeeded` move is refused when a cancel was
        accepted (the caller then finishes the job as cancelled)."""
        if status not in FINAL_STATES:
            raise ValueError(f"{status!r} is not a final state")
        guard = " AND NOT cancel_requested" if status == "succeeded" else ""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    "UPDATE training_jobs SET status = :s, phase = 'done', finished_at = now(), "
                    "error_code = :ec, error_message = :em, result = CAST(:result AS jsonb), "
                    "package_id = :pk, tracking_state = :ts, tracking_job_key = :tk, "
                    "tracking_error = :te "
                    "WHERE id = :id AND worker_id = :w AND status = 'running'"
                    + guard
                    + " RETURNING id"
                ),
                {
                    "s": status,
                    "ec": error_code,
                    "em": error_message,
                    "result": None if result is None else _dump(result),
                    "pk": package_id,
                    "ts": tracking_state,
                    "tk": tracking_job_key,
                    "te": tracking_error,
                    "id": job_id,
                    "w": worker_id,
                },
            ).first()
            if row is None:
                raise InvalidTransition(
                    f"job {job_id} cannot become {status}: it is not running under this worker"
                    + (" or a cancel was accepted" if status == "succeeded" else "")
                )
            self._event(connection, job_id, "state", {"status": status, "error_code": error_code})

    @staticmethod
    def _event(connection: Connection, job_id: int, kind: str, payload: dict[str, Any]) -> None:
        connection.execute(
            text(
                "INSERT INTO job_events (job_id, seq, kind, payload) VALUES (:id, "
                "(SELECT COALESCE(max(seq), 0) + 1 FROM job_events WHERE job_id = :id), "
                ":kind, CAST(:payload AS jsonb))"
            ),
            {"id": job_id, "kind": kind, "payload": _dump(payload)},
        )


class WorkerLock:
    """The single-worker lock, held on a dedicated connection for the worker's whole life."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._connection: Connection | None = None

    def acquire(self) -> None:
        if self._connection is not None:
            raise WorkerLockBusy("this process already holds the worker lock")
        connection = self.engine.connect()
        try:
            got = connection.execute(
                text("SELECT pg_try_advisory_lock(:a, :b)"),
                {"a": WORKER_LOCK[0], "b": WORKER_LOCK[1]},
            ).scalar_one()
            connection.commit()
        except BaseException:
            connection.close()
            raise
        if not got:
            connection.close()
            raise WorkerLockBusy("another worker is already running")
        self._connection = connection

    @property
    def held(self) -> bool:
        """True while the lock is really held by our connection (asked of the server)."""
        connection = self._connection
        if connection is None:
            return False
        try:
            value = connection.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND "
                    "classid::bigint = :a AND objid::bigint = :b AND objsubid = 2 "
                    "AND pid = pg_backend_pid() AND granted)"
                ),
                {"a": WORKER_LOCK[0], "b": WORKER_LOCK[1]},
            ).scalar_one()
            connection.commit()
            return bool(value)
        except DBAPIError:
            return False

    def recover_orphans(self) -> list[int]:
        """Mark every `running` job `failed` (worker_lost). Only the lock holder may call this:
        with the lock held, no other worker exists, so a running row is an orphan of a worker
        that died. Runs on the lock's own connection."""
        if not self.held:
            raise WorkerLockLost("recovery needs the worker lock")
        connection = self._connection
        assert connection is not None
        rows = connection.execute(
            text(
                "UPDATE training_jobs SET status = 'failed', phase = 'done', finished_at = now(), "
                "error_code = 'worker_lost', error_message = 'The worker stopped while this job "
                "was running; it was not retried.' WHERE status = 'running' RETURNING id"
            )
        ).all()
        ids = sorted(int(r.id) for r in rows)
        for job_id in ids:
            JobRepository._event(
                connection, job_id, "state", {"status": "failed", "error_code": "worker_lost"}
            )
        connection.commit()
        return ids

    def release(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:a, :b)"),
                    {"a": WORKER_LOCK[0], "b": WORKER_LOCK[1]},
                )
                connection.commit()
            except DBAPIError:
                pass
            finally:
                connection.close()


def worker_running(engine: Engine) -> bool:
    """Is some worker holding the lock in THIS database right now? (For `/health`; takes no lock
    itself.) A worker connected to another database on the same server does not count."""
    with engine.connect() as connection:
        return bool(
            connection.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND "
                    "classid::bigint = :a AND objid::bigint = :b AND objsubid = 2 AND granted "
                    # advisory locks belong to one database; pg_locks lists the whole server, so a
                    # worker on another database (a demo or test stack) must not count
                    "AND database = (SELECT oid FROM pg_database WHERE datname = "
                    "current_database()))"
                ),
                {"a": WORKER_LOCK[0], "b": WORKER_LOCK[1]},
            ).scalar_one()
        )
