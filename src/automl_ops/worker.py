"""The job worker (P4-4): one process, one job at a time, restart-safe.

Life of a worker: take the single-worker lock (a second worker refuses to start), mark the jobs a
dead predecessor left `running` as failed (only the lock holder may do that), then poll for queued
jobs. A job runs in this process's main thread through the shared `runner.run_pipeline`; a small
thread sends heartbeats. Progress events are written as rows and double as the cancel check:
`record_progress` returns whether a cancel was requested, and the callback then raises
`JobCancelled` (a `BaseException`, see `runner.py`), so a cancel takes effect at the next fold
boundary.

The held-out test part is read only inside `finalize`, which `run_pipeline` calls for `final` jobs
after `enter_uncancellable("finalizing")` succeeded. Nothing here retries a job: a failed or lost
job stays failed, and a test attempt that was started is never repeated.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Engine

from automl_ops.algorithms import AlgorithmError
from automl_ops.dataset_store import (
    DatasetNotFound,
    DatasetStoreError,
    FileDatasetStore,
    VersionNotFound,
)
from automl_ops.finalize import FinalizeError
from automl_ops.job_tracking import TrackingNotAllowed, job_tracking_store
from automl_ops.jobs import (
    RESULT_MAX_BYTES,
    InvalidTransition,
    Job,
    JobRepository,
    WorkerLock,
    WorkerLockLost,
)
from automl_ops.package import PackageError
from automl_ops.pipeline import PipelineError
from automl_ops.profile import read_version_frame
from automl_ops.runner import (
    JobCancelled,
    RunOutcome,
    StopRun,
    WorkerStopping,
    job_result,
    run_pipeline,
    spec_from_config,
    tracking_record,
)
from automl_ops.selection import SelectionError
from automl_ops.tracking import TrackingStore, log_job, scrub
from automl_ops.train import ProgressEvent, TrainingError

log = logging.getLogger("automl_ops.worker")

POLL_SECONDS = 2.0
HEARTBEAT_SECONDS = 5.0
LOST_AFTER_SECONDS = 30.0  # heartbeats failing for this long: the job is given up
_MESSAGE_LIMIT = 500
_PATH = re.compile(r"(?:[A-Za-z]:)?(?:/[^\s'\"):,;]+){2,}")

_ERROR_CODES: tuple[tuple[type[BaseException], str], ...] = (
    (TrainingError, "training_error"),
    (AlgorithmError, "invalid_algorithm"),
    (SelectionError, "selection_error"),
    (FinalizeError, "finalize_error"),
    (PackageError, "package_error"),
    (PipelineError, "pipeline_error"),
    (DatasetNotFound, "dataset_missing"),
    (VersionNotFound, "dataset_missing"),
    (DatasetStoreError, "dataset_unreadable"),
)


def safe_message(text: str) -> str:
    """A message fit for the database and the API: no file paths, bounded length."""
    cleaned = _PATH.sub("<path>", " ".join(str(text).split()))
    return cleaned[:_MESSAGE_LIMIT]


def new_worker_id() -> str:
    return f"w-{os.getpid()}-{secrets.token_hex(4)}"


@dataclass
class WorkerSettings:
    poll_seconds: float = POLL_SECONDS
    heartbeat_seconds: float = HEARTBEAT_SECONDS
    lost_after_seconds: float = LOST_AFTER_SECONDS


class _Heartbeat(threading.Thread):
    """Keeps `heartbeat_at` fresh and notices when the job or the lock is no longer ours."""

    def __init__(
        self,
        repo: JobRepository,
        job_id: int,
        worker_id: str,
        interval: float,
        lost_after: float,
        lock_held: Callable[[], bool],
    ) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{job_id}")
        self.repo, self.job_id, self.worker_id = repo, job_id, worker_id
        self.interval, self.lost_after, self.lock_held = interval, lost_after, lock_held
        self.lost = threading.Event()
        self._done = threading.Event()

    def run(self) -> None:
        failing_for = 0.0
        while not self._done.wait(self.interval):
            try:
                if not (self.repo.heartbeat(self.job_id, self.worker_id) and self.lock_held()):
                    self.lost.set()
                    return
                failing_for = 0.0
            except Exception as error:
                failing_for += self.interval
                log.warning("heartbeat failed (%s)", type(error).__name__)
                if failing_for >= self.lost_after:
                    self.lost.set()
                    return

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=self.interval + 5)


class Worker:
    def __init__(
        self,
        engine: Engine,
        store: FileDatasetStore,
        models_root: Path,
        *,
        settings: WorkerSettings | None = None,
        tracking_store: Callable[[], TrackingStore] = job_tracking_store,
    ) -> None:
        self.engine = engine
        self.repo = JobRepository(engine)
        self.store = store
        self.models_root = models_root
        self.settings = settings or WorkerSettings()
        self.tracking_store = tracking_store
        self.worker_id = new_worker_id()
        self.lock = WorkerLock(engine)
        self._stop = threading.Event()

    # -- control -------------------------------------------------------------------------------

    def request_stop(self) -> None:
        """Finish nothing new: a running job stops at its next fold boundary (or, once the final
        test has started, when it completes); an idle worker leaves its loop."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run(self, *, once: bool = False) -> int:
        """Take the lock, recover orphans, then work. With `once`, return when the queue is empty.
        Returns the number of jobs processed. Raises `WorkerLockBusy` if another worker runs."""
        self.lock.acquire()
        done = 0
        try:
            recovered = self.lock.recover_orphans()
            if recovered:
                log.warning(
                    "marked %d job(s) left running by a dead worker as failed", len(recovered)
                )
            log.info("worker %s ready", self.worker_id)
            while not self.stopping:
                if not self.lock.held:
                    raise WorkerLockLost("the worker lock was lost")
                job = self.repo.claim(self.worker_id)
                if job is None:
                    if once:
                        break
                    self._stop.wait(self.settings.poll_seconds)
                    continue
                self.execute(job)
                done += 1
        finally:
            self.lock.release()
        return done

    # -- one job -------------------------------------------------------------------------------

    def execute(self, job: Job) -> None:
        heartbeat = _Heartbeat(
            self.repo,
            job.id,
            self.worker_id,
            self.settings.heartbeat_seconds,
            self.settings.lost_after_seconds,
            lambda: self.lock.held,
        )
        heartbeat.start()
        try:
            self._run_job(job, heartbeat)
        finally:
            heartbeat.stop()

    def _fail(self, job: Job, code: str, message: str, **extra: Any) -> None:
        log.warning("job %d failed: %s", job.id, code)
        self.repo.finish(
            job.id,
            self.worker_id,
            "failed",
            error_code=code,
            error_message=safe_message(message),
            **extra,
        )

    def _run_job(self, job: Job, heartbeat: _Heartbeat) -> None:
        jid, wid = job.id, self.worker_id
        try:
            spec = spec_from_config(job.kind, job.config)
            tracking: TrackingStore | None = None
            if job.track:
                try:
                    tracking = self.tracking_store()
                except TrackingNotAllowed as error:  # refused before any compute is spent
                    self._fail(job, "tracking_not_allowed", str(error))
                    return
            version, frame = read_version_frame(self.store, job.ref)
            if version.sha256 != job.dataset_sha256:
                self._fail(
                    job,
                    "dataset_changed",
                    f"{job.ref} no longer has the bytes it was submitted with",
                )
                return
            self.repo.set_phase(jid, wid, "training")

            def progress(event: ProgressEvent) -> None:
                if self.stopping or heartbeat.lost.is_set():
                    raise WorkerStopping()
                summary = {
                    "event": event.kind,
                    "algorithm": event.algorithm,
                    "algorithm_index": event.algorithm_index,
                    "algorithm_count": event.algorithm_count,
                    "fold": event.fold,
                    "folds": event.folds,
                }
                if self.repo.record_progress(jid, wid, event.kind, event.to_dict(), summary):
                    raise JobCancelled()

            def before_final() -> None:
                if self.stopping or heartbeat.lost.is_set():
                    raise WorkerStopping()  # still before the test attempt marker
                if not self.repo.enter_uncancellable(jid, wid, "finalizing"):
                    raise JobCancelled()

            outcome = run_pipeline(
                frame,
                version,
                spec,
                self.models_root,
                progress=progress,
                select=True,
                before_final=before_final,
            )
            self._complete(job, outcome, tracking)
        except JobCancelled:
            self.repo.finish(jid, wid, "cancelled")
        except WorkerStopping as stop:
            self._fail(job, stop.code, stop.message)
        except StopRun as stop:  # pragma: no cover - other stop reasons
            self._fail(job, stop.code, stop.message)
        except InvalidTransition:
            raise  # the job is not ours any more; nothing safe to write
        except Exception as error:
            code = next((c for t, c in _ERROR_CODES if isinstance(error, t)), None)
            if code is None:
                log.error("job %d: unexpected error", jid, exc_info=error)
                self._fail(
                    job, "internal_error", "An unexpected error occurred; see the worker log."
                )
            else:
                self._fail(job, code, str(error))

    def _complete(self, job: Job, outcome: RunOutcome, tracking: TrackingStore | None) -> None:
        jid, wid = job.id, self.worker_id
        result = job_result(outcome)
        package_id = None if outcome.final is None else outcome.final.package_id
        try:
            size = len(json.dumps(result, allow_nan=False))
        except ValueError:
            self._fail(job, "result_not_serializable", "The result contains a non-finite number.")
            return
        if size > RESULT_MAX_BYTES:
            self._fail(
                job,
                "result_too_large",
                f"The result is {size:,} bytes, above the limit of {RESULT_MAX_BYTES:,}; "
                "it was not stored.",
                package_id=package_id,
            )
            return
        state, key, problem = "off", None, None
        if tracking is not None:
            if not self.repo.enter_uncancellable(jid, wid, "tracking"):
                self.repo.finish(jid, wid, "cancelled")
                return
            try:
                key = log_job(tracking_record(outcome, self.models_root), tracking).job_key
                state = "logged"
            except Exception as error:  # the job's computation is complete; only logging is pending
                state, problem = "pending", scrub(safe_message(str(error) or type(error).__name__))
                log.warning("job %d: tracking is pending (%s)", jid, type(error).__name__)
        if not any(r.status == "ok" for r in outcome.training.results):
            self._fail(
                job,
                "no_algorithm_succeeded",
                "Every requested algorithm failed; see the result for each error.",
                result=result,
                package_id=package_id,
                tracking_state=state,
                tracking_job_key=key,
                tracking_error=problem,
            )
            return
        try:
            self.repo.finish(
                jid,
                wid,
                "succeeded",
                result=result,
                package_id=package_id,
                tracking_state=state,
                tracking_job_key=key,
                tracking_error=problem,
            )
        except InvalidTransition:
            # A cancel was accepted while the result was being made: honour it (the job's work is
            # discarded). Anything else means the job is not ours: let the caller stop.
            if not self.repo.cancel_requested(jid):
                raise
            self.repo.finish(jid, wid, "cancelled")
