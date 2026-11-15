# Training jobs and the worker (P4-4, BUILD_PLAN Step 19)

A job is one training run (`train`) or one training run followed by the single final test
(`final`), queued through the API and executed by a separate worker process. The API never trains.
Local, single-user, no authentication: see `docs/api.md`.

## Using it

```
make db-up && uv run python scripts/db.py upgrade     # once; the API and worker refuse an old schema
make api                                              # terminal 1: 127.0.0.1:8100
make worker                                           # terminal 2: scripts/worker.py
```

`POST /jobs` body (unknown fields are refused):

| Field | Default | Meaning |
| --- | --- | --- |
| `kind` | required | `train` (cross-validation only, never reads the test part) or `final` |
| `dataset`, `version`, `target` | required | the dataset version and target column |
| `task`, `excluded` | inferred, `[]` | as in `task-check` |
| `algorithms` | all for the task | list of `{key, params}`; see `GET /algorithms` |
| `test_fraction`, `seed`, `folds` | train settings defaults | validated like the CLI |
| `selection_metric` | task default | must be an allowed metric for the task |
| `confirm_final_test` | `false` | must be `true` for `kind: final` (otherwise 422 `confirmation_required`) |
| `track` | **`false`** | log to MLflow only when `true` (see Tracking) |

`GET /jobs/{id}?after_seq=N` returns the job, its progress events after `N`, and the result when
finished. `POST /jobs/{id}/cancel`: 200 a queued job was cancelled; 202 a running job will stop at
its next fold boundary; 409 `cannot_cancel_now` (finalizing or tracking) or `job_finished`.

## States

`queued -> running | cancelled`; `running -> succeeded | failed | cancelled`. Finished jobs never
change. Every change is one conditional `UPDATE ... WHERE status = <expected>` in
`automl_ops.jobs.JobRepository`, and a database trigger (`training_jobs_guard`) refuses a forbidden
move or any change to a finished job even from a hand-written statement. A partial unique index
allows at most one `running` job. The queue holds at most 10 unfinished jobs (409 `queue_full`);
the check runs under an advisory lock so concurrent submits cannot exceed it.

Phases while running: `loading`, `training`, `selecting`, `finalizing` (final jobs), `tracking`.

## Cancellation and the final test

Cancel is accepted only while a job is `queued` or in `loading`/`training`/`selecting`. The step
into `finalizing` (and `tracking`) is one conditional update that requires no cancel request, and a
cancel is one conditional update that requires a cancellable phase, so exactly one of them wins
(50 races tested, one winner each). Once a final job is `finalizing` the API refuses cancellation
and the worker finishes. The worker checks for a cancel at each fold boundary, so the delay is at
most one fold. The existing final-test attempt marker and spent-package rules are unchanged; a
finalization is **never retried automatically** after the marker exists.

## Worker

`scripts/worker.py` (`make worker`): holds a PostgreSQL session advisory lock, so only one worker
runs (a second exits with code 3; schema not current exits 2). Only the lock holder recovers
orphans: running jobs left by a dead worker are marked `failed` with `worker_lost` and are not
retried. A heartbeat thread updates the job every 5 s; a job whose heartbeat is older than 30 s is
reported `stale`. It polls every 2 s. SIGINT/SIGTERM stop it at the next fold boundary (job
`failed`, `worker_stopped`); a second signal exits at once. `GET /health` shows `worker`.

## Tracking (opt-in, D-038)

Off by default. With `track: true` the worker logs after the package is sealed, through
`tracking.py`. PostgreSQL MLflow is allowed when configured. SQLite is allowed only when
`AUTOML_JOBS_MLFLOW_DIR` names a separate directory; the original archive `data/mlflow/` is refused
(same path, inside it, symlink to it, or the same file) and the job fails `tracking_not_allowed`
before any compute. If logging fails after the compute, the job stays `succeeded` with
`tracking_state: pending`.

## Results

Results equal the CLI's: a test runs the real `train --json` and `finalize --json` and compares.
ROC/PR curves are thinned to at most 200 points (10 classes at most) and residuals to at most 1000
points, as library constants. A result over 8 MiB fails the job `result_too_large`; nothing is cut
silently.

## Measured (this machine, one live run; D-038)

| Item | Value |
| --- | --- |
| Stored JSONB result | 22 KB (adult train) to 154 KB (10-class, 20,000 rows, final); 100,000 x 100 regression: 24-62 KB |
| Cancel latency | 3.35 s (stopped after 2 of 5 folds, adult random forest) |
| Worker peak memory | 839 MB (kernel `ru_maxrss`) over all jobs incl. 100,000 x 100 |
| API peak memory | 419 MB (sampled) |
| Largest job | forest, 100,000 x 100: 142 s |
| Disk | models 15 MB, job MLflow dir 1.3 MB, database 8.9 MB |

## Limits

One worker, one machine, no priorities or retries. Cancel delay is bounded by one fold, and a long
single-fold fit cannot be interrupted. Nested library payloads in the result are passed through as
JSON, not typed field by field. A `train` job whose tracking is `pending` has no repair command.
Cross-job reuse of the test part is accepted for V1 (owner decision, D-057): repeated jobs may produce test results and every result keeps the warning that repeated use can make the test score less independent; a test-set lock is a possible future improvement; nothing here counts or restricts it.
Not run on Linux.
