# Model registry (P4-5, BUILD_PLAN Step 20, first part)

A **model** is a name; each **version** is one sealed model package (`data/models/<id>/`). PostgreSQL
is authoritative for a version's status and its history. MLflow only *mirrors* it, and only when
switched on (see "The MLflow mirror" below, P4-6); with the mirror off, MLflow holds no registered
models. Local, single-user,
no authentication (see `docs/api.md`).

## Statuses and moves

`development` (new) -> `staging` -> `production`; `archived` is the retired state.

| Move | Allowed | Event action |
| --- | --- | --- |
| register a package | always creates version N+1 in `development` | `register` |
| development -> staging | yes | `promote` |
| staging -> production | yes; the current production version is archived first, same transaction | `supersede` (old), `promote` (new) |
| archived -> staging | yes | `promote` |
| development or staging -> archived | yes | `archive` |
| development or staging -> production | **no** (no skipping staging) | |
| production -> archived directly | **no** (promote another version, or roll back) | |
| archived -> production | **only by rollback** | `rollback` |

**Rollback** (`POST /models/{name}/rollback`) is the one exception that returns an archived version
straight to production. It archives the current production version and restores the version it
replaced, as two events in one transaction. The previous version is read from the event history: the
current production version must have been *promoted* over another one (the event just before its
promotion is that version's `supersede`). If it cannot be established (no production version, the
first production version, a version that is itself a rollback restore, a previous version that is no
longer archived, or one whose package no longer verifies) the answer is **409 and nothing changes**.
A rollback is not undone by a second rollback; promote a version explicitly instead.

## Registration

`POST /models` with `name` and a `package_id`, a `job_id`, or both.

- A `job_id` must be a **succeeded** job that has a sealed package (only a `final` job makes one).
  If both ids are given they must name the same package (else 422 `provenance_mismatch`).
- The package is verified **deeply** first: seal hashes, read-only files, library versions, and a
  reload with a prediction-parity check. Open, spent, tampered or missing packages are refused (422
  `package_not_registerable`; problem texts never contain the real folder path).
- A package registers **once**, under any name (409 `package_already_registered`). A failed
  registration leaves no empty model behind.
- The version stores a **snapshot** of the package's own `test_result.json` (metrics, confusion matrix,
  per-class results, the limitation text). Curves and residual points are left in the package and
  named in `evaluation.omitted`, never cut silently. Detail reads use only the database.

## Event integrity (enforced by the database, migration 0003)

- A status changes **only** by inserting a `registry_events` row. A BEFORE INSERT trigger checks the
  move, allocates the event's `seq` under a per-model row lock (unique `(model_id, seq)`), and applies
  the new status. A direct `UPDATE ... SET status` is refused.
- A version without its `register` event cannot be committed (deferred constraint trigger).
- Events can never be updated, deleted or truncated. Versions and models are never deleted; a
  version is immutable apart from its status.
- A partial unique index allows at most one `production` version per model; a rollback restore must
  be preceded by its matching rollback archive.
- The repository checks the same rules first (clear errors, row lock held for the whole change), and
  a test replays every model's events and compares the result with the stored statuses.

## Endpoints

| Method and path | What it does |
| --- | --- |
| `POST /models` | Register a package (201) |
| `GET /models`, `GET /models/{name}` | Models, with their production and latest version numbers |
| `GET /models/{name}/versions` | Versions, newest first |
| `GET /models/{name}/versions/{n}` | One version with its stored evaluation summary |
| `POST /models/{name}/versions/{n}/promote` | `{to: staging\|production, reason}` |
| `POST /models/{name}/versions/{n}/archive` | `{reason}` for development or staging versions |
| `POST /models/{name}/rollback` | `{reason}`; returns the restored and the archived version |
| `GET /models/{name}/events?after_seq=` | The append-only history (200 per page) |

Error codes: `model_not_found`, `version_not_found`, `job_not_found` (404); `package_required`,
`provenance_mismatch`, `job_not_registerable`, `package_not_registerable`, `validation_error` (422);
`package_already_registered`, `invalid_transition`, `no_previous_production`, `package_unverified`
(409); `registry_unavailable` (503). Every change needs a `reason` of 1 to 500 characters (registration
defaults to "registered"). Model names follow the dataset-name rule (`a-z`, `0-9`, `_`, `-`; up to 64).

## The MLflow mirror (P4-6, D-042)

**Opt-in:** `AUTOML_REGISTRY_MIRROR=on` (default off). **PostgreSQL is the sole authority**; MLflow
is only a mirror and is never read to decide registry state. An MLflow failure never undoes or blocks
a registry operation: the change has already committed and the model just shows `pending`/`failed`.

**What is mirrored.** A platform-owned MLflow registered model (tag `platform.managed=true`) with one
MLflow version per registry version. Each version's source is `runs:/<run>/package`, the sealed
package logged by the existing resumable tracking (`log_job` of the package; seal hashes checked,
nothing unpickled, no test data read; a package that is not tracked yet is tracked automatically).
Every version carries the tags `platform.package_id`, `platform.registry_version`,
`platform.status` and `platform.selection_sha256`. The **only alias** is `production`, pointing at
the production version. MLflow models, versions and runs are **never deleted**.

**Backend rules (D6, unchanged).** PostgreSQL MLflow is allowed. SQLite only in the separate
directory named by `AUTOML_JOBS_MLFLOW_DIR`. The original archive `data/mlflow/` is refused (same
path, inside it, a symlink to it). Otherwise the sync fails with a clear message and writes nothing.

**Sync (state-based, idempotent).** A pass snapshots the desired state at event number N, applies it
step by step, reads MLflow back and compares. Only if the *complete* desired state is verified does
`registry_sync.synced_seq` advance to N (together with the verified state). Events that arrive during
a pass have a higher number and stay pending. A failed or interrupted pass changes nothing in the
progress row, and a rerun finds what is already in MLflow by the package tag (no duplicates). An
event is "pending sync" while `seq > synced_seq` (events themselves stay append-only).
One PostgreSQL session advisory lock per model (class 7266) means only one syncer works on a model;
a busy lock means skip. The lock vanishes with a crashed process.

**When it runs.** (1) Right after each successful registry change, best effort, at most 15 s;
(2) a background retry in the API every 30 s that reads the *persisted* pending work at start-up and
after every restart, with exponential backoff (30 s up to 15 min) per model; (3) `POST
/models/{name}/sync`; (4) `scripts/registry.py sync|check|status|reconcile`.

**Drift is reported, never silently accepted.** A sync first compares MLflow with what it expects (the
last verified state, the state of the last attempted pass, or the new desired state). Anything else is
**drift** (a moved `production` alias, an edited tag, a deleted version): the sync stops with `result:
drift`, the model shows `drifted`, and nothing is overwritten until an explicit reconcile. Idle models
are re-read every 10 minutes, so drift in a quiet model is shown too.

**Conflicts are never changed:** a model of the same name that the platform does not own, other
aliases, MLflow versions that are not registry versions, duplicate versions for one package, a version
whose package files are gone, and a `production` alias held by a version the platform does not own.
They are reported; a blocking conflict stops a sync, and a reconcile that meets one changes nothing.

**Check** (`GET /models/{name}/mirror/check`) is read-only: it writes nothing in MLflow or PostgreSQL.
**Reconcile** (`POST /models/{name}/reconcile`, `scripts/registry.py reconcile --apply`) needs a
reason of 1 to 500 characters; it resets only platform-owned tags, versions and the `production`
alias, verifies, and writes an audit row to the append-only `registry_sync_log` with what was
detected, what was intended, what was actually done and any failure.

| Endpoint | What it does |
| --- | --- |
| `GET /registry/sync` | Mirror standing of every model, and whether the mirror can run |
| `GET /models/{name}/sync` | One model: `state` (`disabled`, `in_sync`, `pending`, `failed`, `drifted`, `conflict`), `desired_seq`, `synced_seq`, attempts, last error |
| `POST /models/{name}/sync` | Sync now (`up_to_date`, `synced`, `failed`, `drift`, `conflict`, `busy`) |
| `GET /models/{name}/mirror/check` | The findings (`pending`, `drift`, `conflict`), read-only |
| `POST /models/{name}/reconcile` | `{reason}`; explicit, audited repair |

Existing model and version responses gained an additive `mirror` summary. Mirror errors:
`mirror_disabled` (409), `mirror_busy` (409).

**Measured (this machine; real API, real MLflow on PostgreSQL, throwaway databases; D-042).**
Registering and mirroring the two real packages (13.0 MB and 0.4 MB models): the first sync including
tracking about 2 s; a repeat sync with nothing new, no MLflow call; a `check` 0.08 to 0.8 s; a
reconcile of a moved alias 0.6 s. A package adds its own size once in MLflow's artifact store: the
two packages (13.56 MB and 0.50 MB on disk) became 15.14 MB of MLflow artifacts, and MLflow's database
11.2 MB. The new tables are tens of KB. API memory 271 to 285 MB peak (kernel `ru_maxrss`; 229 MB
idle in P4-5). During a real MLflow outage (connections to its database refused) registry changes
still answered 200 in 0.05 to 0.09 s with `mirror: failed`; after the outage the background retry
caught up in about 90 s because of the per-model backoff. A SIGKILL of the API mid-pass (after the
registered model existed, before its versions) left `synced_seq` at 0 and the next start finished it
with exactly one MLflow version per package.

**Limits.** A package that cannot be tracked or verified keeps its model `failed` (the registry is
unaffected). A model with an unresolved drift or conflict stays unsynced until it is fixed in MLflow
or reconciled. Foreign writers can still change MLflow between a read and a write; a pass reads back
to verify, and MLflow's own alias write can collide with a concurrent writer (retried a few times).
Not run: Linux; models larger than 13 MB; a mirror against an MLflow *server* (only the database
backends).

## The held-out test part

Registry code never reads dataset rows or re-evaluates anything. It reads only the sealed package's
`manifest.json` and `test_result.json`, after the package has verified. Tests assert that the registry
modules import nothing that reads data, and that registering, promoting and reading work with every
dataset reader and the test-part reader patched to fail. The standing limitation travels with the
evaluation: repeated use of the test part across jobs can influence model choices; it is not
independent or leakage-proof. The cross-job reuse policy is **still an open decision**; nothing here
counts or restricts it.

## Measured (this machine, one live run; D-040)

Real API process and the two real packages in `data/models` (read only), throwaway database.

| Item | Value |
| --- | --- |
| Register, deep verification of a 13 MB model | 0.13 s; the 0.4 MB model 0.04 s |
| Promote / archive / rollback | 7 to 25 ms |
| API memory | 229 MB idle, 273 MB peak (kernel `ru_maxrss`), so deep verification added about 44 MB |
| Version detail response | about 1.3 KB |
| Database | the three tables 40 KB, 112 KB and 48 KB after 2 versions and 9 events |

Deep verification loads the whole model, so memory grows with the model's size; only these two
packages were measured.

## Limits

One local database, no authentication, no deployments (P4-7). A package
is verified at registration and re-checked (seal hashes only) at each promotion, not continuously. A
model file rewritten together with its hashes would pass (see `docs/selection-and-packaging.md`).
Names are not renamable. Not run on Linux.
