# Experiment tracking with MLflow

BUILD_PLAN Steps 13 and 14 (Chunk 3A). Code: `src/automl_ops/tracking.py` (the only module that
imports `mlflow`); command line: `scripts/datasets.py` (`--track`, `track-package`),
`scripts/mlflow_ui.py`, `scripts/mlflow_smoke.py`. History, comparison and the reproducibility check
(Steps 15-16, Chunk 3B) read these records without writing to them: `docs/reproducibility.md`.

```bash
uv run python scripts/datasets.py train    adult-train-pool@v1 --target income --track
uv run python scripts/datasets.py finalize adult-train-pool@v1 --target income --confirm --track
uv run python scripts/datasets.py track-package m-20261113T050822Z-aa7bf39b   # retry / finish logging
uv run python scripts/mlflow_ui.py            # UI on http://127.0.0.1:5001, Ctrl-C to stop
uv run python scripts/mlflow_smoke.py         # Step 13 smoke test (own throwaway store)
uv run python scripts/datasets.py jobs                        # Step 15: history (read-only)
uv run python scripts/datasets.py compare KEY_A KEY_B         # Step 15: CV-only comparison
uv run python scripts/datasets.py reproduce KEY               # Step 16: re-run CV and compare
```

`--mlflow-dir` (before the command name) changes the SQLite store folder; the default is
`data/mlflow/` (git-ignored); `--mlflow-backend postgresql` selects the PostgreSQL store. Without
`--track` nothing is written and MLflow is never imported.

## The stores: SQLite (default, archive, tests) and PostgreSQL (P4-2b)

One `TrackingStore` class, two backends. **SQLite stays the default**; PostgreSQL is selected
explicitly with `--mlflow-backend postgresql` (before the command name) or `AUTOML_MLFLOW_BACKEND=
postgresql` in the environment or the git-ignored `.env`. Nothing switches by itself.

| | SQLite (default) | PostgreSQL |
| --- | --- | --- |
| metadata | `data/mlflow/mlflow.db` (`--mlflow-dir`) | database `mlflow` on the platform's PostgreSQL server (same credentials as `docs/database.md`; MLflow owns its schema and migrations; separate from `automl_ops`) |
| artifacts | `data/mlflow/artifacts/` | `data/mlflow-pg/artifacts/` (override: `AUTOML_MLFLOW_ARTIFACT_ROOT`, an absolute path) |
| created by | the first write | **only** `uv run python scripts/db.py mlflow-init` (idempotent). Reads, status checks and `--track` never create the database, schema or experiment. |
| used for | the historical archive, the tests, anything not yet moved | the platform's tracking from P4-2b on, once you choose it |

* Both paths are passed to MLflow explicitly; nothing relies on its implicit `./mlruns`. One
  experiment, `automl-ops`, is created with its artifact location.
* `uv run python scripts/db.py mlflow-status` (read-only) shows the schema revision, the experiment,
  its recorded artifact location, whether that matches the configured one, the number of runs, the
  longest run URI and the margin to the limit below.
* **The artifact URI limit.** MLflow's schema stores `runs.artifact_uri` in a `VARCHAR(200)`.
  PostgreSQL enforces that (SQLite does not). A run's URI is `<experiment artifact location>/<32
  character run id>/artifacts`, so the location may be at most **157** characters. Before any
  experiment or run is created on PostgreSQL the location is checked (`tracking.check_artifact_
  location`); an existing experiment is checked by its *stored* location, and a write is refused if
  that differs from the configured location (for example after the project folder was moved; reads
  still work). On this checkout the default location gives a longest run URI of **165** (35 to
  spare). The limit does not apply to SQLite, whose behaviour is unchanged.
* **The UI** is started by hand with `scripts/mlflow_ui.py [--backend postgresql]`, port 5001
  (5000 can be held by macOS AirPlay Receiver). For SQLite it runs `mlflow server --backend-store-uri
  <sqlite> --default-artifact-root <dir> --host 127.0.0.1 --port 5001`. For PostgreSQL the backend
  URI (which holds the password) and the artifact root are given in the server's *environment*
  (`MLFLOW_BACKEND_STORE_URI`, `MLFLOW_DEFAULT_ARTIFACT_ROOT`), not as arguments, so the password is
  not in the process list. `--print-command` shows the command and the variable *names* only.
* **Passwords:** the PostgreSQL URL is built on demand, held only inside the MLflow client of the
  process, never stored in an object that prints, and removed from tracking error messages.
* The files under `data/mlflow*/` contain the same private material as `data/models/` (the model
  and the 25 parity rows of training data). Keep them private; they are git-ignored.

## The historical SQLite archive and the migration (P4-2b)

`data/mlflow/` (SQLite database plus artifacts) is the **archive** of the Phase 3 history. It stays
where it is, unmodified (its stored artifact URIs are absolute paths into it, so it must not be
moved or renamed). Read it any time with explicit selection:

```bash
uv run python scripts/datasets.py --mlflow-backend sqlite jobs        # or just: jobs (SQLite is the default)
uv run python scripts/datasets.py --mlflow-backend sqlite --mlflow-dir data/mlflow job KEY
uv run python scripts/datasets.py --mlflow-backend postgresql jobs    # the PostgreSQL store
```

`scripts/tracking_migrate.py` (`backup`, `plan`, `relog`, `rerun`, `verify`) did the move; see its
docstring. The PostgreSQL records are **new records, not the original MLflow history**: they carry
provenance tags (`platform.origin`, `platform.migration=P4-2b`, and `platform.source_store` or
`platform.replaces_job_key`), the originals do not.

| original job (archive) | kind | PostgreSQL job | origin |
| --- | --- | --- | --- |
| `m-20261113T050754Z-bd711371` | finalized (Abalone) | **same key** | `relogged-from-package` |
| `m-20261113T050822Z-aa7bf39b` | finalized (Adult) | **same key** | `relogged-from-package` |
| `t-20261113T050838Z-a5c37b8d` | CV-only (Abalone, seed 42, 5 folds) | `t-20261113T073505Z-b8d179ab` | `rerun-from-archive` |
| `t-20261113T053314Z-49dd0ba3` | CV-only (Abalone, seed 7, 3 folds) | `t-20261113T073507Z-a5750c05` | `rerun-from-archive` |

Run ids for every job, original and new, are in `data/mlflow-pg/migration/mapping.json` and in
`DECISIONS.md` (D-034). A re-run job is a fresh run of the same configuration (bit-identical numbers
on this machine, verified), not the original run. A backup of the archive taken before the migration
is in `data/backups/mlflow-sqlite-20261113T073432Z/` (with `MANIFEST.json`).

**Rollback / not switching:** nothing uses the PostgreSQL store unless you select it. To go back,
remove `AUTOML_MLFLOW_BACKEND` from `.env` (or do not pass `--mlflow-backend postgresql`); the
archive, the backup, the sealed packages and the dataset store are untouched. The PostgreSQL
database and `data/mlflow-pg/` are kept for inspection; discarding them is a manual step that
needs your explicit decision (`DROP DATABASE mlflow` and removing that folder); nothing resets them
automatically.

## What a job looks like in MLflow

One **parent run** per job (`job <key>`) and one **child run** per algorithm (named after the
algorithm, linked with `mlflow.parentRunId` so the UI nests them).

| where | content |
| --- | --- |
| parent tags | `platform.job_key`, `platform.kind` (`train`/`final`), task, target, dataset ref and SHA-256, seed, selected algorithm, selection metric and SHA-256, package id, **test-row fingerprint** (`platform.test_rows_sha256`, so reuse of a test part can be queried later), `lib.<name>` library versions |
| parent params | seed, folds, test fraction, folds used, algorithm list |
| parent metrics | `training_seconds`, `selected_cv_<metric>_mean` (cross-validation) |
| parent artifacts | `training.json`, `provenance.json`, `selection_report.json` |
| child params | `param.<name>` for every algorithm parameter |
| child metrics | `cv_<m>_mean`, `cv_<m>_std`, `oof_<m>` (pooled out-of-fold), `fold_<m>` (one point per fold, `step` = fold), `fit_seconds`, `fold_fit_seconds` |
| child artifacts | `evaluation.json` (that algorithm's complete result) |
| child status | `FINISHED`, or `FAILED` with `platform.error_type` / `platform.error_message` |
| **selected** child only | `test_<m>` metrics (one point each), `test_result.json` (with the limitation text), `package/` with the six sealed files, tag `platform.final_test=logged` |

Undefined metrics (for example ROC-AUC when a fold lacks a class) are **skipped**, never logged as
zero. `train --track` runs carry **no** test metrics and no package. Non-selected models never have
test scores. The test result is logged once per package, on the selected model's run. Every logged
test result carries the standing limitation: the same test part can be reused by other jobs (an
**open decision**, unchanged here), repeated use can influence model choices, and the result is not
independent or leakage-proof. Nothing in tracking counts, warns about, or blocks that reuse.

Job identity: there is no application database yet, so a job is identified by `platform.job_key`:
the package id for a finalized job, `t-<UTC time>-<random>` for a `train --track` run. Phase 4 will
map these to real job ids.

## After sealing, and recoverable

* **Logging happens after the package is sealed.** `finalize --track` seals first, then logs. A
  tracking failure cannot unseal, spend or alter a package. The command then exits **3**
  ("TRACKING PENDING"), prints the cause and the retry command, and the package still passes
  `package-verify`.
* **The sealed package is self-sufficient.** `track-package <id>` rebuilds every run from
  `manifest.json` (all algorithms with parameters, status, errors, per-fold and pooled metrics,
  durations, the selection report and hash, dataset provenance, split fingerprints, library versions)
  and `test_result.json`. The seal hashes are verified first (`verify_package(deep=False)`); the model
  is **not** unpickled, no data row is read, nothing is retrained, no test row is touched. An open,
  spent or tampered package is refused. No extra tracking record was needed in the package.
* **Idempotent and resumable.** A job's runs are found by tag. An incomplete parent or child is
  **resumed**, not duplicated: metrics carry a fixed timestamp and step so re-logging the same point
  is not a new record, params and tags are rewritten with the same values, artifacts are re-copied
  only if missing or the wrong size, and each run's `platform.logging=complete` tag is its last
  write. The parent is completed only after every child reads back complete and the selected child
  holds the test result. A complete job refuses another attempt (`track-package` exits 2: "already
  tracked") and performs no write.
* `train --track` has no package to retry from; if its logging fails, an incomplete job remains in
  the store (visible as `platform.logging=incomplete`) and the training is simply run again.

## Telemetry

MLflow telemetry is off in every process this project starts: `MLFLOW_DISABLE_TELEMETRY=true`
(the documented switch), `DO_NOT_TRACK=true` and `MLFLOW_DISABLE_AGENT_HINT=1` are set in
`tracking.py` before MLflow is imported and are put in the environment of the UI/server process by
`scripts/mlflow_ui.py`. How it was checked: see the Chunk 3A entry in `DECISIONS.md`.

## Limits

* Two processes logging the *same* job at the same moment are not coordinated (single-user tool).
* SQLite is a single-writer development store, not a multi-user one.
* MLflow's server mode and remote artifact stores are not used here. The model registry and the
  `production` alias are used only by the opt-in registry mirror (P4-6, `docs/registry.md`), through
  additive `mirror_*` functions at the end of `tracking.py`; nothing above them was changed.
* The tracking record is a copy of what the sealed package says; the package stays the source of
  truth. A logged copy is checked against the package seal in the real-data demo, not on every run.
