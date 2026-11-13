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

`--mlflow-dir` (before the command name) changes the store folder; the default is `data/mlflow/`
(git-ignored). Without `--track` nothing is written to it and MLflow is never imported.

## The store (Phase 3 development store)

* Metadata: **SQLite**, `data/mlflow/mlflow.db`. Artifacts: a local folder, `data/mlflow/artifacts/`.
  Both paths are passed explicitly (`sqlite:////abs/path/mlflow.db`, `file:///abs/path/artifacts`),
  nothing relies on MLflow's implicit `./mlruns`.
* One experiment, `automl-ops`, created with that artifact location.
* This database is **separate from the future application database** (PostgreSQL, Phase 4). No
  migration or integration between the two is built or decided here.
* The UI is started by hand with `scripts/mlflow_ui.py`, which runs
  `mlflow server --backend-store-uri <sqlite> --default-artifact-root <dir> --host 127.0.0.1
  --port 5001`. Port 5001 because macOS AirPlay Receiver can hold 5000. `--print-command` shows the
  exact command and environment without starting anything.
* The files under `data/mlflow/` contain the same private material as `data/models/` (the model and
  the 25 parity rows of training data). Keep the folder private; it is git-ignored.

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
* MLflow's server mode, the model registry and aliases, remote artifact stores and PostgreSQL are not
  used here (registry: Steps 17 and 20).
* The tracking record is a copy of what the sealed package says; the package stays the source of
  truth. A logged copy is checked against the package seal in the real-data demo, not on every run.
