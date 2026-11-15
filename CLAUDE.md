# CLAUDE.md

Rulebook for Claude Code on this project: short, stable, read every session. Edit in place; keep
history in `DECISIONS.md`, not here. `project-notes/` is git-ignored (local only).

# AutoML Ops (Automated ML Lifecycle Management Platform)

## What this is
A single-user, local web platform that manages the ML lifecycle for tabular CSV data: dataset
validation, multi-model training, MLflow experiment tracking, evaluation, a model registry,
REST deployment and monitoring. Solo portfolio project for entry-level AI/ML engineering roles.
It demonstrates MLOps engineering, not a new prediction app.

Planning documents (approved as **working guides, not fixed contracts**; do not copy them here):
`SCOPE.md` (boundaries), `project-notes/PROJECT_GUIDE.md` (what and why),
`project-notes/BUILD_PLAN.md` (ordered steps and gates), `project-notes/ARCHITECTURE.md` (design).
Easy-reading step notes for interview prep: `project-notes/BUILD_LOG.md`.

## Current state
Steps 1 (scaffold), 2 (demo data), 3 (CSV ingestion and dataset versions), 4 (profiling and
validation: 4A, and 4B the Pandera feature schema), 5 (task configuration) and 6 (exploration data)
are done and verified, and **Gate 1 is closed** (owner-approved 2026-11-13). **Chunk 2A (Steps 7, 8,
10 and 9: preprocessing pipeline, algorithm catalogue, metrics, split and cross-validation) is done
and verified (owner-approved complete)**. **Chunk 2B (Steps 11-12: selection, the single final test,
model packaging) is done and verified**, and **Gate 2 is closed** (owner-approved, including the
two-process concurrent-finalization check, D-022). **Chunk 3A (Steps 13-14: MLflow setup and
logging of training runs through `tracking.py`) is done and verified, and Gate 3A is closed** (owner-approved; no further changes to 3A). **Chunk 3B (Steps 15-16: experiment history and CV-only comparison, and the read-only reproducibility check) is done and verified, and Gate 3 is closed** (owner-approved 2026-11-13); **Phase 4 is approved as chunks P4-1 to P4-10 (D-028); P4-1 (PostgreSQL foundation and dataset catalog, D-029) is closed; P4-2b (MLflow tracking on PostgreSQL, D-033/D-034) is complete and its gate is approved (D-035); P4-3 (control-plane skeleton and Dataset API, D-036) is complete and its gate is approved (D-037; the httpx/Starlette test-client deprecation warning is a known, non-blocking limitation); **P4-4 (training jobs, worker process, Job API; D-038) is complete and its gate is approved (D-039); the real `automl_ops` database was seen at revision 0003 in a read-only check on 2026-11-14 (the owner runs every upgrade themselves; I never do); **P4-5 (PostgreSQL-only model registry; D-040, `docs/registry.md`) is complete and its gate is approved (D-041); **P4-6 (opt-in MLflow mirror and reconcile of the registry; D-042, `docs/registry.md`) is complete and its gate is approved (D-043; migration 0004 applied by the owner); P4-7 (serving app and deployments, Steps 21-22; D-043, `docs/serving.md`) is complete and its gate is approved (D-044; migration 0005 applied by the owner); P4-8 (feedback labels and live performance, Step 23; D-044, `docs/serving.md`) is complete and its gate is approved (D-045; migration 0006 applied by the owner); P4-9 (drift monitoring, reference samples and alerts, Step 24; D-045, `docs/monitoring.md`) is complete and its gate is approved (migration 0007 applied by the owner); **P4-10 (traffic simulator, API quality pass and the backend end-to-end story, Steps 25, 26 and the backend half of 35; D-046, `docs/simulator.md`) is complete and its gate is approved (owner, 2026-11-14); two owner-approved follow-ups followed (D-047): the performance-drop alert now scores the labels inside the same 1,000-prediction window as drift (seeded bootstrap guard; verdicts triggered, clear, inconclusive), and `make e2e` runs on a throwaway stack made by `scripts/e2e_isolated.py` (new database `automl_test_e2e_*`, temp folders, SQLite MLflow) while `scripts/e2e_demo.py` refuses, before any write, unless both services report isolation on `/health`; Evidently is NOT used (owner decision, D-046): `monitor.py` stays numpy-only. **Phase 5 (frontend, Steps 27-32; D-048, `docs/frontend.md`, `frontend/`) is built and verified and awaits the owner's Gate 5 review** (UI on http://localhost:5173 via `make ui-dev`; `make ui-demo` runs it on a throwaway seeded backend; `make ui-check` and `make ui-e2e`; two additive read-only endpoints `GET /deployments/{id}/predictions` and `/activity`). **Phase 6 (containers, full Compose, CI, final docs, sign-off) is not started and needs its own proposal.** For P4-6 only, the owner authorized additive changes to the otherwise sealed `tracking.py`. SQLite is still the default tracking backend (the old store `data/mlflow` is the untouched archive).** See `DECISIONS.md` and `BUILD_LOG.md`. Runtime dependencies: pandas, numpy, pandera,
scikit-learn, joblib, mlflow, psycopg[binary], sqlalchemy, alembic, fastapi, uvicorn, starlette, pydantic, python-multipart. Dev: pytest, ruff, mypy, pandas-stubs, httpx.
Code: `scripts/ui_stack.py` (D-048: throwaway, optionally seeded backend for the UI and its browser tests), `isolation.py` and `e2e_stack.py` (D-047), `simulator.py`, `e2e.py`, `scripts/simulate_traffic.py` and `scripts/e2e_demo.py` (P4-10: `make simulate`, `make e2e`; the simulator accepts only the known demo monitoring files, never training data, always `simulated: true`, localhost only), `monitor.py`, `reference.py` and `monitoring.py` (P4-9, control plane only, never imported by serving), `feedback.py` (P4-8), `deployments.py`, `serving_client.py`, `api_serving/` and `scripts/serve_model.py` (P4-7: `make serve` on 127.0.0.1:8200, setting `AUTOML_SERVING_PORT`; `api_control/deployments.py`), `registry_mirror.py` and `scripts/registry.py` (P4-6: setting `AUTOML_REGISTRY_MIRROR`, off by default), `registry.py` (P4-5: statuses, rollback, append-only events enforced by migration 0003 triggers), `jobs.py`, `worker.py`, `runner.py` (shared by CLI and worker) and `job_tracking.py` (P4-4: `docs/jobs.md`; `make worker`; settings `AUTOML_JOBS_MLFLOW_DIR`, `AUTOML_MODELS_DIR`; job tracking is opt-in), `api_control/` and `dataset_reports.py` (P4-3: Dataset API on 127.0.0.1:8100, `docs/api.md`), `db/` (P4-1: config, models, migrate, catalog; `docs/database.md`), `demo_data.py` (Step 2), `ingest.py` and `dataset_store.py` (Step 3), `profile.py` and
`validate.py` (4A), `task.py` (5), `explore.py` (6), `schema.py` (4B), `pipeline.py` (7),
`algorithms.py` (8), `evaluate.py` (10), `train.py` (9), `selection.py` and `finalize.py` (11),
`package.py` (12), `tracking.py` (13-14, plus the 3B read side; the only module that imports `mlflow`), `compare.py` and `reproduce.py` (15-16, no MLflow), `hashing.py`, and
the scripts `prepare_demo_data.py`, `datasets.py`, `db.py`, `mlflow_smoke.py`, `mlflow_ui.py`, `serve_control.py`, `worker.py` and `export_openapi.py`. Demo data (UCI Adult and Abalone, CC BY 4.0) is prepared by
`make demo-data` into git-ignored `data/`; see `docs/data.md`. Dataset versions live in git-ignored
`data/store/` (`docs/ingestion.md`); profile, validate, task, explore and schema results are computed
on demand and never stored (`docs/profiling.md`, `docs/task-config.md`, `docs/exploration.md`,
`docs/schema.md`); `train` is cross-validation only and never reads the held-out test part
(`docs/training.md`); only `finalize` reads it, once per package, for the selected model, after the
frozen selection is verified and the attempt marker exists (`docs/selection-and-packaging.md`). Model
packages live in git-ignored `data/models/`; only sealed packages load. Tracking (`--track`,
`track-package`, `docs/tracking.md`) logs to a git-ignored SQLite + local-artifact MLflow store in
`data/mlflow/`, only after a package is sealed; it is resumable and never touches test rows. `jobs`, `job`, `compare` and `reproduce` (`docs/reproducibility.md`) are read-only: `compare` uses CV numbers only; `reproduce` re-runs CV from the record, never reads or scores the test part and writes nothing. scikit-learn has no type information, so mypy has a narrow override for
`sklearn.*` only. Native thread pools are capped at 2 in `automl_ops/__init__.py`. Next planned work
(P4-2 onward) needs the owner's approval first; each chunk is proposed, approved and gated. The cross-job test-set reuse policy is
**still an open decision**.

## Working process (approval gates)
Work in **chunks**: a phase, or a group of steps, sized by where the owner's input is needed.
1. **Propose first.** Before each new chunk, present a proposal (purpose, steps covered, files and
   records it would change, tools/packages and any approval needed, risks, how it will be checked
   and what the checks do and do not show, what can wait) and **wait for the owner's explicit
   approval**. Silence or a general "okay" is not approval. Approval of one chunk never approves
   later work.
2. **Implement only what was approved.** Follow the owner's stated choices; keep provisional
   limits and defaults unless the owner approves a change. If a material problem needs a change,
   stop and ask.
3. **Update the directly related records and documentation** (below) so they match reality, then
   **report**: what changed, the commands run and their actual results (PASS/FAIL/NOT RUN), what the
   checks establish and do not cover, deviations, and any decision needed. Then **stop** at the next
   decision or gate. Never start the next chunk on your own.
- **Provisional plan.** Tools, packages, versions and design details in the plan are provisional
  recommendations. Open decisions are made at their step using what is known then. If evidence
  suggests a better choice, present the reason and trade-offs to the owner **before** a material
  change (scope, versions, commitments, dependencies, data use, deliverables), and record the change
  and why. Routine factual updates that reflect completed work may be made and reported. Ask the
  owner only when a decision materially affects the next work. Never silently change scope, skip a
  dependency, or mark unfinished work complete.
- These approvals are separate and always required: starting each phase/chunk (the phase gates), and
  any Git or external action not covered by an approved chunk (see the dependency rule below).
- Do not create future packages or files just to reserve a place for them.
- If unclear whether the owner wants discussion, a plan change or implementation, ask first.

## Hard rules
- **No Git actions** (init, status, add, commit, push, remotes, check-ignore, visibility) unless the
  owner explicitly asks. A `.git` folder exists; the owner manages it.
- **Dependencies are approved per chunk.** Each chunk proposal lists every direct dependency and why
  it is needed, the likely transitive dependencies, compatibility checks, size/resource estimates
  (marked verified or unverified), and the package-index or other external checks and downloads
  involved. The owner's approval of the chunk approves that listed set. If implementation needs a
  dependency or outside action that was not in the approved chunk (another direct dependency, a
  change to pandas or numpy, or any other external contact), **pause and ask before proceeding**.
  Report the exact resolved packages and versions afterwards. Declare every package the code imports
  directly (no reliance on transitive installs).
- **No external actions** without approval of exactly what is sent: no uploads, tokens or outside
  contact. Package-index checks and installs, and dataset downloads, are covered only when the
  approved chunk lists them; anything else needs approval first.
- Never expose or commit data, secrets, tokens or model files.
- **Scratch files:** keep temporary files in this session's scratchpad directory only, and delete
  only by the exact full path of a file created in this session. Never use a wildcard across the temp
  root (`/private/tmp/claude-501/*/...` also spans other projects and sessions). If a pattern must be
  used, list what it matches first and confirm every match is mine. Leave the scratchpad itself for the
  system to clear (incident recorded in D-017).
- Re-check disk and memory before clearly heavy work (installs of MLflow/Evidently, `node_modules`,
  Docker builds, large runs). Do not delete anything to free space; tell the owner what is reclaimable.
- Keep code compatible with Linux and macOS; give a fallback for any platform-specific command.
- One implementation per capability, shared by every entry point (API, worker, serving, scripts).
- Only `automl_ops/tracking.py` imports `mlflow` (a test enforces it); nothing MLflow-typed crosses it.

## Project rules to keep
- **Sealed test set.** Regular runs record cross-validation results only. A job's test part is read
  once, for the selected model, after its configuration is frozen (BUILD_PLAN Step 11). Reuse of the
  test part across jobs is an **open decision for Step 11**; until then every reported test result
  carries the limitation that repeated use across jobs can influence model choices. Never call it
  independent or leakage-proof.
- **Separate monitoring data.** Training, cross-validation, selection and the final test use only the
  training pool. The traffic simulator and its shifted copy use only the separate monitoring stream.
  Simulated traffic is always labelled as simulated.
- **Registry authority (proposed, settled at Step 20).** PostgreSQL is authoritative for model status
  and history; MLflow aliases mirror it. Sync failures stay "pending sync"; disagreements are reported
  and fixed only by an explicit, logged reconcile.
- **Compose and demo workflow.** `docker compose up` starts services only. Demo data is prepared by a
  one-time script; the end-to-end script (or the UI) runs the workflow.
- Saved models are pickle-based joblib files: load only from the platform's own store, after the hash
  check. Never load user-uploaded model files.
- Local, single-user, no authentication, localhost only. Do not claim production readiness.

## Truthfulness
- Never say something works unless a relevant check was run. Label claims verified (and how),
  assumed, recommended or unverified. Owner-reported results are "owner-reported", not verified.
- Report failures, warnings and my own mistakes plainly. Prefer primary sources for changing
  technical or licence details and say what was checked and when. No legal conclusions.
- Keep claims about model capability and project impact proportionate to the evidence.

## Records to keep up to date
- **`DECISIONS.md`** (root, tracked): append-only log. After each approved phase/chunk: number, date,
  type/status, what and why, files changed, commands run and real results, verification, anything
  unverified. Owner choices are labelled **owner decision**; provisional defaults and assumptions are
  labelled as such. Never delete entries; when a decision changes, add a new entry and mark the old
  one superseded.
- **`project-notes/BUILD_LOG.md`** (private): a plain-language entry for every executed plan step
  (purpose, method, findings, interview questions); a phase summary at each phase end. Never mark a
  step done unless it was run and verified. Owner-approved deferrals are marked
  **deferred (owner-approved)** with the reason, never done.
- **`docs/adr/`**: only for meaningful checkpoint decisions.
- **`.gitignore`**: when a new file/folder should stay untracked, add it, check it hides no needed
  file, and record it in `DECISIONS.md`.

## Environment and commands
- Machine: MacBook Air M1, 8 GB RAM, about 10 GiB free disk; Docker VM limit about 4 GB. CPU only.
- Python 3.12 (Homebrew interpreter) managed by **uv**; environment is `.venv` in the project root.
  Terminal activation: `source .venv/bin/activate`. Use `uv sync`, `uv run ...`, `uv add ...`;
  never `pip install` directly. Python downloads are disabled (`UV_NO_PYTHON_DOWNLOADS=1`) unless
  the owner approves one.
- `make lint` (ruff), `make typecheck` (mypy strict), `make test` (pytest), `make check` (all three).
  Run them after code changes and report real results.
- `uv run python scripts/datasets.py ingest|list|show|verify|profile|validate|task|explore|schema|train`
  manages dataset versions (default store `data/store/`). Versions are immutable and read-only; never
  edit or replace a stored file. All commands except `ingest` read a version and change nothing.
  `train` cross-validates on a training part only; it never reads the held-out test part.
  `train|finalize --track` and `track-package` log to MLflow (`--mlflow-dir`, default `data/mlflow`);
  `scripts/mlflow_ui.py` starts the UI on 127.0.0.1:5001 by hand. MLflow telemetry stays disabled
  (`MLFLOW_DISABLE_TELEMETRY=true`, set by `tracking.py` and `mlflow_ui.py`).
  `jobs|job|compare|reproduce` (Steps 15-16) read the MLflow records; `reproduce` exits 0 reproduced, 1 not
  reproduced, 2 cannot run, and never reads the held-out test part.
- MLflow tracking backends (P4-2b, `docs/tracking.md`): SQLite is the default; PostgreSQL (database `mlflow`,
  artifacts `data/mlflow-pg/artifacts`, max 157-character location) only via `--mlflow-backend postgresql` or
  `AUTOML_MLFLOW_BACKEND` and after `scripts/db.py mlflow-init`; `mlflow-status` is read-only. Never delete,
  move or modify `data/mlflow` (the archive) or `data/backups/`. `scripts/tracking_migrate.py` did the migration.
- `make api` starts the control-plane API (127.0.0.1:8100; needs PostgreSQL at the migration head; `AUTOML_API_PORT`, `AUTOML_UI_ORIGIN`); `make openapi` regenerates `docs/openapi-control.json` (a test fails if it is stale).
- `make db-up` / `make db-stop` run the application PostgreSQL (`compose.yaml`, 127.0.0.1:55432, local
  image only, password in git-ignored `.env`). `uv run python scripts/db.py upgrade|status|check|sync`
  (migrations run only on `upgrade`; `sync` catalogs the file store and never changes files). DB tests
  are marked `postgres` and report NOT RUN (skipped) when the server is unreachable.
- Data-reading policy: cells are read as text; only an empty cell is missing; NA-like text and
  values with leading zeros (`007`) keep their original text and are reported, never converted.
  Task-suitability checks (task type, class sizes) live in `task.py` (Step 5), not in profile/validate.
  Schema results separate structural failures from observations about the training data; unseen
  categories and out-of-range values are observations, never failures, and not drift detection.
- `make demo-data` prepares the demo datasets (downloads from archive.ics.uci.edu only if
  `data/raw/` lacks them; the only approved external contact). The pool is the only file uploaded
  for training; the monitoring stream and shifted copy are never used for training or testing.
- Layout: `src/automl_ops/` (src layout), `tests/`, `scripts/`, `docs/`. Create sub-packages only
  when their step needs them. pandas is type-checked through `pandas-stubs` (the old
  `ignore_missing_imports` override for pandas was removed).
