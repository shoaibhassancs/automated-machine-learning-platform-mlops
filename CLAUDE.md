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
logging of training runs through `tracking.py`) is done and verified, and Gate 3A is closed** (owner-approved; no further changes to 3A). **Chunk 3B (Steps 15-16: experiment history and CV-only comparison, and the read-only reproducibility check) is implemented and verified and awaits the owner's Gate 3 review**; the model registry and everything after it are not started. See `DECISIONS.md` and `BUILD_LOG.md`. Runtime dependencies: pandas, numpy, pandera,
scikit-learn, joblib, mlflow. Dev: pytest, ruff, mypy, pandas-stubs.
Code: `demo_data.py` (Step 2), `ingest.py` and `dataset_store.py` (Step 3), `profile.py` and
`validate.py` (4A), `task.py` (5), `explore.py` (6), `schema.py` (4B), `pipeline.py` (7),
`algorithms.py` (8), `evaluate.py` (10), `train.py` (9), `selection.py` and `finalize.py` (11),
`package.py` (12), `tracking.py` (13-14, plus the 3B read side; the only module that imports `mlflow`), `compare.py` and `reproduce.py` (15-16, no MLflow), `hashing.py`, and
the scripts `prepare_demo_data.py`, `datasets.py`, `mlflow_smoke.py` and `mlflow_ui.py`. Demo data (UCI Adult and Abalone, CC BY 4.0) is prepared by
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
(Phase 4) needs the owner's approval and a proposal first. The cross-job test-set reuse policy is
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
