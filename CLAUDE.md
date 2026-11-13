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
Steps 1 (Python scaffold) and 2 (demo data) are done and verified; see `DECISIONS.md` and
`BUILD_LOG.md`. Code so far: `src/automl_ops/demo_data.py` and `scripts/prepare_demo_data.py` only.
Runtime dependencies: pandas, numpy. Demo data (UCI Adult, UCI Abalone, CC BY 4.0) is prepared by
`make demo-data` into git-ignored `data/`; sources, licences and the pool/stream/shifted-copy rules
are in `docs/data.md`. Next planned work: Phase 1, Step 3 (needs the owner's approval).

## Working process (approval gates)
- Work in the plan's phases. At the start of a phase, explain its purpose, steps, recommended
  approach, dependencies, verification and any decision needed, then **wait for explicit approval**.
  Approval of one phase never covers a later one. Silence or a general "okay" is not approval.
- After each completed step, update every affected record (below) so it matches reality.
- **Provisional plan.** Tools, packages, versions and design details in the plan are provisional
  recommendations. Open decisions are made at their step using what is known then. If evidence
  suggests a better choice, present the reason and trade-offs to the owner **before** a material
  change, and record the change and why. Ask the owner only when a decision materially affects the
  next work. Never silently change scope, skip a dependency, or mark unfinished work complete.
- Do not create future packages or files just to reserve a place for them.
- If unclear whether the owner wants discussion, a plan change or implementation, ask first.

## Hard rules
- **No Git actions** (init, status, add, commit, push, remotes, check-ignore, visibility) unless the
  owner explicitly asks. A `.git` folder exists; the owner manages it.
- **Ask before adding dependencies**: explain why each is needed, wait for approval, and declare every
  package the code imports directly (no reliance on transitive installs).
- **No external actions** without approval of exactly what is sent: no uploads, tokens or outside
  contact. Downloads/installs from package indexes and datasets need approval first.
- Never expose or commit data, secrets, tokens or model files.
- Re-check disk and memory before clearly heavy work (installs of MLflow/Evidently, `node_modules`,
  Docker builds, large runs). Do not delete anything to free space; tell the owner what is reclaimable.
- Keep code compatible with Linux and macOS; give a fallback for any platform-specific command.
- One implementation per capability, shared by every entry point (API, worker, serving, scripts).

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
- `make demo-data` prepares the demo datasets (downloads from archive.ics.uci.edu only if
  `data/raw/` lacks them; the only approved external contact). The pool is the only file uploaded
  for training; the monitoring stream and shifted copy are never used for training or testing.
- Layout: `src/automl_ops/` (src layout), `tests/`, `scripts/`, `docs/`. Create sub-packages only
  when their step needs them. mypy treats pandas as untyped (no pandas-stubs; not approved).
