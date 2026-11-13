# Decisions and execution log

Append-only. Each entry: number, date, type/status, what and why, files changed, commands run and
real results, verification, and anything unverified. Old entries are never deleted; if a decision
changes, a new entry is added and the old one is marked **Superseded**.

**Labels:** *Owner decision* = you decided it. *Provisional default* = my recommendation, in place
until its step or evidence says otherwise. *Assumption* = believed but not checked. *Verified* means a
relevant check was run and its result is recorded.

---

## D-001 · 2026-11-13 · Owner decision · Planning documents approved as working guides
- **Decision:** `SCOPE.md`, `project-notes/PROJECT_GUIDE.md`, `project-notes/BUILD_PLAN.md` and
  `project-notes/ARCHITECTURE.md` are approved as **working guides, not fixed contracts**. Open
  decisions stay open until their step. Tools, packages, versions and details are provisional; if
  evidence shows a better choice it is presented with trade-offs before a material change, and the
  change is recorded here. Phase gates remain. Approval is still required for starting implementation
  of each phase, adding dependencies, and Git or external actions.
- **Why:** the owner reviewed the drafts over several revisions (sealed test set, separate monitoring
  data, registry authority, MLflow storage sequence, Compose/end-to-end wording).
- **Still open (decide at the step named):** final-test reuse across jobs (Step 11); Postgres-
  authoritative registry rule (Step 20); MLflow development store and package size (Step 13);
  monitoring-stream share and dataset choice (Step 2); Evidently version (Step 24); job runner
  (Step 19); serving model loading (Step 21); chart library (Step 29).
- **Files changed:** none in this entry (the documents were edited in earlier planning rounds).

## D-002 · 2026-11-13 · Owner decision · Private notes ignored; DECISIONS.md and SCOPE.md tracked
- **Decision:** `/project-notes/` is git-ignored (local only). `DECISIONS.md` (repository root) and
  `SCOPE.md` stay tracked and must never be ignored.
- **Why:** same arrangement as the owner's earlier projects.
- **Consequence:** the planning documents other than `SCOPE.md`, and `BUILD_LOG.md`, are not in Git.
  (`ARCHITECTURE.md` lives in `project-notes/` and is therefore ignored too.)

## D-003 · 2026-11-13 · Owner decision · Python 3.12 from Homebrew; `.venv` managed by uv
- **Decision:** use the already-installed Homebrew Python 3.12; download no other Python. The
  environment is `.venv` in the project root, managed by uv. Terminal activation:
  `source .venv/bin/activate`.
- **Correction of a factual error:** `PROJECT_GUIDE.md` §10 said there was no Python 3.12 on this
  Mac. It was wrong (my earlier command output was cut off). Python 3.12.16 is at
  `/opt/homebrew/bin/python3.12`. §10 corrected.
- **Verified:** `uv venv` reported "Using CPython 3.12.16 interpreter at
  /opt/homebrew/opt/python@3.12/bin/python3.12"; `uv run python` resolves to the project's `.venv`.

## D-004 · 2026-11-13 · Owner decision · Step 1 chunk approved (Parts A and B), dev tools only
- **Decision:** do Part A (records and guard files) and Part B (Python scaffold) together. Add only
  pytest, ruff and mypy as development tools and **no runtime dependencies**.
- **Why:** smallest set needed to check a scaffold; runtime packages arrive with the phase that uses
  them, each with its own approval.

## D-005 · 2026-11-13 · Owner decision · No Git commands, including `git check-ignore`
- **Decision:** no Git commands at all in this chunk. A `.git` folder (branch `main`, no remote in
  its config) already existed when the project was reviewed. **Assumption:** the owner created it; I
  did not and have not run any Git command on it.
- **Consequence:** `.gitignore` could not be verified with Git. See D-006 for the substitute check.

## D-006 · 2026-11-13 · Executed · Step 1: setup records, guard files and Python scaffold
**Files created:** `.gitignore`, `CLAUDE.md`, `DECISIONS.md`, `pyproject.toml`, `.python-version`,
`uv.lock`, `README.md`, `Makefile`, `src/automl_ops/__init__.py`, `tests/test_smoke.py`, and the
`.venv/` environment (git-ignored). **Files changed:** `project-notes/BUILD_LOG.md` (status, tracker,
Step 1 entry, one sentence in the planning entry), `project-notes/PROJECT_GUIDE.md` (§10 Python row).

**Order:** `.gitignore` was written first, before any generated file existed.

**Commands run and real results**
| Command | Result |
| --- | --- |
| `uv venv --python /opt/homebrew/bin/python3.12 --no-python-downloads .venv` | PASS, CPython 3.12.16 |
| `uv add --dev --no-python-downloads pytest ruff mypy` | PASS, 13 packages installed: pytest 9.2.0, ruff 0.17.5, mypy 2.4.1 and 10 they depend on; downloaded from the package index; `uv.lock` written |
| `uv sync` | PASS (resolved 14, checked 13) |
| `uv sync --locked` | PASS (exit 0) |
| `uv run pytest` | PASS, 2 passed |
| `uv run ruff check .` | PASS, all checks passed |
| `uv run mypy` | PASS, no issues in 2 source files |
| `make check` (ruff, mypy, pytest) | PASS |
| `uv run python -c "import sys; print(sys.executable)"` | PASS, `.venv/bin/python`, Python 3.12.16 |

**`.gitignore` review without Git:** PASS. A scratch script (in the session scratch folder, not in
the project) used `pathspec` (a package mypy pulls in; not a project dependency and not imported by
project code) to apply the file's rules to sample paths. 25 paths that should be ignored were (data,
`.env`, `.env.local`, models/`.joblib`, MLflow dirs, `reports/`, `.venv`, caches, frontend build
output, `dist`/`build`/`egg-info`, `.DS_Store`, `project-notes/`); 19 that must stay visible were
(source, tests incl. `tests/data/`, docs, `.env.example`, `uv.lock`, `pyproject.toml`,
`.python-version`, `Makefile`, `README.md`, `SCOPE.md`, `DECISIONS.md`, `CLAUDE.md`, `.gitignore`,
`frontend/src`, `frontend/package.json`, a sub-package named `models`). **Limit:** this imitates
Git's rules; it is not Git. Running `git check-ignore` would be stronger; it is the owner's call.

**Version constraints set (provisional defaults):** `requires-python = ">=3.12,<3.13"`;
`.python-version` = `3.12`; mypy `python_version = "3.12"`. Dev dependency lower bounds written by
uv: `pytest>=9.2.0`, `ruff>=0.17.5`, `mypy>=2.4.1`. Other provisional defaults: build backend
`uv_build>=0.13.6,<0.14.0`; ruff line length 100 and rules E, F, I, B, UP; mypy `strict = true`;
pytest `testpaths = tests`. `__version__` is read from installed package metadata.

**Deviations from the plan and the approved proposal**
- `pyproject.toml` was written by hand instead of `uv init` (which can create its own `.gitignore`/
  `README.md` and start Git). The result is equivalent; no `uv init` was run.
- **Deferred (owner-approved, in the approved proposal): the wheel-compatibility check** in BUILD_PLAN
  Step 1's method ("check that Python 3.12 has arm64 wheels for scikit-learn, MLflow and Evidently").
  **Not done.** It will be done at the step that first installs each package. `PROJECT_GUIDE.md`
  assumption A2 and BUILD_PLAN Step 1 text still say it is verified at Step 1; I propose to correct
  that wording and ask for the owner's go-ahead before editing the plan.
- `UV_NO_PYTHON_DOWNLOADS=1` was used to guarantee that no Python was downloaded. Package downloads
  from the package index (the three tools and their dependencies) did occur, as approved.

**Resources:** `.venv` about 100 MB; `uv.lock` 36 KB; disk still about 10 GiB free (96% used) after the
install. uv's shared cache (about 1.8 GB, from earlier projects) was not touched.

**Unverified / not run:** Linux behaviour; a fresh-clone install; whether `uv_build` is the best
backend (provisional; works here); any ML package support on Python 3.12 (deferred above); GitHub CI
(none exists). These checks show only that the scaffold and three tools work.

## D-007 · 2026-11-13 · Executed · Compatibility-check wording corrected (documentation only)
- **What:** `project-notes/PROJECT_GUIDE.md` assumption A2 and `project-notes/BUILD_PLAN.md` Step 1
  said the scikit-learn / MLflow / Evidently Python 3.12 compatibility check happens at Step 1. They
  now say it happens when each package is introduced: scikit-learn at Step 7, MLflow at Step 13,
  Evidently at Step 24 (fallback to 3.13 only with evidence). This matches the owner-approved
  deferral recorded in D-006; no decision changed.
- **Why:** the old wording was stale after D-006 and would have shown Step 1 as incomplete.
- **Files changed:** the two documents above. No code, dependencies, data or Git state touched.
- **Verification:** NOT RUN (text edits only). D-006's deferral note ("I propose to correct that
  wording") is now done by this entry; D-006 itself is left unedited (append-only).

## D-008 · 2026-11-13 · Owner decision · Step 2 approved with provisional dataset choices
- **Decision:** UCI Adult (classification) and UCI Abalone (regression); add **pandas and numpy**
  as runtime dependencies (earlier than Step 3, because the preparation script needs them);
  contact **only** the official UCI dataset pages and download endpoints, for licence/source
  verification and the two downloads. Pandera and any other package stay out until their steps.
- **Provisional defaults I set within that approval (not permanent):** monitoring-stream share 30%;
  split seed 42 (shift seed 43); stratified split for Adult, plain random for Abalone; the shift
  parameters in D-009; Adult `?` markers turned into empty values.
- **Condition set by the owner:** verify page, licence and URLs before downloading and stop if they
  differ materially from the proposal. (They did not; see D-009.)

## D-009 · 2026-11-13 · Executed · Step 2: demo data prepared and verified
**Source verification before downloading (contact with UCI only):** read the official pages
`https://archive.ics.uci.edu/dataset/2/adult` and `.../dataset/1/abalone` (and UCI's JSON
`/api/dataset?id=` metadata, same host) with a generic client name. The API has no licence field; the
licence was read from the pages. **Verified:** both pages state "licensed under a Creative Commons
Attribution 4.0 International (CC BY 4.0) license ... provided that the appropriate credit is given";
the download links are `https://archive.ics.uci.edu/static/public/2/adult.zip` and
`.../1/abalone.zip` (as proposed); DOIs 10.24432/C5XW20 and 10.24432/C55C7W; documented rows 48,842
and 4,177; Adult has missing values, Abalone none. No material difference from the proposal, so I
proceeded. Access date **2026-11-13**. (My own licence reading, not legal advice.) Requests
without a trailing slash were used; the slash variants answered with same-host redirects, which I
refused (`--max-redirs 0`) and did not follow elsewhere.

**Downloads:** the two zips via the project's own `download()` (HTTPS, UCI host only, no redirects,
size cap): `adult.zip` 620,237 bytes, SHA-256 `7537312d...cc21bb`; `abalone.zip` 55,357 bytes,
SHA-256 `755a6a67...6210ee4` (hashes computed locally; UCI publishes none). Both answered 200
directly. **Note:** I ran `download()` directly (a `python -c` call) so I could inspect the files
before writing the merge; the script later found the files already present and did **not**
download. The script's own download branch was therefore **NOT RUN** end to end (the function it
calls was).

**Original Adult formats checked before merging:** `adult.data` 32,561 rows, labels ` <=50K` / ` >50K`;
`adult.test` 16,281 rows after a `|1x3 Cross validator` comment line, labels ` <=50K.` / ` >50K.`
(trailing period); both use `?` for missing. 32,561 + 16,281 = 48,842, equal to UCI's count.

**Duplicates, counted before deciding:** Adult 52 extra exact copies (101 rows in 49 groups,
largest 3); Abalone 0. Decision: **no rows removed**; exact-duplicate groups kept together on one
side of the split so no content-identical row spans pool and stream. Rationale: 0.1% of Adult;
removal would silently change the data. (UCI's own note mentions "6 duplicate or conflicting
instances", a different measure; same-features-different-label rows were not examined.)

**Results:** Adult 48,842 rows -> pool 34,190 / stream 14,652 (class balance 0.7607 vs 0.7607);
Abalone 4,177 -> 2,924 / 1,253. Adult missing: workclass 2,799, occupation 2,809, native-country 857.
Shifted copies (from stream rows only, labels unchanged): Adult age +14, hours-per-week +12 (one
stream SD rounded to whole numbers), occupation set to Exec-managerial for 2,985 rows (25% of
eligible); Abalone Whole_weight +0.490056, Shell_weight +0.139391, Sex set to I for 250 rows (30%
of eligible).

**Files created:** `src/automl_ops/demo_data.py`, `scripts/prepare_demo_data.py`,
`tests/test_demo_data.py`, `docs/data.md`, `data/raw/` (2 zips) and `data/demo/` (6 CSVs +
`manifest.json`), both git-ignored. **Files changed:** `pyproject.toml` and `uv.lock` (pandas,
numpy; mypy/ruff cover `scripts/`; mypy ignores missing pandas stubs), `Makefile` (`demo-data`
target), `CLAUDE.md`, `project-notes/BUILD_PLAN.md` (Step 2 status), `project-notes/PROJECT_GUIDE.md`
(§11 row: decided), `project-notes/BUILD_LOG.md`, this file.

**Checks (all actually run):**
| Check | Result |
| --- | --- |
| Row counts equal UCI's documented counts (48,842; 4,177) | PASS |
| Pool + stream = source rows (nothing dropped) | PASS (both datasets) |
| No row content shared by pool and stream (full-row comparison of the written CSVs) | PASS, 0 shared (both) |
| Shifted copy: same columns and rows as stream, target column identical, only documented columns differ, means moved by the documented amount, changed-row counts match | PASS |
| No `?` remains; class balance within 0.5 points; file hashes match manifest; raw hashes match manifest | PASS |
| Reproducibility: script run three times (once into a separate folder, once after code reformatting): SHA-256 of all 7 output files identical | PASS |
| `uv run pytest` | PASS, 27 passed (25 in `test_demo_data.py` including 1 on the real files, plus 2 smoke) |
| `uv run ruff check .`, `uv run mypy` (strict, 5 files) | PASS |
| `uv sync --locked` | PASS (exit 0) |
| `make check` | PASS |
| `make demo-data` re-run | PASS (ALL PASS) |
| Script's own download branch | NOT RUN (see above) |

**Estimates vs measured:** proposal estimates of under 10 MB downloaded and about 15 MB prepared;
measured 676 KB downloaded (zips; Adult's two files are 6.0 MB unzipped) and 8.3 MB prepared. The
pandas/numpy install took `.venv` from about 100 MB to 183 MB. Disk free 13 GiB (varies; other
activity on the Mac).

**Deviations and provisional implementation choices:**
- mypy cannot type pandas without `pandas-stubs` (not approved), so `pandas` is set to
  `ignore_missing_imports`. Consequence: pandas calls are unchecked by mypy. Whether to add
  `pandas-stubs` can be raised at Step 3.
- `ruff format` was run on the new files (style only); `scripts/` was added to ruff and mypy scope.
- `sha256` of the raw zips and the shifted-file checks are my own design; no source checksum exists.
- Dependencies added: pandas 3.1.1, numpy 2.5.4 (direct) and python-dateutil 4.0.0, tzdata 2026.6
  (pulled in by pandas). Compatibility with Python 3.12 on this Mac was shown by the install and the
  tests; other platforms were not tested.

**Unverified / not run:** Linux; other pandas/numpy versions; the script's download branch; whether
these licences suit every use of the repository (my reading only); same-features-different-label rows;
any ML behaviour. The checks show that the files are intact, separated and reproducible, not that they
are suitable for modelling or that the shift is realistic.

## D-010 · 2026-11-13 · Owner decision (carried forward) · No fairness or real-world drift claims
- Adult contains sex and race and 1994 income labels; all demo data and shifted copies are labelled
  demo/synthetic in `docs/data.md`, the manifest and the script output. Whether to exclude sensitive
  columns is left as a user setting for Step 5.
