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

## D-011 · 2026-11-13 · Owner decision · Step 3 approved; working process recorded
- **Decisions (owner):** (1) store accepted CSVs as UTF-8, comma-separated files and use the hash of
  that normalised content as the version identity; keep the original upload's hash and format
  details as provenance; do **not** retain the original upload bytes. (2) Include the small
  `scripts/datasets.py` CLI (`ingest`, `list`, `show`, `verify`). (3) Defer `pandas-stubs` to Step 4;
  do not add it now. (4) Keep the proposed limits (50 MiB, 100,000 rows, 100 columns) and the
  encoding/delimiter rules as provisional defaults; do not silently expand scope or change a limit;
  if a material problem forces a change, stop and ask. (5) Use the existing Adult and Abalone
  prepared files to exercise ingestion; no downloads, no new dependencies.
- **Working process (owner):** work in chunks sized by where owner input is needed; propose before
  each chunk and wait for approval; after approval implement only that chunk, update directly
  related records, report real results, and stop at the next decision or gate; approval of one chunk
  never approves later work; the plan is a working guide and material changes need the owner's
  approval first. Recorded in `CLAUDE.md` ("Working process"), consistent with the phase gates and
  the dependency / Git / external-action approvals, which are unchanged.

## D-012 · 2026-11-13 · Executed · Step 3: CSV ingestion and dataset versions
**Files created:** `src/automl_ops/ingest.py`, `src/automl_ops/dataset_store.py`,
`scripts/datasets.py`, `tests/test_ingest.py`, `tests/test_dataset_store.py`,
`tests/test_datasets_cli.py`, `tests/test_ingest_real_data.py`, `docs/ingestion.md`.
**Files changed:** `CLAUDE.md` (current state, working process, commands),
`project-notes/ARCHITECTURE.md` (components row split into the two modules; layout line),
`project-notes/BUILD_PLAN.md` (Step 3 status), `project-notes/BUILD_LOG.md`, this file.
**Not changed:** `pyproject.toml` / `uv.lock` (no dependencies), the demo datasets (hashes compared
before and after). No `data/store/` was created: tests use temporary folders and my CLI exercise
used a scratch folder outside the project.

**What was built:** `ingest.prepare_csv` checks one upload and returns canonical bytes (UTF-8,
comma, `\n`, minimal quoting) plus counts and provenance; `FileDatasetStore` keeps versions as
`<dataset>/vNNNN.csv` + `.json` (read-only; the record is written last as the commit marker);
version identity is the SHA-256 of the canonical bytes. Rules: a new name must not exist; new
content under an existing name needs `--add-version`; `--add-version` on a missing name is refused;
identical content returns the existing version and writes nothing; names are `[a-z0-9][a-z0-9_-]{0,63}`
and references are `name@vN`; the uploaded file name is stored as a base name for information only.
Limits and format rules exactly as proposed (see `docs/ingestion.md`); no limit was changed.

**Checks (all actually run):**
| Check | Result |
| --- | --- |
| CSV parsing with invented CSVs: valid file, exact cell text (quoted comma, embedded newline, empty cell, unicode), 11 packagings of the same data (UTF-8, BOM, UTF-16 LE/BE, semicolon, tab, pipe, CRLF, CR-only, no final newline, trailing blanks) give identical canonical bytes and hash; provenance recorded; 18 kinds of bad file rejected with the expected code (empty, blank-only, header-only, one column, empty/duplicate headers, ragged rows, blank line before/between, NUL, invalid UTF-8, UTF-32, bad quoting, ambiguous delimiter, oversized cell) | PASS (43 tests in `test_ingest.py`) |
| Limits: row, column and byte boundaries (small injected limits, exactly-at and one-over); default values equal the approved ones; default byte cap rejects before parsing; file reader refuses big files before reading them | PASS |
| Limits on real-size files (scratch, deleted afterwards): 100,000 rows / 46.8 MB accepted; 100,001 rows rejected ("more than 100,000 data rows"); a 52.8 MB file refused ("limit is 52,428,800 bytes") reading about 22 MB of memory | PASS |
| Version immutability and idempotence: v1 created with read-only files; identical and repackaged re-ingest return v1 and write nothing; changed content refused without the flag and creates v2 with it; v1 bytes and record unchanged afterwards; returning to old content returns the old version; typo name with `--add-version` refused; rejected CSV leaves no folder; invalid names (including `../x`) refused; simulated disk failure leaves no half-made version; orphan data file never overwritten | PASS (48 tests in `test_dataset_store.py`) |
| Hash verification: untouched store passes; changed file (same size), changed size, missing file, wrong record, corrupt record, orphan and leftover temp files are each reported | PASS |
| CLI as a real subprocess: ingest / re-ingest / refusal / `--add-version` / list / show / verify (pass, then FAIL after tampering); bad input gives a message and exit code 2, no traceback | PASS (2 tests) |
| Real demo files (Adult and Abalone, all six prepared CSVs): each ingests with the manifest's row count and columns, was **not** re-encoded, its version hash equals the Step 2 manifest hash and the original hash, stored bytes equal the file, re-ingest is idempotent, `verify` passes, all six coexist in one store; demo file SHA-256s identical before and after | PASS (7 tests in `test_ingest_real_data.py`; plus a manual CLI run on all six) |
| Timing and memory, measured (not claimed beyond this Mac): Adult pool 3.7 MB about 0.2 s; 46.8 MB / 100,000 rows about 1.2 s with a peak of about 407 MB | measured (peak memory from `/usr/bin/time -l`, approximate) |
| `uv sync --locked`, `ruff check`, `mypy` (strict, 12 files), `pytest` (127 passed in total), `make check` | PASS |

**Findings during the work (corrections to my own first draft):** two of my own tests failed on the
first run and showed real weaknesses, both fixed before finishing: a file of only blank lines gave a
misleading "bad delimiter" message (now "contains only blank lines"), and a cell over Python's field
limit also gave "bad delimiter" (now reported as a CSV syntax problem). No limit or rule changed.

**Deviations:** none from the approved proposal. `ruff format` was applied to the new files. No
dependency was added.

**Not covered / assumptions:** several processes writing at once (only the atomic version-number claim
is tested); Windows; other Python versions; read-only bits stop accidents, not a determined user;
`verify` detects change but cannot repair; the store does not know file roles, so a monitoring-stream
file could be ingested like any other (BUILD_PLAN Step 25 adds the hash guard); line endings inside
quoted cells are kept as written, so such files can map to different versions; there is no delete or
archive feature (manual file removal); whether pandas reads the stored files and how types and missing
values behave is Step 4; peak-memory figure is from one run. The checks show that parsing, limits,
immutability, hash verification, idempotent re-ingestion and the demo files behave as designed, not
that the design is complete for every real-world CSV.

## D-013 · 2026-11-13 · Owner decision · Chunk 4A approved; Step 4 split; plan wording moved
- **Decisions (owner):** (1) Step 4 is split: **4A** profiling and rule-based validation now; **4B**
  Pandera schema is a separate future chunk (Pandera not installed, its support for pandas 3.1.1
  unverified). (2) Use the proposed necessary rules and provisional thresholds; optional extras left
  out. (3) Only empty cells are missing; NA-like text is kept as a value and reported as a warning.
  (4) **Added by the owner:** a leading-zero check: values like `007` may be codes; keep their text and
  flag the column as ambiguous / possibly identifier-like; never silently reinterpret as `7`. (5) Task
  suitability and class-size checks **move from Step 4 to Step 5** (plan wording updated). (6) Reports
  are computed on demand; stored versions are not modified. (7) The profile's kinds are the planned
  shared reference for later typed loading, while Step 4 preserves raw cell values. (8) `pandas-stubs`:
  a limited attempt only (contact the package index only to check compatibility and dependency/size
  information; add as a dev dependency only if compatible; fix reasonable related typing issues; stop
  and report otherwise; do not change pandas or numpy; do not force it).

## D-014 · 2026-11-13 · Executed · Step 4, chunk 4A: profiling and rule-based validation
**`pandas-stubs` attempt (the only external contact: the package index).** Contacted `pypi.org` for
`pandas-stubs` metadata, then added it with uv. **Facts read from the index:** latest
`3.1.0.261029` (released 2026-10-29), BSD-3-Clause, requires Python >=3.11, **only dependency
`numpy>=1.23.5`** (already installed), wheel 179,851 bytes. **Compatibility judgement:** the stubs
version number tracks pandas 3.1.0; the installed pandas is 3.1.1 (a patch release; no 3.1.1 stubs
exist). It installed with no new transitive packages and no change to pandas or numpy. I confirmed in
`.venv` that pandas 3.1.1 ships **no `py.typed` marker** (numpy does), which is why mypy could not use
pandas's own types and the `ignore_missing_imports` override had been needed (I had added it up front
in Step 2 without testing what mypy would say). **Result:** with the override removed, strict mypy
reported **one** error in existing code (`demo_data.py`, `Series.to_numpy(dtype=object, na_value=None)`)
and none in the new 4A code; fixed by `column.astype(object).to_numpy()`. The Step 2 outputs were
regenerated into a scratch folder and are **byte-identical** (all seven file hashes equal); the project
demo files were not rewritten. **Outcome: the stubs are kept (dev dependency), the pandas override is
removed (`pyproject.toml`), and strict mypy passes on 17 files with no `type: ignore` anywhere.** Stubs
for pandas 3.1.1 exactly do not exist, so a future pandas upgrade may need a stub bump; this is a
limitation, not a problem today.

**Built:** `src/automl_ops/profile.py` (cells read as text with only empty cells missing; strict
number syntax; kinds `integer`, `float`, `categorical`, `text`, `mixed`, `code`, `empty`; per-column
counts, missing, distinct, top values, statistics, NA-like and leading-zero facts; dataset-level
duplicates and missing counts; `profile_version`, which refuses a stored file whose hash no longer
matches its record), `src/automl_ops/validate.py` (15 finding codes over provisional thresholds; target
quality only), `profile` and `validate` commands (text and `--json`; exit 0/1/2) in
`scripts/datasets.py`, and `docs/profiling.md`. Thresholds and settings (all provisional, documented):
rows error <30, warning <100; high missing >40%; ID-like checks need 50+ rows and 90% distinct; more
than 50 distinct text values = `text`; 50% numeric share = `mixed`; top 20 values kept. Unchanged:
ingestion limits, stored versions, demo datasets.

**Records changed:** `CLAUDE.md` (state, data-reading policy, commands, typing note),
`project-notes/BUILD_PLAN.md` (Step 4 split and status; class-size wording moved to Step 5),
`project-notes/ARCHITECTURE.md` (profile/validate rows; schema module marked not built),
`project-notes/BUILD_LOG.md`, this file. **Not changed:** `ingest.py`, `dataset_store.py` (reused as
is), the demo data, anything in Git (no Git command run).

**Checks (all actually run):**
| Check | Result |
| --- | --- |
| Invented-data profile tests: kinds incl. boundaries (50/51 categories, 50% numeric share), strict number syntax (`1_000`, `nan`, `inf`, `1e999`, spaces, non-ASCII digits are not numbers), leading zeros kept as text and flagged, NA-like text kept and counted, only empty cells missing, statistics equal the standard `statistics` module on several inputs, ranking and ties, duplicates, determinism, strict JSON, stored-file tamper refused, store never changed | PASS (47 tests) |
| Invented-data rule tests: every rule fires and does not fire, exactly at its threshold (29/30/99/100 rows; 40%/41% missing; 49/50 rows for ID-like; 179/180 distinct of 200; 50/60 categories), target errors, task suitability deliberately NOT checked (a one-row class passes), ordering, purity, JSON shape | PASS (37 tests) |
| Real demo data (all six prepared files): rows and columns equal the manifest; kinds as expected; Adult missing counts (pool + stream) equal Step 2's recorded totals; target class counts equal the manifest; target statistics, duplicate counts equal independent standard-library calculations; the shifted copy's means moved by exactly the documented amounts; no errors and no warnings on any file (only the informational `duplicate_rows` on Adult); hand-broken in-memory copies (blanked target, constant and ID columns, `N/A` in a number column, zero-padded codes, 10-row file, wrong target name) each give the expected finding; demo file hashes unchanged | PASS (33 tests) |
| CLI as a real subprocess: text and `--json`, exit codes 0/1/2, JSON identical across runs, store untouched, tampered file refused without a traceback | PASS (2 new tests) |
| JSON determinism on real size: `profile --json` of a 100,000 x 100 file run twice gives the same SHA-256 | PASS |
| Memory and time at the largest shapes the ingestion limits allow (one run each, scratch files, deleted afterwards): 100,000 rows x 12 columns, 46.8 MB: profile about 2.6 s, peak about 330 MB; 100,000 x 100 columns, 47.0 MB: about 4.1 s, peak about 767 MB; Adult pool 3.7 MB: about 0.4 s, about 106 MB | measured (peak from `/usr/bin/time -l`, approximate; my estimate of about 1 GB for the wide case was too high) |
| `uv sync --locked`, `ruff check`, strict `mypy` (17 files), `pytest` (246 passed in total), `make check` | PASS |
| `prepare_demo_data.py --verify-only` (Step 2 files still pass their own checks) | PASS |

**Findings during the work (my own mistakes, caught by my own tests or checks):** (1) the NA-like
warning first summed only the five listed example spellings, so it would have under-reported the cell
count; the profile now carries the full `na_like_count` (test added). (2) Three of my new test files
failed strict mypy (a sort key, an untyped return, a tuple-vs-`approx` comparison); fixed properly, not
suppressed. (3) Two of my test expectations were wrong (an NA-token example, a "mixed" example below
the 50% share); the code was right.

**Deferred / not done (by plan):** 4B (Pandera schema); Step 5 (task configuration and class-size
checks); Step 6 (exploration data); a typed pandas loader (Step 7 onward); persistence of reports (a
database-era question); API and UI surfaces (Phases 4 and 5).

**Not covered / limitations:** heuristic kinds and provisional thresholds on arbitrary real-world CSVs;
dates and times are not recognised (shown as categorical/text); whole numbers beyond about 15 digits
lose precision in statistics only; an identifier stored as an ordinary number is only caught when every
value is distinct or has a leading zero; one measurement run per shape; `pandas-stubs` 3.1.0 was
checked against pandas 3.1.1 (a patch difference). The checks show the rules, kinds, statistics and
reports behave as specified on invented and demo data, not that the thresholds suit every dataset.

## D-015 · 2026-11-13 · Owner decision · One chunk for Steps 5, 6 and 4B; dependency approval becomes chunk-based
- **Decisions (owner):** (1) Steps 5 (task configuration), 6 (exploration data) and 4B (Pandera schema)
  are one chunk ending at **Gate 1**; the plan's step numbers do not force separate chunks. Order:
  Pandera compatibility check, Step 5, Step 6, Step 4B, records. (2) **Dependency approval is
  per chunk:** the chunk proposal lists direct and likely transitive dependencies, compatibility
  checks, package-index contact and resource estimates; approving the chunk approves that set; an
  unlisted dependency or outside action means pause and ask. This replaces the earlier per-package
  rule (recorded in `CLAUDE.md`). (3) Approved dependency set: **Pandera** as the only new direct
  dependency, with the transitive packages its normal resolution requires; stop and ask if it needs
  a change to pandas or numpy; if Pandera proved incompatible, do not downgrade pandas or substitute
  another library (finish Steps 5-6, leave 4B, report options).
- **Design (owner):** the task suggestion is advisory and overrideable; a high class count or a
  continuous-looking target never rejects a classification by itself (warning only); a class with
  fewer than 2 rows is an error because the planned stratified split needs 2. Schema results are
  computed on demand (not stored); schema checks run on text as read (no conversion); user-declared
  domain rules are deferred (no code-only feature); structural failures are separated from
  observations about the training data, and unseen categories, extra columns, new blanks and
  out-of-range values are observations, never failures, and not drift detection (Step 24).
- **Provisional defaults (assumptions, changeable):** whole-number target with more than 20 distinct
  values suggests regression; classes warning above 50 classes and below 5 rows (5 is a placeholder
  until Step 9 fixes the number of folds); 10 sample rows, 20 histogram bins, Pearson correlation on
  numeric columns only.

## D-016 · 2026-11-13 · Executed · Steps 5, 6 and 4B: task configuration, exploration data, Pandera feature schema
**Dependency (the only external contact: the Python package index, via uv).** Checked disk first
(about 12 GiB free, 95% used; 8 GiB RAM). `uv pip install --dry-run pandera` resolved without
changing pandas or numpy; then `uv add pandera`. **Resolved (exact):** `pandera 0.36.0` (declared in
`pyproject.toml` as `pandera>=0.36.0`) plus new transitive `pydantic 2.14.3`, `pydantic-core 2.50.4`,
`annotated-types 0.8.0`, `typeguard 4.6.0`, `typing-inspect 0.9.0`, `typing-inspection 0.4.5`
(7 packages added; `packaging`, `typing-extensions`, `mypy-extensions` were already present; pandas
3.1.1 and numpy 2.5.4 unchanged; lock diff confirmed against a copy taken before). Pandera 0.36.0's
metadata: core requires packaging, pydantic, typeguard, typing_extensions, typing_inspect; pandas
support is the `pandas` extra (`pandas>=2.1.1`, `numpy>=1.24.4`, no upper bound). I declared plain
`pandera` because pandas and numpy are declared directly; the extra would add nothing. It ships
`py.typed`. **Compatibility (verified by running it):** `pandera.pandas` imports and validates a
text frame (pandas 3.1.1 `str` dtype), lazy validation reports missing columns and malformed cells;
strict mypy passes over the new code; no clash with `pandas-stubs`. **Cost (measured):** added
packages about 13 MB on disk (pandera 3.9 MB, pydantic 3.0, pydantic-core 5.4, typeguard 0.3, rest
small); importing Pandera takes about 1.5 s, so it is imported lazily inside the schema check only
(tested). My proposal's list of likely transitive packages was right except it omitted
`typing-inspection`.

**What was built.** `task.py`: `TaskConfig`, advisory `suggest_task`, `check_task_config` (errors:
unknown/missing/constant target [reusing 4A code], target excluded, unknown excluded column, no feature
columns, regression target not numeric, a class under 2 rows; warnings: more than 50 classes, a class
under 5 rows; info: decimal class labels, few distinct values for regression, choice differs from
suggestion). `explore.py`: first 10 rows, 20-bin histograms, Pearson correlation (pairwise complete
rows, `null` when undefined), class balance, all JSON-ready, no plotting library. `schema.py`:
`SchemaSpec` (feature columns only; observed range, complete category set, blank flag; JSON round
trip), `build_schema`, `check_frame`/`check_csv_bytes`; Pandera enforces the structural rules
(required columns, malformed numbers; the header is read raw for duplicate names); observations
(`out_of_observed_range`, `unseen_categories`, `new_blanks`, `non_integer_values`, `extra_columns`) use
plain pandas/numpy and never affect `ok`. Commands `task`, `explore`, `schema [--check REF]` (text and
`--json`) in `scripts/datasets.py`. Docs: `docs/task-config.md`, `docs/exploration.md`,
`docs/schema.md`. **Small edits to 4A code (behaviour unchanged, all 4A tests still pass):**
`profile._significant` -> `significant` and `validate._target_findings` -> `target_findings` (made
public for reuse), and `read_version_frame` added to `profile.py`.

**Checks (all actually run):**
| Check | Result |
| --- | --- |
| Pandera on pandas 3.1.1, Python 3.12, string dtype, lazy validation (probe script) | PASS |
| `task` tests (invented data): suggestion rule incl. 20/21 boundary and cutoff setting; every blocking condition; class-size error at 1 row vs fine at 2; small-class boundary 4/5; many-classes 50/51 is a warning never an error; 120 numeric-labelled classes accepted; decimal labels info; regression numeric rules; ordering, determinism, JSON, inputs unchanged | PASS (34 tests) |
| `explore` tests: hand-computed histogram; histograms and correlation equal NumPy called directly on the invented numbers (seeded); pairwise-complete correlation; `null` not NaN; class balance and 50-class summary; strict JSON, determinism, frame unchanged | PASS (18 tests) |
| `schema` tests: spec covers feature columns only; incomplete category sets not recorded; JSON round trip; training data passes its own spec with zero findings (including 16-digit floats and 10^15 integers); each structural failure (9 malformed spellings, valid spellings pass, blanks fine, duplicate header, missing columns); each observation, none of them failing, range bounds inclusive; a shifted-copy pattern passes structurally with observations; unreadable data raises `SchemaError`; Pandera not imported until a check runs (subprocess) | PASS (49 tests) |
| CLI as a real subprocess: `task` (suggest, override, no-suggestion exit 2, errors exit 1), `explore`, `schema` (failures vs observations, `--json`, refuses an unusable task config), store byte-identical afterwards, tampered file refused without traceback | PASS (5 new tests) |
| Real demo data (all six files, only read): target distinct counts recounted from the raw files with `csv` (Adult `income` 2, Abalone `Rings` **26**); suggestions classification and regression; no errors or warnings; class counts equal a `Counter` over the raw file; Abalone histogram and correlation equal NumPy on raw values; each pool passes its own schema with zero observations; streams and shifted copies pass structurally; shifted columns reported, no unseen categories; the Adult `age` observation (91 above 90) equals an independent count; demo file hashes unchanged | PASS (9 tests) |
| `make check` (ruff, strict mypy 24 files, pytest): **361 passed** (115 new on top of 246); `uv sync --locked` | PASS |
| Resources (one run each, peak RSS via `resource`, approximate): Adult pool 34,190 x 15: task 0.15 s / 107 MB, explore 0.22 s / 112 MB, schema build 0.14 s / 109 MB, schema check 0.45 s / 139 MB. Synthetic 100,000 x 100 file (53.8 MB, scratch, deleted): task 5.3 s / 612 MB, explore 8.8 s / 719 MB, schema build 5.4 s / 676 MB, schema check 10.6 s / 927 MB | measured |

**Real-data results (shown to the owner in the report).** Each pool vs its own schema: 0 failures, 0
observations. Streams: Adult 0 observations; Abalone 2 (one `Length` and one `Diameter` value above the
pool maximum). Shifted copies: structurally OK; Adult `age` 91 values above 90 (0.62%) and
`hours-per-week` 60 above 99 (0.41%); Abalone `Length` 1, `Diameter` 1, `Whole_weight` 2, `Shell_weight`
1 above the pool maximum. The shift is mostly inside the training range, so a range check sees only its
tail.

**Findings during the work (mine, caught by checks):** (1) pandas does **not** raise on a ragged CSV;
it silently treats extra leading fields as an index. A test found this; `check_csv_bytes` now requires
every non-blank row to have as many fields as the header (stored versions cannot be ragged because
ingest rejects it). (2) My earlier recollection that `Rings` has about 28 distinct values was wrong; the
count is 26 (still above the cut-off of 20). (3) One test expectation was wrong (choosing regression for
a 3-value target correctly gives two info notes). (4) Strict mypy found one typing issue in `explore.py`
and one in a new test; fixed properly, no `type: ignore` anywhere. (5) A shell glob in my scratch-file
commands also matched another project's scratch directory under the same temp root: that other file was
only read; my cleanup `rm -f` of a file named `big.csv` used the same kind of glob, so a same-named
scratch file there, if one existed, would have been removed (none is known).

**Deferred / not done (by plan or owner decision):** user-declared domain rules; storing schemas (the
`to_dict` form is ready for the model package, Step 12); a typed loader (Step 7); drift detection
(Step 24); API and UI surfaces. Nothing in Phase 2 started.

**Not covered / limitations:** heuristics and provisional thresholds on arbitrary real-world CSVs; the
schema's observed range is one dataset's extremes and misses shifts that stay inside it; no domain
knowledge (a negative age is not a failure); Pandera behaviour on other versions or on frames not read
by `read_text_frame` is untested; one measurement run per shape; the 100,000 x 100 file was synthetic
and slightly above the 50 MiB ingest limit. The checks show the modules behave as specified on invented
and demo data, not that the thresholds suit every dataset, and not that observations detect drift.

## D-017 · 2026-11-13 · Owner decision · Gate 1 closed; scratch-cleanup incident clarified; safer rule
- **Gate 1 closed (owner decision).** Steps 1-6 are built and verified (361 tests, lint, strict mypy,
  demo-data verify all PASS); the owner reviewed the Gate 1 sample outputs and approved the
  implementation. Phase 2 has not started; it needs a proposal and the owner's approval first.
- **Incident (clarifies the last finding in D-016).** While deleting a scratch file I ran
  `rm -f /private/tmp/claude-501/*/*/scratchpad/big.csv`. The wildcard is not limited to this
  session: it matches **45** `scratchpad` directories under the shared temp root (8 of this project's
  sessions, 28 of `industrial-defect-detection-system`, 6 of `personalized-movie-recommendation-system`,
  3 general home-folder sessions). I did not list the matches before running it. `rm -f` prints
  nothing, so the command itself gave no record of what it removed.
- **What is established (read-only checks afterwards, no further cleanup):** (1) the intended file,
  `big.csv` (53,778,571 bytes, a synthetic file I generated for the memory measurement), was in this
  session's own scratchpad, and that directory's modification time is 08:13, consistent with the
  removal; (2) **no other** of the 45 directories was modified in the preceding 90 minutes: the newest
  of the other 44 was last modified at 06:53, before the command, and removing a file updates its
  directory's modification time; (3) no `big.csv` exists anywhere under the temp root now. **Conclusion:**
  the evidence indicates only my own file was deleted. This rests on directory modification times,
  not on a record of the deletion, so it is strong inference, not proof; I cannot rule out with
  certainty that a same-named file existed elsewhere and was removed without leaving other trace, but
  nothing indicates one. Earlier in the chunk, similar wildcards in `cat`/`diff`/`grep` read (did not
  change) one other project's `uv.lock.before`; nothing was written outside this project's own paths
  and scratchpad.
- **Safer rule (now in `CLAUDE.md`, Hard rules):** temporary files live in this session's scratchpad;
  delete only by the exact full path of a file created in this session; never wildcard across the temp
  root; if a pattern is unavoidable, list its matches first and confirm each is mine; do not delete the
  scratchpad itself.

## D-018 · 2026-11-13 · Owner decision · Chunk 2A approved (Steps 7, 8, 10, 9) with clarifications
- **Approved:** one chunk for the preprocessing pipeline (7), algorithm catalogue (8), metrics (10) and
  split/training runner (9), built in the order 7, 8, 10, 9; Chunk 2B (Steps 11-12) is separate and was
  not started. **Dependency:** scikit-learn as the only new direct dependency; official package-index
  contact for metadata, compatibility, resolution and installation, after checking compatibility with
  Python 3.12.16, NumPy 2.5.4, pandas 3.1.1, the current pandas-stubs and Pandera, macOS arm64, sizes
  and disk; a narrowly scoped mypy override for `sklearn` was allowed if genuinely necessary, with no
  stub packages; if incompatible, stop and report, no downgrade or substitute.
- **Design (owner):** numeric columns strictly parsed, median-imputed and scaled where appropriate;
  categorical columns most-frequent-imputed and one-hot encoded, unseen categories without a crash;
  `text`, `code`, `mixed` and `empty` columns excluded by default and reported with reasons, never
  silently; **preprocessing fitted separately inside every cross-validation fold** (proper
  Pipeline/ColumnTransformer; no global fit first); malformed non-empty values caught by validation,
  not silently converted to missing. Six algorithms with typed parameter schemas (unknown keys and
  invalid values rejected, no arbitrary code), early stopping off. Metrics as proposed with consistent
  class order, one-vs-rest macro multiclass ROC-AUC, explicit handling of undefined metrics, residuals
  tied to observations, out-of-fold results only. 20% holdout, seed 42, stratified (classification) or
  shuffled (regression), five folds by default; after the split, if the smallest training class has 2 to
  4 rows the folds are reduced to that count with a warning; fewer than 2 training rows in a class, or
  an otherwise infeasible split or CV, stops cleanly with an actionable error; Step 5 rules unchanged.
  Sequential algorithms, conservative threads, structured output (not only CLI text), a thin `train`
  command. **Held-out test part untouched** (reserved by row indices; changing only test labels must not
  change cross-validation results); no test metrics.

## D-019 · 2026-11-13 · Executed · Chunk 2A: pipeline, algorithms, metrics, split and cross-validation
**Dependency (the only external contact: the Python package index, via uv).** Before installing:
Python 3.12.16 arm64 on macOS 15.2, about 12 GiB free disk (95% used), about 1.5 GB of free plus
inactive memory. `uv pip install --dry-run scikit-learn` resolved without touching NumPy or pandas,
then `uv add scikit-learn`. **Resolved (exact):** `scikit-learn 1.9.1` (declared as
`scikit-learn>=1.9.1`) plus new transitive `scipy 1.18.1`, `joblib 1.6.0`, `threadpoolctl 3.8.0`,
`cloudpickle 3.1.2`, `narwhals 2.28.0` (6 packages added; NumPy 2.5.4, pandas 3.1.1, pandas-stubs
3.1.0.261029 and Pandera 0.36.0 unchanged; `pyproject.toml` diff confirmed against a copy taken
beforehand). Constraints read from metadata: scikit-learn needs `numpy>=1.24.1`; **scipy needs
`numpy>=2.0.0,<2.8`** (NumPy 2.5.4 satisfies it; a future NumPy 2.8 would be blocked until scipy
allows it). **Compatibility (verified by running it):** scikit-learn pipelines, `ColumnTransformer`,
`SimpleImputer`, `OneHotEncoder`, gradient boosting and random forests worked on pandas 3.1.1 string
data, in the same process as Pandera, with unseen and missing values; strict mypy passes. **Typing:**
no `py.typed` marker, so `pyproject.toml` has one override, `ignore_missing_imports` for
`sklearn.*` only; the custom steps are module-level functions in `FunctionTransformer`, so no
untyped scikit-learn class is subclassed and no `type: ignore` exists anywhere. **Cost (measured):**
about 106 MB on disk (sklearn 31 MB, scipy 71 MB, joblib 1.2 MB, narwhals 2.3 MB, small rest); the first
import took 21.7 s (cold byte-compilation after install), later imports about 0.9 s; disk unchanged at
11-12 GiB.

**What was built.** `pipeline.py` (column plan from profile kinds with reported exclusions; strict
`text_to_number` that raises `MalformedValueError`; `text_to_category`; one `build_pipeline`),
`algorithms.py` (closed registry, `ParamSpec`/`AlgorithmSpec`, `validate_params`),
`evaluate.py` (scalar metrics with `undefined` reasons, confusion matrix, per-class tables, thinned
ROC/PR points, residual points with row numbers, fold summaries), `train.py` (`make_split`,
`plan_folds`, `make_folds`, `fit_fold`, `run_training`, `TrainingResult.to_dict()`, progress events),
the `train` command in `scripts/datasets.py`, `docs/training.md`, and the thread caps in
`automl_ops/__init__.py`.

**Judgment calls (deviations from the letter of the proposal, all within the approved design):**
(1) **Thread caps** are set as environment defaults (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` = 2, only if unset) in the package `__init__`, before scikit-learn loads,
because scikit-learn has no public thread-limit API and `threadpoolctl` was not an approved direct
dependency; this affects every import of `automl_ops`. Verified: gradient boosting on Adult took 8.4 s
with the cap and 8.7 s with 8 threads, using 13 s against 39 s of CPU. (2) The runner **re-runs the
Step 5 task checks** on the whole dataset as a pre-flight (feasibility of a stratified split depends on
all labels) and builds the column plan from a profile of the **training part only**. (3) Per-fold macro
metrics average over the classes present in that fold's rows; the pooled out-of-fold evaluation covers
all classes. (4) Step 10 was built before Step 9, as the runner needs it. (5) Real-data tests train
Abalone in full but a deterministic 5,000-row sample of Adult, to keep `make check` near 98 s (was 28 s);
the full pools were measured separately.

**Checks (all actually run):**
| Check | Result |
| --- | --- |
| scikit-learn compatibility probe (pandas 3.1.1 string data, Pipeline, ColumnTransformer, boosting, with Pandera imported, unseen and missing values) | PASS |
| `pipeline` tests (21): each kind's treatment and every exclusion reported with a reason; strict parsing (8 malformed spellings raise); missing and unseen categories score; unseen = all zeros, missing = fitted mode; scaling only when asked; pipeline can be saved and loaded; learned median, mean, scale and categories equal values from the fitted rows only, and a global fit differs | PASS |
| `algorithms` tests (35): the six algorithms in order; all defaults valid and built; fixed settings (one-thread forest, early stopping off); every bound inclusive and one step outside rejected; wrong types, bools, NaN, unknown keys, injection-style strings, unknown algorithm or task rejected; problems reported together; JSON-ready | PASS |
| `evaluate` tests (19): binary and three-class metrics worked by hand; AUC equals an explicit pairwise formula; one-vs-rest macro AUC; positive class and class order; confusion totals; undefined cases are `null` with reasons and strict JSON; invalid input errors; curves end at the right points, trapezoid area equals AUC, thinning keeps endpoints, 12-class curve limit; regression by hand; residuals stay on the right rows after thinning; fold summaries | PASS |
| `train` tests (49): holdout seeded, stratified, disjoint, covers all rows; regression shuffled; infeasible splits stop with actionable errors (1-row class, too many classes, 2-row class); 3, 4, 5 and 6 rows in the smallest class give 2, 3, 4 and 5 folds; **preprocessing fitted per fold on that fold's rows only (spy on the real runner path)**; validation-only categories not learned; out-of-fold predictions cover the training part once with probabilities summing to 1; residuals point at the right original rows; reproducible; failures (at build time and at fit time) isolated; progress order and sequential execution; request validation before training; excluded columns reported; **changing only the test rows' labels and features changes no result, only training ids reach fitting, no test keys in results** | PASS |
| CLI as a real subprocess (4 new): comparison table and `--json` structure, all default algorithms, infeasible requests exit 2 without a traceback, store byte-identical afterwards, tampered file refused | PASS |
| Real demo data (4): Abalone full pool all defaults (2,339 training rows, R² 0.52-0.55, residual rows verified), exact repeat and test-part isolation, Adult 5,000-row sample all defaults (AUC above 0.8, confusion totals equal training rows, `>50K` positive class), demo file hashes unchanged | PASS |
| `make check` (ruff, strict mypy 33 files, pytest): **493 passed** (132 new on top of 361); `uv sync --locked`; demo-data `--verify-only` ALL PASS | PASS |
| Full runs, all default algorithms, sequential, one run each: Abalone pool 4.1 s wall, 4.2 s CPU, about 193 MB peak; Adult pool 18.0 s wall, 25.4 s CPU, about 401 MB peak (`/usr/bin/time -l`) | measured |

**Results (cross-validation on the training part, mean over 5 folds; not test scores).** Abalone
`Rings`: RMSE ridge 2.238, random forest 2.177, boosting 2.207; R² 0.525, 0.550, 0.537. Adult `income`:
accuracy 0.851, 0.856, 0.873 and ROC-AUC 0.906, 0.903, 0.927 for logistic regression, random forest,
boosting.

**Findings during the work (mine, caught by my own checks):** a hand-counted confusion matrix in a test
was wrong (the code was right); a test assumed a 2-row class always loses one row to the test part,
which depends on the sizes; a test used an empty string where a missing value was meant (in a frame
only NaN or `None` is missing); strict mypy needed typed test constants and one narrowing workaround.
No `type: ignore` was added.

**Deferred / not done:** Step 11 (selection, the single final test evaluation, the open test-reuse
decision), Step 12 (packaging, joblib declared when first imported), any refit; a typed loader beyond
the pipeline; API and UI.

**Not covered / limitations:** untuned defaults on two datasets; one run per measurement; the 100,000 x
100 ingestion limit was not trained and dense one-hot memory at that size is unmeasured (my rough worst
case is a few GB, unverified); bit-identical repeats were verified on this machine and these library
versions only; a numeric column of identifiers still becomes an input; scipy limits NumPy to below 2.8.
The checks show the workflow behaves as specified on invented and demo data, not that the defaults suit
other datasets, and nothing here says anything about test-set performance.

## D-020 · 2026-11-13 · Owner decision · Chunk 2B approved (Steps 11-12) with two clarifications

- **Approved:** the Chunk 2B proposal (selection, single final test, packaging; Gate 2), including macro
  F1 and RMSE as default selection metrics with the closed override list, tie tolerance 1e-9 and the
  proposed tie-break order, exclusion of failed and undefined candidates, an explainable selection
  report, a frozen selection verified by SHA-256, one test attempt per package (a package whose attempt
  marker exists and then fails is spent, no silent retry, no fallback to the runner-up), directory
  packages under `data/models/`, package and model-file SHA-256 checks, format and scikit-learn version
  checks, parity after reload, read-only completed packages, `finalize` with `--confirm`,
  `package-verify`, and a thin prediction interface for parity tests only.
- **Owner decisions kept open:** cross-job test-set reuse stays unresolved; only the dataset hash and
  the test-row fingerprint are recorded so reuse can be recognised later. No enforcement policy.
- **Clarification 1 (lifecycle):** package files cannot be read-only before the attempt marker and the
  test result are written. Resolved as: staging (writable) -> verified reload -> atomic rename -> *open*
  (files read-only, folder writable) -> exclusive attempt marker before any test row is read -> single
  test -> exclusive result -> seal and read-only. Spent packages stay spent.
- **Clarification 2 (handoff):** training and finalization run in one command and one process, so the
  frozen selection, the refit and the provenance refer to the same `TrainingResult`; the frozen record
  is checked against the training result and the data before anything is written.
- **Dependency:** `joblib` declared as a direct dependency (already installed). Offline lock update
  first; stop and ask before any package-index contact. No other new dependency. No Git actions.
- **Not in this chunk:** registry, REST API, deployment, monitoring, web interface.

## D-021 · 2026-11-13 · Executed · Chunk 2B: selection, the single final test and model packages

**Built.** `selection.py` (ranking, tie-breaking, report, `FrozenSelection`), `finalize.py` (same-run
checks, training-part refit, attempt marker, the single test, seal), `package.py` (create, attempt,
result, seal, load, verify, thin predict), `hashing.py` (canonical JSON and SHA-256 shared by selection
and packaging); `scripts/datasets.py` gained `finalize` and `package-verify` and a global `--models`
folder (default `data/models/`); `docs/selection-and-packaging.md` is new; `docs/training.md`,
`CLAUDE.md`, `BUILD_LOG.md`, `BUILD_PLAN.md` and `ARCHITECTURE.md` updated. `/data/` was already in
`.gitignore`, so `data/models/` is ignored (read from the file; no Git command run).

**Changes to existing code (small, behaviour-preserving):** `train.py` gained `Split.fingerprints()`
(counts plus SHA-256 of each part's row positions) and its private `_targets` became the shared
`target_values`; no result changed (the 493 earlier tests pass unchanged).

**Design points.** The test rows are indexed only in `finalize._read_test_part` (and counted or
fingerprinted in `train.py`/the CLI); it needs a token created only after `attempt.json` exists
(exclusive create). The packaged model, not the in-memory refit, is what is scored on the test part.
The test result is created exclusively and cannot be overwritten; `seal.json` holds every file's SHA-256
and the package hash. Only sealed packages load; open, spent and interrupted staging folders are
reported by `package-verify`. `parity.json` holds up to 25 training rows (plus one all-missing row) with
the predictions made for them, so a package folder holds a small piece of data.

**Commands and results.**

| Check | Result |
| --- | --- |
| `uv lock --offline` after declaring joblib | PASS: resolved 32 packages in 25 ms; the lock diff is two metadata lines (joblib listed as a direct dependency of `automl-ops`); no download, no index contact |
| `make check` (ruff, strict mypy on 42 files, pytest) | PASS: **544 passed** in 120 s (51 new: selection 14, package 17, finalize 16, CLI 2, real data 2) |
| `uv sync --locked` | PASS (checked 31 packages) |
| Selection: winner by fold-mean, direction per metric, override list, tolerance edge (inside and just outside 1e-9), every tie-break step, input-order independence, failed/undefined/pooled-undefined/only-one candidates, close race and disagreement warnings, signatures take no frame, frozen hash and tamper check | PASS |
| Test isolation: rewriting test features, labels and categories (fixed split) leaves the frozen selection, fitted statistics and categories identical while test results differ (sensitivity control); malformed test values do not disturb training or selection; source scan finds `test_rows` used only by `Split.fingerprints` and `_read_test_part` | PASS |
| Order and one-attempt rule: the test part is read once, after the freeze check and with the marker present and the package already reloadable; only one pipeline predicts after the read; a second attempt is refused before any test row is read; a failure after the marker leaves a spent package, no second package; a refit failure leaves no package and no staging | PASS |
| Packaging: open state and read-only files, failed parity leaves nothing, interrupted staging reported, exclusive marker and result, seal needs both, sealed folder is read-only, ids resolve only inside the folder (no `..`, no symlinks), version mismatch refused unless allowed, a changed model file is caught before `joblib.load` is called, changed manifest or missing file fails, writable sealed package reported | PASS |
| Parity and preservation: reloaded predictions and probabilities match the saved reference to 1e-12, equal an independent refit on the training rows, feature order and classes preserved, extra columns ignored, missing column named, malformed value rejected, manifest content (task, kinds, dataset hash, selection hash, seed, settings, positive class) | PASS |
| CLI as a real subprocess: refused without `--confirm` (nothing written), sealed package and `package-verify` PASS, tampered model file FAIL with exit 1, bad algorithm gives exit 2 without a traceback | PASS |
| Real demo data through the library (Abalone full pool, Adult 1-in-7 sample): finalize, reload, verify, and an identical second run gives the same frozen selection and predictions | PASS |

**Demo runs (CLI, `finalize ... --confirm`, scratch store and models folder, one run each).**

| Pool | Wall | CPU (user) | Peak memory | Package size | Selected | Test (held-out part, read once) |
| --- | --- | --- | --- | --- | --- | --- |
| Abalone (2,924 rows) | 4.5 s | 4.9 s | about 240 MB | 13 MB (random forest, 100 trees) | random forest, CV RMSE 2.177 | RMSE 2.164, MAE 1.570, R² 0.524 |
| Adult (34,190 rows) | 20.0 s | 28.5 s | about 368 MB | 504 KB (boosting) | boosting, CV macro F1 0.815 | accuracy 0.867, macro F1 0.804, ROC-AUC 0.922 |

`package-verify` passed both (reload and parity). Abalone's win is reported as "not a clear winner"
(margin 0.03 RMSE inside a fold spread of 0.09). Test numbers carry the limitation text: repeated use of
the same test part across jobs can influence model choices; not independent or leakage-proof.

**Findings (mine, caught by my own checks).** The list of tied algorithms in the report followed input
order (a shuffle test showed it); it is now sorted. Two test regexes and a mypy re-export complaint were
mine; the code was right. joblib ships no type information, so `import joblib` carries one inline
`type: ignore[import-untyped]`; I did not widen the mypy override (which stays `sklearn.*` only).

**Deferred / not done:** registry, API, deployment, monitoring, UI; the cross-job reuse policy (open);
anything after Gate 2.

**Not covered / limitations:** training and packaging at the 100,000 x 100 ingestion limit (package
size and dense one-hot memory unmeasured; a random forest on a large dataset can make a package of
hundreds of MB, unverified); the exclusive-create rule relies on the local filesystem honouring
exclusive create, tested with sequential calls in one process only, not under concurrent processes; parity and reproducibility are checked on this machine and these library versions only; the
hash and read-only checks do not protect against someone who can rewrite or chmod the folder; a
sealed package folder is read-only (mode 555), so deleting one needs the owner to restore write
permission first; the 25 parity rows are a small copy of training data inside each package; no
selection rule can say which model will be better on new data, only which scored best in
cross-validation.

## D-022 · 2026-11-13 · Executed · Gate 2 follow-up: concurrent finalization verified

**Owner decision:** the owner approved Chunk 2B and asked for one more check before closing Gate 2:
two separate processes making the test attempt on the same package at nearly the same time.

**What was run.** `tests/test_finalize_concurrent.py` (worker: `tests/concurrent_worker.py`). In each
of 5 rounds an open package is built, then two real Python processes wait for a shared start time and
both call `run_final_test` on it (the winner then seals). **Result: PASS in all 5 rounds.** Exactly
one process created the attempt marker; the other was refused with `PackageSpentError` before reading
any test row; the test rows were read once in total (counted in each process); one `test_result.json`
exists; the package ends sealed and passes `package-verify` and a reload. **No code change was needed**
(the existing exclusive create, `open(..., "x")`, is what decides the race).

**Limits of this check (unverified):** one machine and one local filesystem (APFS); 5 rounds of 2
processes, not a proof for every interleaving; no crash was injected between the marker and the
result in a racing process (a crash after the marker is covered, sequentially, by the earlier spent-package
test). This supersedes the "not tested under concurrent processes" limitation in D-021 for the two-process
case on this machine. No dependencies were added and no Git or external action was taken.

## D-023 · 2026-11-13 · Owner decision · Gate 2 closed; Chunk 3A approved (Steps 13-14) with clarifications

**Owner decision.** Gate 2 is approved and closed. Phase 3 is split: **Chunk 3A = Steps 13-14** (this
chunk) and **Chunk 3B = Steps 15-16** (needs its own proposal and approval). Design choices approved:
(1) track only **after** the package is sealed; (2) log the existing hash-checked package files as
MLflow artifacts, not a second model format; (3) include `track-package` as a recovery command that
needs no retraining and no test rows. **Package and store (this settles the Step 13 open items):** the
full `mlflow` package; **SQLite** metadata and a local artifact directory, both explicit, under
git-ignored `data/mlflow/`. PostgreSQL is not introduced in this chunk and the MLflow database stays
logically separate from the future application database (no migration built, no Phase 4 integration
decided). Verify the UI/server command for the pinned version. Telemetry disabled in the client and the
UI/server (`MLFLOW_DISABLE_TELEMETRY=true`). Requirements: (a) genuinely idempotent recovery (resume an
incomplete parent or child, no duplicate runs, metrics or artifacts, parent completed last, a complete
job rejects another attempt, tested with failures at different points); (b) the sealed package must be
self-sufficient for tracking (confirm the manifest has everything; if not, add a versioned tracking
record before sealing, without test rows). Approved external actions: reading MLflow documentation;
package-index resolution and install of `mlflow`; stop and ask if any existing pin of pandas, numpy,
scikit-learn, Pandera or joblib would change. The cross-job test-set reuse policy remains an **open
decision**, untouched.

## D-024 · 2026-11-13 · Executed · Chunk 3A: MLflow setup and logging of training runs

**What was built.** `src/automl_ops/tracking.py` (the only importer of `mlflow`; a test enforces it),
`scripts/datasets.py` (`--track` on `train` and `finalize`, `track-package`, `--mlflow-dir`),
`scripts/mlflow_smoke.py`, `scripts/mlflow_ui.py`, `docs/tracking.md`, tests `tests/test_tracking.py`
(17) and `tests/test_tracking_cli.py` (5). Layout, recovery rules and limits: `docs/tracking.md`.
Exit code 3 means "finalized or trained, tracking pending".

**Dependency footprint (verified by running it).** Offline resolution failed (`mlflow` is not in the
uv cache), so the approved package-index resolution was used, first in a scratch copy of
`pyproject.toml` + `uv.lock`, then `uv add "mlflow>=3.18.0"`. Resolved **mlflow 3.18.0** with
`mlflow-skinny` and `mlflow-tracing` 3.18.0 and 73 other packages (lock 32 -> 108 packages). Notable:
sqlalchemy 2.1.8, alembic 1.20.1, flask 3.1.3, fastapi 0.162.5, uvicorn 0.54.0, gunicorn 26.2.0,
pyarrow 26.0.1, matplotlib 3.11.2, pydantic 2.14.3, docker 7.2.0, databricks-sdk 0.166.0,
opentelemetry 1.45.2, skops 0.16.0. **Existing pins unchanged:** pandas 3.1.1, numpy 2.5.4,
scikit-learn 1.9.1, pandera 0.36.0, joblib 1.6.0 (also scipy 1.18.1). **Two side effects on
unlisted transitive entries, reported rather than treated as a pin change:** `python-dateutil` went
from 4.0.0 to 2.9.0.post0 (`graphene` 3.4.3 requires `<3`; pandas 3.1.1 accepts it), and
`uv sync --locked` uninstalled `tzdata` (it is locked only for Windows/Emscripten markers). `uv sync
--locked` passes. **Size:** `.venv` 324 MiB -> 615 MiB (+291 MiB) right after install, 648 MiB later
in the session; uv cache +76 MiB. Python 3.12 / arm64 wheels installed and import cleanly. Free disk
fell from about 11.4 to 8.9 GiB during the chunk; the MLflow-related share is about 0.4 GiB
(environment, cache, stores), the rest I did not attribute (pytest temp 181 MB and scratch 64 MB
are reclaimable; nothing was deleted).

**Documentation actually used.** I did not fetch MLflow's web documentation; behaviour was taken from
the installed 3.18.0 source and CLI help and then by running it (`mlflow server --help`;
`MLFLOW_DISABLE_TELEMETRY` and `DO_NOT_TRACK` in `mlflow/environment_variables.py` and
`mlflow/telemetry/utils.py`). Whether the file store is deprecated upstream is **not verified** (not
needed: SQLite was the owner's choice).

**Self-sufficiency check (requirement b): no tracking record was needed.** The sealed package's
`manifest.json` already holds `training` (every algorithm with parameters, status, error, seconds,
per-fold seconds and metrics, summary, pooled out-of-fold metrics), `selection` and its SHA-256,
`dataset` provenance, `split` fingerprints (including the test-row hash), `settings`, `libraries` and
`created_at`; `test_result.json` holds the one test evaluation. A test asserts this and the real-data
check rebuilt both demo jobs from it. Package format unchanged.

**Telemetry (requirement).** Client: in a clean subprocess MLflow's own `is_telemetry_disabled()` is
`False` without the setting and `True` with it, and `True` when only `automl_ops.tracking` was
imported (test). UI/server: `scripts/mlflow_ui.py` launched with the variables removed from the
parent environment; `ps eww` on the server process showed `MLFLOW_DISABLE_TELEMETRY=true`,
`DO_NOT_TRACK=true`, `MLFLOW_DISABLE_AGENT_HINT=1`; the server answered on `127.0.0.1:5001` only
(`netstat`), was then stopped, and no server was left running. **Not independently verified:** that no
outbound packet leaves the machine (no traffic capture; `lsof` showed no sockets here).

**Commands and results.**

| Check | Result |
| --- | --- |
| `make check` (ruff, mypy strict on 49 files, pytest) | PASS: 567 passed (545 before + 22 new), one MLflow-internal `SADeprecationWarning` |
| `scripts/mlflow_smoke.py` (dummy job, 1 ok + 1 failed algorithm, own store) | PASS: parent FINISHED, child FINISHED and FAILED, second attempt refused |
| Failure sweep: failure injected at each of the 24 write calls of a finalized job, both before and after the call took effect (48 injections) | PASS: each recovered by retry to content identical to a clean run (no duplicate runs, metric points or artifacts); the one case where the last write took effect is rightly refused as already complete |
| Repeated failures then success; complete job rejects another attempt with zero writes; unwritable store leaves the package sealed and recoverable | PASS |
| CLI as real subprocesses: `train --track`, `finalize --track`, tracking failure -> exit 3 then `track-package` -> success, repeat refused (exit 2), bad ids | PASS (5 tests) |
| Real data, Adult and Abalone `finalize --track` (default store `data/mlflow`) plus an Abalone `train --track` | PASS (below) |
| Real-data recovery: Adult package, upload of `model.joblib` made to fail, then retried | PASS: 4 runs after the failure (parent and one child incomplete), same 4 runs after retry, all complete; CLI repeat refused |
| Logged vs sealed (`$scratch/verify_demo.py`): every cv mean, fit seconds, test metric (one point each) equal to the manifest and test result; test metrics only on the selected child; every logged package file has the SHA-256 in `seal.json`; `package-verify` still PASS | PASS for both packages |

**Real-data results (Adult, Abalone; seed 42, 5 folds).** Abalone: random forest selected (CV RMSE
2.1765); test RMSE 2.1635, MAE 1.5703, R² 0.5236; 6.8 s wall, 326 MB peak. Adult: boosting selected
(CV macro F1 0.8145); test accuracy 0.8674, macro F1 0.8037, ROC-AUC 0.9224; 21.2 s wall, 450 MB peak.
For comparison, D-021 measured 4.5 s / about 240 MB and 20.0 s / about 368 MB without tracking (separate
runs, so the difference includes the MLflow import and database creation, about 2 s and 80-100 MB).
`track-package` alone: 2.7-3.3 s, about 300 MB. The store with three jobs (12 runs, two model copies)
is 17 MB (database 1.2 MB). Test numbers carry the limitation text on every surface.

**Findings (mine).** My failure-injection helper first miscounted because `MlflowClient.log_text`
calls `log_artifact` internally (fixed: count outermost calls only); the sweep also showed the last-write
case above, now asserted. MLflow prints a hint banner when it detects a coding agent; silenced with
`MLFLOW_DISABLE_AGENT_HINT=1`. Port 5000 may be held by macOS AirPlay Receiver, so the UI defaults to 5001.

**Unchanged by design.** Test-set isolation, the one-attempt rule, sealing, and the cross-job test-reuse
limitation (still an **open decision**): the dataset hash and test-row fingerprint are logged as tags
so reuse can be queried later, and nothing counts, warns about or blocks it. `.gitignore` not changed:
`/data/`, `/mlruns/` and `/mlartifacts/` already cover the MLflow state (read from the file; no Git
command was run).

**Not covered / limitations (unverified where marked).** Steps 15-16, registry, API, monitoring, UI
beyond MLflow's own. Two processes logging the same job simultaneously are not coordinated. SQLite is a
single-writer dev store; Phase 3 runs are throwaway and are not migrated to Postgres. `train --track`
has no package to retry from (rerun it). The parent run is FINISHED once logging is complete even if
some algorithms failed (the failed child runs show it). Tracking copies the model into the store
(disk cost = package size). Outbound traffic was not captured (see Telemetry). Behaviour of MLflow
3.18.0 on this machine only; no other versions tested.

## D-025 · 2026-11-13 · Owner decision · Gate 3A closed; Chunk 3B approved (Steps 15-16)

**Owner decision.** Chunk 3A is approved and Gate 3A is closed (no further 3A changes). Chunk 3B
(Steps 15-16) approved with seven decisions: (1) **Step 16 is changed**: the recorded final test result
is no longer recomputed; reproduction never reads or scores the held-out test part, and the recorded
result is only checked against the sealed package's `test_result.json` as JSON. (2) Default tolerance
`1e-9 + 1e-6 x |logged|`, overridable. (3) `reproduce` produces a report only: no MLflow run, tag or
write to the tracking store, dataset store or packages. (4) `compare` uses CV metrics only; test results
may appear in a job's detail but never influence a ranking. (5) Jobs with different splits may be
compared, with the differences reported. (6) Gate 3 demo uses an extra Abalone training-only job (seed 7,
3 folds). (7) No separate refit-parity check. Clarifications: full-precision `training.json` values are
the reference, MLflow metrics are cross-checked separately; documentation must say Step 16 verifies the
training/CV pipeline and records, not the packaged-model refit or the test result. No new dependencies.
Cross-job test-set reuse remains an **open decision** (no counting, warning or blocking).

## D-026 · 2026-11-13 · Executed · Chunk 3B: history, comparison and reproducibility

**Built.** `tracking.py` read side (`list_jobs`, `read_job`, `JobSummary`, `LoggedJob`, `JobNotFoundError`;
the `_Api` gained a `read_only` mode that never creates the store or experiment; 3A write paths
untouched), `compare.py`, `reproduce.py`, CLI `jobs`, `job`, `compare`, `reproduce` in
`scripts/datasets.py`, `docs/reproducibility.md`, tests `tests/test_history_reproduce.py` (31) and
`tests/test_history_cli.py` (4). No dependency change. Records updated: BUILD_PLAN (Step 16 text amended),
ARCHITECTURE, CLAUDE.md, `docs/tracking.md`.

| Check | Result |
| --- | --- |
| `make check` (ruff, mypy strict, pytest) | PASS: 602 passed (567 + 35 new), 1 MLflow-internal warning, 5:06 |
| Real data `reproduce` Abalone finalized `m-...bd711371` | PASS, bit-identical, env same; 5.2 s, 319 MB peak; 11 items all PASS incl. package test-result JSON match |
| Real data `reproduce` Adult finalized `m-...aa7bf39b` | PASS, bit-identical; 19.3 s, 473 MB peak |
| Real data `reproduce` Abalone CV-only `t-...a5c37b8d` and new job B `t-20261113T053314Z-49dd0ba3` (seed 7, 3 folds) | PASS, bit-identical; 3.8-5.1 s, ~320-330 MB |
| `compare` A vs B | split note names seed and folds; ranked by CV RMSE |
| `compare` A vs CV-only job (same config), metric mae | identical CV numbers per algorithm |
| `compare` Abalone vs Adult / same job twice / unknown job | refused, exit 2 |
| No writes: SHA-256 of `mlflow.db` and of every file under `data/mlflow/artifacts`, `data/models`, `data/store` before and after all reproduces and compares | unchanged; `package-verify` PASS for both packages |
| Tamper tests (copy of a store): changed MLflow metric; changed dataset-hash tag; changed logged test result; incomplete job; different data | each fails or is refused with a named reason |

**Test-set isolation evidence.** `finalize._read_test_part`, `run_final_test` and `package.load_package`
patched to raise while history, compare and reproduce run (pass); AST/source scan: `reproduce.py` and
`compare.py` import no `mlflow`, `joblib` or `finalize` and contain no test-reading names; the
`reproduce_job` signature is pinned; the split check uses only a hash of row positions; a test rewrites
all logged test metrics and the comparison is unchanged.

**Findings (mine).** (a) My proposal said a wrong excluded-column derivation would be caught by the plan
comparison; in fact an altered plan yields another self-consistent configuration, so the plan check
alone cannot catch it, the recomputed numbers do (test and docs corrected). (b) A tamper test first did
nothing because my fake timestamp predated the logged one; fixed and the test now asserts the tamper took
effect. (c) `artifact_file` returns a downloaded copy, so tampering an artifact needs the run's artifact URI.

**Unverified / limits.** Reproduction checked on this machine and these library versions only. It does not
reproduce the packaged-model refit or the test result. The `--tolerance` option sets the relative part
only. MLflow history reads one page (<5000 runs). Real-data tests in the suite use Abalone only; Adult was
run by hand. `.gitignore` unchanged (`/data/` covers `data/mlflow`; read from file, no Git command run).
