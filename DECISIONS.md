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
