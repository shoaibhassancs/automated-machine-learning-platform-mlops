# History, comparison and reproducibility

BUILD_PLAN Steps 15 and 16 (Chunk 3B, Gate 3). Code: `tracking.py` (the read side: `list_jobs`,
`read_job`), `compare.py`, `reproduce.py`; command line: `scripts/datasets.py`. Everything here is
**read-only**: nothing is written to the MLflow store, the dataset store or the model packages, and
no new MLflow run is created.

```bash
uv run python scripts/datasets.py jobs [--kind train|final] [--task T] [--dataset REF] [--algorithm A] [--state S] [--limit N] [--json]
uv run python scripts/datasets.py job JOB_KEY [--json]
uv run python scripts/datasets.py compare JOB_KEY JOB_KEY... [--metric M] [--json]
uv run python scripts/datasets.py reproduce JOB_KEY [--tolerance X] [--json]
```

`JOB_KEY` is the package id (`m-...`) of a finalized job, or the `t-...` key printed by
`train --track`. `--mlflow-dir`, `--store` and `--models` go before the command name.

## History (Step 15)

* `jobs` lists the tracked jobs, newest first: kind, logging state, dataset, seed, folds, selected
  algorithm with its cross-validation score, algorithms ok/failed, and whether a test result is
  recorded. A store that does not exist gives an empty list and is **not** created.
* `job` shows one job: configuration, library versions, each algorithm's cross-validation numbers,
  the selection explanation and, for a finalized job, the recorded single test result with its
  limitation text. This is the only place a test result is shown.
* `compare` puts the algorithms of two or more jobs side by side for one metric (default: the
  metric the jobs were selected on, else macro F1 / RMSE) and ranks them by the **cross-validation
  mean** in the metric's direction. Test results are never used, ranked or shown by `compare` (a
  test rewrites every logged test metric and shows the comparison is unchanged).
* Jobs of different task or target are refused. Jobs with a different dataset, seed, fold count or
  test fraction **are** compared, with a plain note naming each difference, because their
  cross-validation scores then come from different splits and are not directly comparable. Differing
  parameters and library versions are listed. This note is about the validity of the comparison;
  it says nothing about reuse of the held-out test part, which stays an **open decision** (no
  counting, warning or blocking anywhere).
* The values come straight from the MLflow records written in Chunk 3A (tags, params, metrics and the
  `training.json`, `provenance.json`, `selection_report.json` and `test_result.json` artifacts); no
  MLflow object leaves `tracking.py`. Tests compare the output with direct MLflow client reads and
  with the sealed package's manifest.

## Reproducibility (Step 16)

`reproduce JOB_KEY` rebuilds the job's configuration **from its MLflow record only** (so training-only
jobs work too), re-runs cross-validation through `train.run_training` (the one training
implementation) on the same dataset version, and compares the result with the record.

**Configuration** comes from the tags and the stored `training.json`: dataset reference and SHA-256,
task, target, seed, folds, test fraction and every algorithm's validated parameters. `train` records the
column plan but not `--exclude` itself, so excluded columns are derived as *dataset columns that are
neither the target nor in the logged plan*; the re-run's own plan is then compared with the logged one.
(That check cannot expose a record whose plan was edited into another valid configuration; the
recomputed numbers do, and a test shows it.)

**Compared (all must pass):**

| item | what |
| --- | --- |
| dataset | the loaded version's SHA-256 equals the logged one, else the check **cannot run** (exit 2) |
| record consistency | every MLflow metric (per-fold, mean, std, pooled out-of-fold, test) equals the value in the job's own full-precision `training.json` / test result, within the tolerance |
| configuration | settings, split sizes, folds used, warnings, classes, class counts, column plan |
| each algorithm | status (and error type for a failure), parameters, per-fold, summary and pooled out-of-fold metrics |
| selection (finalized jobs) | selected algorithm and ranking order recomputed from cross-validation only; the frozen-selection hash is reported (identical or not, informational) |
| split fingerprint (finalized jobs) | the re-made split's test-row **position** hash equals the logged `platform.test_rows_sha256`; no row is read |
| recorded test result (finalized jobs) | the MLflow copy equals the sealed package's `test_result.json`, compared as JSON; the package seal hashes are verified first, the model is not loaded |

Timings (`seconds`, `fold_seconds`) are never compared. **Reference values:** where `training.json`
holds full-precision numbers (it does for every job) those are authoritative; the MLflow metrics are
cross-checked separately, so a rounded display value cannot cause a false failure.

**Tolerance (provisional).** A value passes if `|new - logged| <= 1e-9 + 1e-6 x |logged|`;
`--tolerance X` replaces the relative part. The report states the largest difference and whether the
result was bit-identical. On this machine, same libraries, the real jobs below reproduced bit for bit.
Everything that is not a number (statuses, names, text, hashes, bools) must be exactly equal.

**Environment.** The current library versions are compared with the logged `lib.*` tags. A
difference never fails the check but changes the verdict to "REPRODUCED under a different
environment" and is listed. Other hardware or library versions may legitimately give small numerical
differences; that has been tested only on this machine (MacBook Air M1, one set of pinned versions).

**Exit codes.** 0 reproduced; 1 not reproduced (the output names the metrics or items); 2 cannot run
(unknown or incomplete job, dataset version missing or altered, different dataset hash, a package that
fails its seal check). If the sealed package folder is absent the package-vs-MLflow test-result
comparison is shown as `SKIP ... NOT CHECKED`, not as a pass.

## What this does and does not establish

* **Establishes:** the training and cross-validation pipeline, run again from the recorded
  configuration on the same data, gives the recorded cross-validation results and the same
  selection; the recorded test result in MLflow is the sealed package's copy.
* **Does not:** reproduce the final packaged-model refit; re-evaluate or re-score the held-out test
  part (no test row is read and nothing is scored on it); prove the same results on other hardware or
  library versions; say anything about reuse of the test part across jobs.
* BUILD_PLAN Step 16 originally said the recorded final test result would be recomputed. That was
  **changed by owner decision** (D-025): reproduction never touches the test part; the recorded result
  is only checked against the sealed copy, as JSON.

## Isolation, in code

`reproduce.py` and `compare.py` import neither `mlflow` nor `finalize` nor `joblib` (a test parses
their imports and source); `reproduce_job` takes a frame, a hash, library versions and already-read
JSON, no model path, test positions or store. Tests also patch `finalize._read_test_part` and
`package.load_package` to raise while history, comparison and reproduction run, and hash the whole MLflow
store and the model packages before and after to show nothing changed.

## Cost

One re-run costs one training run: Abalone about 4-5 s and 320 MB peak; Adult about 19 s and
470 MB peak (measured, D-026). `jobs`, `job` and `compare` take about 2-3 s (MLflow import).
