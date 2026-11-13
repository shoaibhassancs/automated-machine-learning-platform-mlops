# Selection, the single final test, and model packages

BUILD_PLAN Steps 11 and 12 (Chunk 2B, Gate 2). Code: `src/automl_ops/selection.py`, `finalize.py`,
`package.py` and `hashing.py`; command line: `scripts/datasets.py finalize` and `package-verify`.
All numbers below marked *provisional* are defaults that can change with the owner's approval.

```bash
uv run python scripts/datasets.py finalize adult-train-pool@v1 --target income --confirm
uv run python scripts/datasets.py finalize abalone-train-pool@v1 --target Rings --confirm --json
uv run python scripts/datasets.py package-verify            # every package in data/models/
uv run python scripts/datasets.py package-verify m-20261113T120000Z-1a2b3c4d
```

`finalize` runs training and finalization **in one process**: the same `TrainingResult` feeds the
selection, the freeze, the refit and the package, so they cannot refer to different runs. It refuses
to run without `--confirm`, because it reads the held-out test part. `train` is unchanged and still
never reads the test part. Exit codes: 0 finalized, 2 a refused or failed run (the message says why);
`package-verify` exits 1 if any package fails.

## Step 11: selection

`select_model(training, metric=None)` receives **only** the `TrainingResult`. It has no data frame and
no test row positions, so no test information can influence the ranking, a tie or the frozen record
(a test pins the function signatures, and another rewrites the test rows and shows the frozen hash is
unchanged).

- **Metric.** Default macro F1 (classification) or RMSE (regression). Closed list of overrides:
  classification `f1_macro, accuracy, precision_macro, recall_macro, roc_auc`; regression
  `rmse, mae, r2`. RMSE and MAE are lower-is-better, the rest higher-is-better. Candidates are ranked
  by the **mean over the folds**.
- **Eligible** candidates finished, have out-of-fold predictions, and have the metric defined in every
  fold and on the pooled out-of-fold rows. Others are listed under `excluded` with the reason (for
  example "roc_auc is undefined in 2 of 3 folds"). If none is eligible, selection stops with an error.
- **Ties.** Scores within 1e-9 are tied. Broken in order by: the pooled out-of-fold value of the
  metric; a secondary metric (accuracy, or MAE; F1 / RMSE when that is the primary); the smaller
  fold standard deviation; the fixed catalogue order. Fit time is never used.
- **Positive class.** For two classes it is `classes[1]`, the second sorted label. It is recorded in
  the selection report, the frozen selection, the manifest and the test result.
- **The report** (`SelectionReport.to_dict()`, also stored in the manifest) holds the ranking (mean,
  standard deviation, pooled value, which rule decided each place), the excluded candidates, the
  margin to the runner-up, whether the pooled values would have ranked someone else first, warnings,
  and one plain sentence. Warnings include "not a clear winner" when the margin is within the
  winner's fold-to-fold standard deviation, and "selected by default" when only one candidate
  completed. A close race is reported, not hidden.
- **Freezing.** `freeze_selection` produces an immutable `FrozenSelection`: canonical JSON of the
  algorithm, validated parameters, seed, settings, column plan, class order, positive class, split
  fingerprints (counts and SHA-256 of the row positions of each part) and the full report, plus its
  SHA-256. `verify()` fails if the text or the hash was changed.

## The final test (Step 11)

`finalize` does, in order: check `--confirm`; verify the frozen hash; check that the frozen selection,
training result and data are one run (split fingerprints, column plan, settings, row count, dataset
hash); refit a **fresh** pipeline on the training part only; build and verify the package; create the
attempt marker; only then read the test part, score the packaged model once and write the result;
seal the package.

- Only the selected model is scored on the test part. No other candidate is ever fitted on it,
  predicted on it, or compared on it.
- Nothing can change parameters or the selection after the freeze: there is no code path from a test
  score back to a candidate.
- **No fallback.** If the refit or the test fails, the run stops. A failed refit leaves no package. A
  failure after the attempt marker leaves a **spent** package that is never retried and never loadable.
- **Enforced in code and tests:** the test rows are indexed in exactly one function
  (`_read_test_part`, which needs a token only the attempt-marker step can create); a test parses the
  source and fails if any other code touches them; tests record the order of events, that exactly one
  pipeline predicts after the read, and that a second attempt is refused before any test row is read.
- **Open decision, not settled here: reuse of the test part across jobs.** The same dataset version
  and seed give the same test rows in every job. Nothing in this chunk allows, forbids or counts
  that. Each package records the dataset SHA-256 and the test-row fingerprint so reuse can be
  recognised later, and every test result states: the same test part can be reused by other jobs,
  repeated use can influence model choices, and the result is not independent or leakage-proof.

## Step 12: the package

A package is `data/models/<id>/` (git-ignored), id `m-YYYYMMDDTHHMMSSZ-xxxxxxxx`:

| file | content |
| --- | --- |
| `model.joblib` | the fitted preprocessing + estimator `Pipeline`, one object |
| `manifest.json` | task, ordered input columns with kinds, dropped columns and reasons, classes, positive class, algorithm and parameters, seed, settings, split fingerprints, dataset provenance (ref, SHA-256, rows, columns), CV results of every candidate, the selection report and its SHA-256, library versions, format versions, the model file's SHA-256, limitations |
| `parity.json` | up to 25 **training** rows (plus one all-missing row) as text, with the predictions and probabilities made for them |
| `attempt.json` | the one-attempt marker |
| `test_result.json` | the single test evaluation |
| `seal.json` | SHA-256 of each file above and of the whole package |

`parity.json` holds a small piece of the data; keep the models folder private (it is git-ignored).

**Lifecycle (nothing weakens the one-attempt rule):**

1. *Create.* Everything is written in `.staging-<id>/`, then the package is reloaded from there and
   its predictions compared with the saved reference. Failure removes the staging folder. Success
   moves it into place with one atomic rename. State **open**: files read-only, folder writable.
2. *Attempt.* `attempt.json` is created exclusively (`open(..., "x")`) **before any test row is read**.
   A second attempt fails, including one made by another process at the same moment (tested with two
   real processes; this relies on the local filesystem honouring exclusive create). State **spent**.
3. *Result.* `test_result.json` is created exclusively and cannot be overwritten.
4. *Seal.* `seal.json` is written, every file and the folder become read-only. State **sealed**.

Only a **sealed** package loads. A crash after step 2 leaves a spent package. `package-verify`
reports `open`, `spent` and interrupted `.staging-*` folders.

**Loading** (`load_package(root, id)`): the id is resolved inside the models folder only (no paths,
no symlinks); seal hashes and the read-only check run first; then format versions and the
scikit-learn version (a mismatch is refused unless `allow_version_mismatch=True`); the model file's
SHA-256 is checked **before** `joblib.load`; then the loaded object must be a `Pipeline` whose input
columns equal the manifest's feature order. `ModelPackage.predict` / `predict_proba` is a thin
interface: extra columns are ignored, a missing column is an error that names it, and a malformed
number is rejected exactly as in training.

**Trust.** Model files are pickle-based joblib files. **`joblib.load` can execute arbitrary code, so
never load a model file that did not come from this platform's own store.** The hashes catch
accidental corruption and wrong files. They do **not** protect against someone who can rewrite the
whole folder, because they could rewrite the hashes too.

**Reproducibility.** The same data, settings and seed give the same selection hash and the same
predictions; this is checked on this machine and these library versions only. The model file is not
claimed to be byte-identical between runs.

## Measurements

See the Chunk 2B entry in `DECISIONS.md` (D-021) for the measured time, memory and package sizes.
Logging a finalized job to MLflow (`finalize --track`, `track-package`) is described in `tracking.md`.
