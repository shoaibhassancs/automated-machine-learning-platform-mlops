# Training and cross-validation

Preprocessing, the algorithm catalogue, metrics, and the cross-validation runner (BUILD_PLAN Steps 7,
8, 10 and 9; Chunk 2A). Code: `src/automl_ops/pipeline.py`, `algorithms.py`, `evaluate.py`,
`train.py`; command line: `scripts/datasets.py train`. Everything is computed on demand and a stored
dataset version is never modified. All defaults and bounds below are **provisional**.

```bash
uv run python scripts/datasets.py train adult-train-pool@v1 --target income
uv run python scripts/datasets.py train abalone-train-pool@v1 --target Rings --json
uv run python scripts/datasets.py train NAME@v1 --target COL --algorithms ridge,random_forest --folds 3
```

Exit codes: 0 every algorithm finished, 1 an algorithm failed, 2 bad input or a run that cannot be
carried out (the message says what to change). Progress lines go to stderr; results go to stdout (a
table, or `--json`). The same structured result (`TrainingResult.to_dict()`) is what a future web
interface will read.

**`train` never reads the held-out test part and computes no test-set metric.** Selecting a model
and scoring it once on the test part is `finalize` (Step 11, Chunk 2B): see
`docs/selection-and-packaging.md`.

## Split and cross-validation

- **Holdout:** 20% of the rows are reserved as the test part, by row index only. Seed 42.
  Classification is **stratified**; regression is shuffled.
- **Cross-validation:** 5 folds on the training part (stratified for classification, shuffled for
  regression), same seed.
- **Isolation:** after the split, only `split.train_rows` is indexed. No test row, label or statistic
  reaches preprocessing, fitting, fold creation or scoring. Two caveats, stated plainly: the
  stratified split itself uses all labels to place rows (any stratified split does), and the
  pre-flight task checks (Step 5 rules, run again inside the runner) look at the whole dataset
  because the feasibility of the split depends on it. A test shows that rewriting the test rows'
  labels and features (even with malformed numbers) for a fixed split changes no result.
- **Small classes (classification):**

| Situation | Outcome |
| --- | --- |
| a class with fewer than 2 rows in total | error: a stratified split is impossible |
| more classes than rows in the test part | error: the test part cannot hold every class |
| after the holdout, a class has fewer than 2 training rows | error: cross-validation needs 2 |
| after the holdout, the smallest training class has 2 to 4 rows | folds are **reduced to that count**, with a warning |
| smallest training class has 5 or more rows | 5 folds, no warning |

  For example 50 + 50 + 3 rows gives 2 training rows in the small class and 2 folds; 50 + 50 + 2 stops.
  Whether a 2-row class lands one row in the test part depends on the data sizes, so it is reported
  rather than assumed. The Step 5 rules (error under 2 rows, warning under 5) are unchanged.
- **Regression:** needs at least as many training rows as folds. Settings bounds: test fraction 0.05
  to 0.5, folds 2 to 10.

## Preprocessing (Step 7)

One builder (`build_pipeline`) serves training and later inference, so they cannot differ. The
pipeline takes the data **as text** (what `read_text_frame` produces) and converts inside it.

| Column kind (from the profile of the training part) | Treatment |
| --- | --- |
| `integer`, `float` | strict number parsing, median imputation, standard scaling for linear models only |
| `categorical` | most-frequent imputation, then one-hot encoding that ignores unseen categories (unseen = all zeros) |
| `text`, `code`, `mixed`, `empty` | **not used as inputs**; each is listed with its reason in the result (`columns.dropped`), never dropped silently |

- **No leakage:** medians, scaling and category lists are learned by `fit`, inside a scikit-learn
  `Pipeline`. Every fold clones the unfitted pipeline and fits it on **that fold's training rows
  only**; nothing is fitted on the whole dataset first. Tests compare each fold's learned median,
  mean, scale and category list with values computed from that fold's training rows, and show a
  global fit would have given different numbers.
- **Missing and unseen:** an empty cell (missing) is imputed; a category not seen in fitting scores
  without error.
- **Malformed numbers are not hidden:** a non-empty value in a numeric column that is not a clean
  finite number (`abc`, `1_000`, `nan`) raises `MalformedValueError`; it is never turned into a
  missing value. Data should pass the Step 4B schema check first.
- In a frame, only NaN or `None` is missing; an empty string is a value. Reading a CSV already turns
  an empty cell into missing; a future API layer must do the same for request fields.
- NA-like text (`NA`, `?`) stays an ordinary category, as everywhere else on the platform.
- Dense one-hot output is used (gradient boosting needs it). Memory grows with the number of
  categories times rows; the largest case allowed by the ingestion limits was **not measured**.

## Algorithms (Step 8)

A closed registry. Unknown parameter names, wrong types (a bool is not a number), non-finite numbers
and out-of-range values are rejected, all problems reported together; nothing executable and no other
estimator can be selected. Bounds are inclusive.

| Task | Key | Parameters (default; bounds) |
| --- | --- | --- |
| classification | `logistic_regression` | `C` (1.0; 1e-4 to 1e4), `max_iter` (1000; 50 to 5000), `class_weight` (`none`; `none`, `balanced`) |
| regression | `ridge` | `alpha` (1.0; 1e-4 to 1e4) |
| both | `random_forest` | `n_estimators` (100; 10 to 500), `max_depth` (null; 1 to 50), `min_samples_leaf` (1; 1 to 50), `max_features` (`sqrt` for classification, `all` for regression; `sqrt`, `log2`, `all`) |
| both | `hist_gradient_boosting` | `max_iter` (100; 10 to 500), `learning_rate` (0.1; 0.01 to 1.0), `max_depth` (null; 1 to 30), `min_samples_leaf` (20; 1 to 200), `l2_regularization` (0.0; 0 to 100) |

Fixed, not editable: logistic regression uses the L2 penalty and the `lbfgs` solver; the random forest
runs on one thread; gradient boosting has **early stopping off**; the seed is the run's seed.

**Threads.** Algorithms run one at a time. Native thread pools are capped at 2 by default
(`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, set in `automl_ops/__init__.py`
before scikit-learn loads; a value already in the environment wins). Measured: gradient boosting on
the Adult pool took the same 8.4 s with the cap and 8.7 s with 8 threads, but used 13 s of CPU time
against 39 s.

## Metrics (Step 10)

All in `evaluate.py`; plain numbers, no NaN or infinity (an undefined metric is `null`, with the
reason under `undefined`).

- **Class order:** one sorted list of class labels is used everywhere (confusion matrix rows and
  columns, per-class table, probability columns). For two classes the **positive class is the
  second**, for example `>50K`.
- **Classification:** accuracy; macro precision, recall and F1 (averaged over the classes that occur
  in the rows being scored, 0 where a class is never predicted); ROC-AUC (the binary area, or the
  **one-vs-rest macro average** for more classes; undefined if a class does not occur in the rows);
  confusion matrix (its total equals the rows scored); ROC and precision-recall curve points from
  predicted probabilities (one curve for binary; the 10 largest classes one-vs-rest otherwise; each
  thinned to at most 200 points; a curve needs both positive and negative rows or it is skipped and
  noted).
- **Regression:** RMSE, MAE, R² (undefined when the actual values are all equal), and residual points
  `actual - predicted`, each with the row number of the observation in the original data (evenly
  thinned to at most 1,000 points, with a note).
- **What is compared:** for each algorithm, the per-fold metrics (mean and sample standard deviation)
  and the **pooled out-of-fold evaluation** over the whole training part, where every training row
  is predicted exactly once by a model that did not see it. No test-set predictions are included.

## The result

`TrainingResult.to_dict()` holds: settings, split sizes, folds requested and used, warnings, the
column plan (used and dropped with reasons), class labels and training class counts, and per
algorithm its parameters, status, error (`type` and a message of at most 500 characters), seconds
(total and per fold), per-fold metrics, summary, and the pooled out-of-fold evaluation. A failing
algorithm is reported and the others still finish. Progress is reported as events
(`algorithm_started`, `fold_finished`, `algorithm_finished`, `algorithm_failed`). Out-of-fold
predictions and probabilities are also kept on the Python objects for later steps, not in the JSON.

## Measured (one run each on this Mac, all default algorithms, sequentially)

| Dataset | Rows (training part) | Wall time | Peak memory | Cross-validation results (mean over 5 folds) |
| --- | --- | --- | --- | --- |
| Abalone pool, `Rings`, regression | 2,924 (2,339) | 4.1 s | about 193 MB | RMSE: ridge 2.238, random forest 2.177, boosting 2.207; R² 0.52, 0.55, 0.54 |
| Adult pool, `income`, classification | 34,190 (27,352) | 18.0 s | about 401 MB | accuracy 0.851, 0.856, 0.873; ROC-AUC 0.906, 0.903, 0.927 (logistic, forest, boosting) |

These are cross-validation scores on the training part, not test scores.

## Limits

- Provisional defaults and bounds; one run per measurement. The 100,000 x 100 ingestion limit was
  not run.
- Models are not tuned; a result says how the defaults did on these folds.
- The pipeline treats a numeric column as numbers only when the whole column is clean numbers (the
  profile's `integer`/`float`); an identifier stored as a number is still used as an input.
- Gradient boosting reproduced bit-for-bit across repeated runs here; this was checked on this
  machine and library versions only.
