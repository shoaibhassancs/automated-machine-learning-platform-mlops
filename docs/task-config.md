# Task configuration

What you tell the platform to predict (BUILD_PLAN Step 5). Code: `src/automl_ops/task.py`; command
line: `scripts/datasets.py task`. Everything is **computed on demand**; nothing is stored and a stored
dataset version is never modified. All thresholds below are **provisional defaults**.

## Using it

```bash
uv run python scripts/datasets.py task adult-train-pool@v1 --target income
uv run python scripts/datasets.py task abalone-train-pool@v1 --target Rings --json
uv run python scripts/datasets.py task NAME@v1 --target COL --task regression --exclude id,notes
```

Without `--task`, the **suggested** task is used. Exit codes: 0 usable (warnings and info do not
fail), 1 the configuration has errors, 2 bad input (for example no suggestion is possible and no
`--task` was given, or a stored file no longer matches its recorded SHA-256).

## A configuration

`TaskConfig(target, task, excluded)`: the column to predict, `classification` or `regression`, and
the columns left out of the model inputs. The **input (feature) columns** are every column except
the target and the excluded ones.

## The task suggestion is advisory

| Target looks like | Suggestion |
| --- | --- |
| text (kind `categorical`, `text`, `mixed` or `code`) | classification |
| decimal numbers (kind `float`) | regression |
| whole numbers, **more than 20** distinct values | regression |
| whole numbers, 20 or fewer distinct values | classification |
| no values, or no such column | none |

You can always choose the other type. A choice that differs from the suggestion is only an **info**
note (`task_differs_from_suggestion`), never an error. On the demo data (counted directly from the
files, not recalled): Adult `income` has 2 distinct values, so classification; Abalone `Rings` has
26 distinct whole numbers, so regression.

## What is an error, a warning, or information

A configuration is rejected (**error**) only when the chosen task is impossible or incompatible with
the planned seeded stratified split (Step 9):

| Code | When |
| --- | --- |
| `unknown_target`, `target_missing`, `target_constant` | target quality; the same code as Step 4A `validate` |
| `target_excluded` | the target is also listed as excluded |
| `unknown_excluded_column` | an excluded name is not a column |
| `no_feature_columns` | nothing is left once the target and excluded columns are removed |
| `target_not_numeric` | regression chosen, but not every target value is a clean number (kind `integer` or `float`) |
| `class_too_small` | classification, and some class has **fewer than 2 rows**: a stratified split cannot put it in both the training part and the test part |

**Warnings** (the configuration still works):

| Code | When (provisional) |
| --- | --- |
| `many_classes` | more than **50** classes. Allowed: numeric labels can be classes and many classes are legitimate. The warning is because every class needs enough rows for the split and cross-validation, per-class scores get noisy, and the confusion matrix chart becomes very large |
| `small_class` | smallest class has fewer than **5** rows. A placeholder until Step 9 fixes the number of cross-validation folds |

**Info:** `decimal_class_labels` (classification on a decimal-looking target: each distinct text
value becomes a class, and `1` and `1.0` are different), `few_distinct_values_for_regression` (20 or
fewer distinct values; may be labels), `task_differs_from_suggestion`.

A high class count, or a target that "looks continuous", never rejects a classification by itself.
Classes are the exact text values as read; how labels are encoded is decided at Step 7.

## Not here

Task *suitability* beyond the above (for example whether a model would perform well) is not judged.
Step 4A `validate` still reports data problems that are independent of the task.

## Limits

- Thresholds (20, 50, 5, 2) are provisional; the "2" follows from the planned split.
- The minimum class size actually needed also depends on the test fraction and fold count chosen at
  Step 9; the checks here are a floor, not a guarantee that training will be sensible.
