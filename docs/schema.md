# Feature schema

What a model's input columns looked like in the training data, and a check of other data against it
(BUILD_PLAN Step 4, chunk 4B). Code: `src/automl_ops/schema.py`; command line:
`scripts/datasets.py schema`. **Computed on demand and not stored.** Dependency: Pandera.

```bash
uv run python scripts/datasets.py schema adult-train-pool@v1 --target income
uv run python scripts/datasets.py schema adult-train-pool@v1 --target income --check adult-monitor-shifted@v1
uv run python scripts/datasets.py schema NAME@v1 --target COL --exclude id --json
```

Exit codes: 0 no structural failures (observations never fail), 1 structural failures or an unusable
task configuration, 2 bad input.

## What it is

A `SchemaSpec` is plain JSON-friendly data (`to_dict` / `from_dict`, `schema_format` 1) built from the
Step 4A profile and the Step 5 `TaskConfig`. It covers **feature columns only** (not the target, not
excluded columns) and records, per column: position, profile kind, and what was **observed** in the
training data: numeric minimum and maximum, the complete category set (for `categorical` columns), and
whether blanks occurred. The spec is the source of truth; Pandera is used to check it.

## Two different things are reported

**Structural failures** (severity error; they make `ok` false):

| Code | When |
| --- | --- |
| `missing_required_column` | a feature column is absent (the target and excluded columns are not required) |
| `duplicate_column_names` | the header repeats a column name |
| `malformed_numeric` | in an `integer` or `float` column, a non-blank cell is not a clean finite number (`abc`, `1_000`, `nan`, `inf`, `1e999`, ` 5`, `1,5`); same strict syntax as the profile |

**Observations about the training data** (severity info; they **never** change `ok`):

| Code | Meaning |
| --- | --- |
| `out_of_observed_range` | numbers below the training minimum or above the maximum (inclusive bounds are fine) |
| `unseen_categories` | values not in the training category set |
| `new_blanks` | blanks in a column that had none in training |
| `non_integer_values` | numbers not written as whole numbers in a column that held only whole numbers |
| `extra_columns` | columns that are not part of the schema (the target and excluded columns are not "extra") |

These describe how the data differs from **one training dataset**. They are valid inputs that may or
may not matter, **not** errors, and **not drift detection**: whether a difference is meaningful is a
monitoring question (Step 24). A general CSV platform has no domain knowledge, so ranges and category
sets are only what the training data contained. **User-declared domain rules** (for example "age
cannot be negative") are a deferred feature; until then nothing makes a *value* a hard failure except
being a malformed number.

## How data is checked

- **As text, as read** (every cell text; only an empty cell is missing). Nothing is converted, and
  the data and stored versions are never changed. Numbers are parsed only to test them.
- Observed limits are rounded to 12 significant digits like the profile; data values are rounded the
  same way before comparing, so a value equal to the training maximum is never flagged.
- A category set is recorded only when it is complete (the profile for the schema keeps all
  categories; `SCHEMA_PROFILE_SETTINGS`). High-cardinality `text`, `code` and `mixed` columns have no
  category set or range.
- Raw CSV bytes (`check_csv_bytes`) must have the same number of fields in every non-blank row as the
  header, otherwise a `SchemaError` is raised (pandas would otherwise reshape such a file silently).
- Pandera is imported only when a check runs (about 1.5 s to import), so other commands do not pay.

## On the demo data (checked, Step 4B)

Each training pool passes its own schema with no failures and **no** observations. The monitoring
streams and the shifted copies pass structurally. The shifted copies report out-of-range numbers in
the shifted columns, for example Adult `age` (91 of 14,652 values above the pool maximum of 90) and
`hours-per-week` (60 above 99), and Abalone `Whole_weight`/`Shell_weight`; no unseen categories, since
the category shift reuses existing categories. The shift is mostly **inside** the training range, so a
range check sees only its tail: this is why observations are not a drift test.

## Limits

- Provisional and heuristic: kinds come from the Step 4A profile.
- Observations compare against one dataset's extremes; a range check misses shifts that stay inside
  the range.
- Serving (Step 21) and monitoring (Step 24) will use this spec; how a stored spec travels with a
  model is decided with the model package (Step 12).
