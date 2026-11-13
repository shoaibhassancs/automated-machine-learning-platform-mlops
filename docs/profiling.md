# Profiling and validation

What the platform can tell you about a stored dataset version before any model is trained
(BUILD_PLAN Step 4, chunk 4A). Code: `src/automl_ops/profile.py` and `src/automl_ops/validate.py`;
command line: `scripts/datasets.py`. Reports are **computed on demand** and never saved; a stored
dataset version is never modified. All settings and thresholds below are **provisional defaults**.

## Using it

```bash
uv run python scripts/datasets.py profile adult-train-pool@v1            # readable table
uv run python scripts/datasets.py profile adult-train-pool@v1 --json     # for the future API/UI
uv run python scripts/datasets.py validate adult-train-pool@v1 --target income
uv run python scripts/datasets.py validate abalone-train-pool@v1 --target Rings --json
```

`validate` exits 0 when there are no errors (warnings and info do not fail), 1 when there are errors,
and 2 for bad input (unknown version, or a stored file that no longer matches its recorded SHA-256;
the profile refuses such a file). The `--json` output is deterministic and strictly valid JSON (no
`NaN`), so it is the format a later API and web page can serve directly.

## How values are read (important)

- Every cell is read **as text**; nothing is converted. `007` stays `007`, `NA` stays `NA`.
- **Only an empty cell is missing.** Text that often means "missing" (`NA`, `N/A`, `null`, `none`,
  `nil`, `nan`, `?`, `-`, `--`, compared ignoring case and surrounding spaces) is kept as an
  ordinary value and **reported as a warning**. This avoids the common trap where a library silently
  turns a real value (such as the country code `NA`) into a missing value.
- Numbers must match a strict syntax: optional sign, digits, optional decimal part, optional
  exponent (`12`, `-3.5`, `.5`, `1e5`). Things Python itself would accept but a data file should not
  quietly become numbers (`1_000`, `nan`, `inf`, `1e999`, ` 5`, `1,5`, non-ASCII digits) are treated as
  non-numbers.
- A whole-number value with a **leading zero** (`007`, `00`, `-012`, but not `0`, `0.5` or `7`) may be a
  code or identifier, not a number. Its text is preserved and the column is flagged.

## Column kinds

The `kind` assigned here is the planned shared reference for later typed loading (Step 7 onward);
Step 4 itself converts nothing.

| Kind | Rule |
| --- | --- |
| `empty` | no non-missing values |
| `code` | every value looks numeric but at least one has a leading zero: ambiguous, kept as text, no statistics |
| `integer` | every non-missing value is a whole number |
| `float` | every non-missing value is a number, some with a decimal part or exponent |
| `mixed` | at least 50% look numeric (`mixed_min_numeric_share`) but not all |
| `categorical` | otherwise, at most 50 distinct values (`max_categories`) |
| `text` | otherwise, more than 50 distinct values |

Per column the profile gives: count, missing count and percent, distinct values, the 20 most common
values with counts (`top_n`; ties broken alphabetically), the share that looks numeric, up to 5
non-numeric examples (mixed), up to 5 NA-like spellings and the total NA-like count, the leading-zero
count and examples, and, for `integer` and `float`, min, max, mean, sample standard deviation
(n - 1), median and quartiles (linear interpolation, like NumPy's default). Dataset level: rows,
columns, extra copies of exact duplicate rows, rows with any missing value, missing cells.

## Findings (validation rules)

Severity: **error** = training on this data (or with this target) is invalid whatever is excluded;
**warning** = review it, or exclude the column; **info** = good to know. (Which warnings an exclusion
resolves is decided in Step 5.)

| Code | Severity | When (provisional thresholds) |
| --- | --- | --- |
| `too_few_rows` | error | fewer than 30 rows |
| `few_rows` | warning | 30 to 99 rows |
| `duplicate_rows` | info | any exact duplicate rows (duplicates are kept) |
| `all_missing` | warning | every value of a column is missing |
| `high_missing` | warning | more than 40% missing (not all) |
| `constant_column` | warning | exactly one distinct non-missing value |
| `mixed_types` | warning | kind is `mixed`; examples listed |
| `na_like_values` | warning | NA-like text present; kept as values, counted |
| `leading_zero_values` | warning | values with a leading zero; text preserved, column not treated as numeric |
| `id_like_integer` | warning | whole-number column where every value is different (50+ rows) |
| `id_like_text` | warning | text/code/categorical column with at least 90% distinct values (50+ rows) |
| `many_categories` | warning | `text` kind (over 50 distinct values) that is not ID-like |
| `unknown_target` | error | with `--target`: the column does not exist |
| `target_missing` | error | with `--target`: the target has any missing value |
| `target_constant` | error | with `--target`: the target has a single value |

Findings are ordered errors first, then dataset-level, then by column position.

**Not checked here (Step 5):** the type of prediction task, class sizes and other task suitability
(for example a class with a single row); see `docs/task-config.md`. With a target, Step 4 checks target
*quality* only. Schema checks of new data against a training dataset are in `docs/schema.md`.

**Optional extras left out of V1:** near-constant columns, mostly-zero columns, inconsistent
categories ("Male"/"male"), numeric columns with very few distinct values, whitespace-only cells.

## On the demo data (checked)

Adult and Abalone, all six prepared files: no errors and no warnings; the only finding is the
informational `duplicate_rows` on Adult (30 extra copies in the pool, 22 in the stream, 16 in the shifted
copy). Adult: six `integer` columns (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`,
`hours-per-week`), eight `categorical` columns including the two-class target `income`; missing values only
in `workclass`, `occupation` and `native-country`. Abalone: seven `float` columns, `Sex` categorical, and the
whole-number target `Rings`.

## Limits

- Kind inference and thresholds are heuristics and provisional.
- Dates and times are not recognised in V1 (they appear as `categorical`/`text`).
- A numeric column that holds identifiers is only caught when every value is distinct or a leading zero
  appears.
- Whole numbers beyond about 15 digits lose precision in the statistics (the text is unaffected).
- Statistics are rounded to 12 significant digits so results do not depend on last-digit noise.
- Measured on this Mac (one run each): Adult's 3.7 MB pool in about 0.4 s (about 106 MB peak memory); a
  100,000-row by 12-column 46.8 MB file in about 2.6 s (about 330 MB); a 100,000-row by 100-column 47 MB
  file in about 4.1 s (about 767 MB). These are the largest shapes the ingestion limits allow.
