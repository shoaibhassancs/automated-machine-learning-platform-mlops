# Exploration data

The numbers the preview page needs (BUILD_PLAN Step 6). Code: `src/automl_ops/explore.py`; command
line: `scripts/datasets.py explore`. Output is plain JSON-ready data; **no plotting library** is used
(the front end renders it). Computed on demand; nothing is stored or converted in the data.

```bash
uv run python scripts/datasets.py explore abalone-train-pool@v1                 # readable summary
uv run python scripts/datasets.py explore adult-train-pool@v1 --target income --json
```

## What it returns

| Key | Content |
| --- | --- |
| `sample_rows` | the first **10** rows (provisional), cells as the original text; an empty cell is `null` |
| `histograms` | for each `integer` / `float` column: `min`, `max`, `count`, `missing`, and **20** equal-width bins (provisional) as `edges` (21) and `counts` (20), from NumPy's `histogram` |
| `correlation` | Pearson correlation between numeric columns, using the rows where both values exist; `columns`, a square `matrix`, and the 10 strongest `top_pairs`. A pair with a constant column has no correlation and is `null`, never `NaN` |
| `class_balance` | only for a **classification** config: each class with `count` and `share` (up to 50 listed), `total_classes`, `largest_class_share`, `smallest_class_rows`, `largest_to_smallest_ratio`; otherwise `null` |

Columns of kind `categorical`, `text`, `code`, `mixed` or `empty` are not in the histograms or the
correlation. Numbers are rounded to 12 significant digits, so output is deterministic and strict JSON.

## Limits

- Correlation shows linear association between numeric columns only; it says nothing about
  categorical columns and is not evidence of cause.
- Bin count and sample size are provisional defaults.
- A column of whole numbers that is really a category (a code) appears as numeric unless it has
  leading zeros.
