# Demo data: sources, licences and how the files are prepared

**All data here is demo data for a portfolio project.** It is not for real decisions. No fairness
claim is made about the Adult data (it contains sex and race columns and 1994 census income labels).
The shifted monitoring files are **synthetic**: they say nothing about real-world data drift.

Sources and licence wording below were read from the official UCI dataset pages on
**2026-11-13**. The licence reading is mine; it is not legal advice.

## Sources

| | Adult (Census Income) | Abalone |
| --- | --- | --- |
| Task used for | Classification (`income`: `<=50K` / `>50K`) | Regression (`Rings`) |
| Official page | https://archive.ics.uci.edu/dataset/2/adult | https://archive.ics.uci.edu/dataset/1/abalone |
| Download used | https://archive.ics.uci.edu/static/public/2/adult.zip | https://archive.ics.uci.edu/static/public/1/abalone.zip |
| DOI | 10.24432/C5XW20 | 10.24432/C55C7W |
| Creators | Barry Becker, Ronny Kohavi (1996) | Warwick Nash, Tracy Sellers, Simon Talbot, Andrew Cawthorn, Wes Ford (1994) |
| Rows per UCI | 48,842 (train 32,561 + test 16,281) | 4,177 |
| Licence on the page | CC BY 4.0 | CC BY 4.0 |
| Downloaded file | 620,237 bytes | 55,357 bytes |
| SHA-256 (computed locally) | `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` | `755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4` |

UCI publishes no checksum, so the hashes above were not compared with an official value; they only
let a later download be compared with this one.

**Licence text as shown on both pages:** "This dataset is licensed under a Creative Commons
Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the
datasets for any purpose, provided that the appropriate credit is given." Licence link:
https://creativecommons.org/licenses/by/4.0/legalcode

**Attribution (citations as given on the pages):**
- Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5XW20.
- Nash, W., Sellers, T., Talbot, S., Cawthorn, A., & Ford, W. (1994). Abalone [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C55C7W.

The pages say the datasets are from the UCI Machine Learning Repository. The prepared files are
adapted (merged, split, cleaned, and partly synthetically shifted) as described below.

## What was checked against the proposal

The URLs and CC BY 4.0 licence matched what was proposed before downloading. The dataset pages also
state, and the files confirmed, the following. Adult's page says "Has Missing Values? Yes" (columns
`workclass`, `occupation`, `native-country`); Abalone's says "No". Abalone's page says continuous
values were scaled for use with an ANN (divided by 200) and rows with missing values were removed
from the original data.

## Original file formats (checked before merging Adult)

| | `adult.data` | `adult.test` |
| --- | --- | --- |
| Rows | 32,561 | 16,281 |
| First line | data | `|1x3 Cross validator` (a comment line, skipped) |
| Label values | ` <=50K`, ` >50K` | ` <=50K.`, ` >50K.` (**trailing period**) |
| Missing marker | `?` | `?` |

The trailing period is removed so both files use `<=50K` / `>50K`. Only those two values are
accepted; anything else stops the script.

## Preparation (`scripts/prepare_demo_data.py`, `src/automl_ops/demo_data.py`)

1. Download the two zip files (HTTPS, host `archive.ics.uci.edu` only, no redirects, size cap), only
   if they are not already in `data/raw/`.
2. Clean lightly: strip whitespace; turn `?` into a real empty value (Adult only has `?`); fix the
   Adult label format; merge Adult's two files. No other value is changed. Column names are the
   UCI names (Abalone's spaces become underscores).
3. **Count exact duplicate rows first, then decide.** Adult has **52 extra copies** (101 rows in 49
   groups, largest group 3); Abalone has none. Decision: **nothing is removed.** Duplicates are 0.1%
   of Adult, and deleting them would silently change the dataset. Instead each exact-duplicate group
   is kept together on one side of the split, so no content-identical row can appear in both the
   pool and the stream. (Duplicates that fall inside the training pool can still appear in different
   cross-validation folds; this effect is tiny and is not corrected.) Rows with the same features but
   different labels were not examined.
4. **Partition once, with a seed (42):** about 70% to the *training pool* and 30% to the *monitoring
   stream*. Adult is split per label so both sides keep the same class balance; Abalone is a plain
   random split.
5. **Shifted copy:** derived **only from the monitoring-stream rows**, never from the pool. Feature
   columns only; the target column and row order are unchanged. Adult: `age` + 14 and
   `hours-per-week` + 12 (one standard deviation of the stream, rounded to whole numbers), and 25% of
   the eligible rows get `occupation = Exec-managerial` (2,985 rows). Abalone: `Whole_weight` +
   0.490056 and `Shell_weight` + 0.139391 (one stream standard deviation) and 30% of the eligible
   rows get `Sex = I` (250 rows). The shift is invented to test monitoring; it is not meant to be
   realistic.
6. Write the files and `data/demo/manifest.json` (seed, counts, hashes, shift parameters, source and
   licence details, access date).

## The three roles (kept separate)

| File | Rows (Adult / Abalone) | Role |
| --- | --- | --- |
| `data/demo/<name>_train_pool.csv` | 34,190 / 2,924 | The only file uploaded for training. The platform makes its own train/test split and cross-validation inside it. |
| `data/demo/<name>_monitor_stream.csv` | 14,652 / 1,253 | Traffic simulator source and feedback labels only. Never used for training, cross-validation, model selection or the final test. |
| `data/demo/<name>_monitor_shifted.csv` | 14,652 / 1,253 | Synthetic shifted copy of the stream, for the drift demo. Always labelled as simulated. |

## Files and git

`data/` is git-ignored (not committed). Recreate it with `make demo-data`. The output is
reproducible: running the script twice gave byte-identical files on this machine (same Python,
pandas 3.1.1, numpy 2.5.4); other library versions were not tested.

## Limits

- Not checked: that these licence terms suit every possible use (my reading only); Linux behaviour
  of the script; other pandas or numpy versions.
- Adult's income labels are from 1994 US census data and the dataset contains sensitive attributes.
  Whether to exclude them is a user setting in the platform (BUILD_PLAN Step 5), not decided here.
- These datasets are small and public, chosen to demonstrate lifecycle features. Model scores on
  them say nothing about real-world performance.
