# CSV ingestion and dataset versions

How the platform accepts a CSV and stores it as an immutable dataset version (BUILD_PLAN Step 3).
Code: `src/automl_ops/ingest.py` (checks and conversion), `src/automl_ops/dataset_store.py`
(storage), `scripts/datasets.py` (command-line tool). All of this uses the Python standard library
only. The limits and format rules below are **provisional defaults**.

## Using it

```bash
uv run python scripts/datasets.py ingest adult data/demo/adult_train_pool.csv   # new dataset -> adult@v1
uv run python scripts/datasets.py ingest adult fixed.csv --add-version          # new content -> adult@v2
uv run python scripts/datasets.py list [NAME]
uv run python scripts/datasets.py show adult@v1
uv run python scripts/datasets.py verify [NAME]
```

The default store is `data/store/` (git-ignored). `--store PATH` selects another folder.

## What is accepted

| Rule | Value (provisional) |
| --- | --- |
| Size | at most 50 MiB (52,428,800 bytes), checked before the file is read in full |
| Rows | at most 100,000 data rows (the header is not counted) |
| Columns | 2 to 100 |
| Encoding | UTF-8 (with or without BOM) or UTF-16 with a BOM. Anything else is rejected with a "re-save as UTF-8" message; other encodings are never guessed |
| Delimiter | comma, semicolon, tab or pipe, chosen by consistent field counts in the first 50 records. If two would fit, the file is rejected |
| Column names | not empty; no duplicates (compared ignoring case and surrounding spaces); otherwise kept as given |
| Rows | every record must have the same number of fields as the header |
| Blank lines | allowed only at the very end (counted); a blank line before or between records is an error |
| Other | no NUL characters; quoting must be valid; a single cell over 131,072 characters is rejected (Python's default CSV field limit) |

Cells are kept as the exact text uploaded. Nothing is trimmed, typed or converted; missing values
and column types are handled later (Step 4).

## Canonical form and identity

An accepted file is converted to **UTF-8, comma-separated, `\n` line endings, minimal quoting**.
That converted file is what is stored, and the **SHA-256 of the converted bytes is the identity of
the version**. So the same data saved with a BOM, semicolons, tabs, CRLF or UTF-16 maps to the same
version. The SHA-256 of the original upload, its encoding, delimiter and size are kept in the version
record as provenance only. **The original upload bytes are not kept.**

Limit: line endings *inside* a quoted cell are kept as written, so the same data with `\r\n` inside
a quoted cell is a different version from the `\n` form.

## Store layout

```
data/store/<dataset>/v0001.csv     the canonical file (read-only)
data/store/<dataset>/v0001.json    the version record (read-only); written last, so it marks a complete version
data/store/<dataset>/v0002.csv ...
```

Dataset names: 1 to 64 characters from `a-z`, `0-9`, `_`, `-`, starting with a letter or digit. A
version is referred to as `name@vN`. The record holds: dataset, number, sha256, original_sha256,
size_bytes, original_size_bytes, rows, columns, source_filename (base name only; never used in a
path), original_encoding, original_delimiter, reencoded, trailing_blank_lines_ignored, created_at
(UTC). These mirror the future `dataset_versions` table.

## What prevents accidents

- **Re-upload of identical content** (even in another packaging) returns the existing version and
  writes nothing.
- **Different content under an existing name** is refused unless `--add-version` is given. A new
  name must not exist yet, and `--add-version` on a name that does not exist is refused, so a
  mistyped name cannot silently create or extend a dataset.
- **Nothing is ever overwritten.** A new version number is claimed with an exclusive file create;
  interrupted writes leave files that `verify` reports and that later versions never reuse. Stored
  files are made read-only.
- **A rejected CSV writes nothing** (no empty dataset folder is left behind).
- **`verify`** re-reads every stored file and checks size, SHA-256, that it still parses and is in
  canonical form, that rows and columns match the record, and reports orphan or leftover files.
- **No implicit "latest".** Anything that uses a dataset must name an exact version.

## Limits of this design

Built for one user on one machine. No protection against several processes writing at the same time
beyond the atomic number claim. The store does not know what a file is for: a monitoring-stream file
could be ingested like any other (the separation rule is enforced elsewhere; BUILD_PLAN Step 25 adds
a hash guard). There is no delete or archive feature; removing a version is a manual file operation.
`verify` detects changes but cannot repair them, and read-only permission bits stop accidents, not
a determined user. Measured on this Mac: a 46.8 MB, 100,000-row file ingested in about 1.2 s with
a peak of about 407 MB of memory; Adult's 3.7 MB pool in under 0.2 s.

A database-backed store replaces `FileDatasetStore` in Phase 4 (Steps 17 and 18) behind the same
`DatasetStore` interface.
