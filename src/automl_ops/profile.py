"""Dataset profiling (BUILD_PLAN Step 4, chunk 4A).

A profile is a factual description of a CSV: per-column kind, missing values, distinct values,
numeric statistics, most common values, and a few warning signs. It is computed on demand from a
stored dataset version and never saved or changed.

**Raw values are preserved.** The CSV is read with every cell as text (`dtype=str`), and only a
completely empty cell counts as missing. Text such as `NA`, `N/A`, `null` or `007` stays exactly as
uploaded; the profile only *reports* these. The column `kind` decided here (integer, float,
categorical, text, mixed, code, empty) is the planned shared reference for later typed loading
(Step 7 onward); Step 4 itself never converts any value.

Settings and thresholds are provisional defaults (see docs/profiling.md).
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from automl_ops.dataset_store import DatasetStore, DatasetStoreError, DatasetVersion
from automl_ops.ingest import sha256_bytes

PROFILE_FORMAT = 1

Kind = Literal["integer", "float", "categorical", "text", "mixed", "code", "empty"]

# Strict, ASCII-only number syntax. Python's float() also accepts things like "1_000", "nan" and
# "inf", which a data file should not silently turn into numbers.
NUMBER_RE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
INTEGER_RE = re.compile(r"[+-]?[0-9]+")
LEADING_ZERO_RE = re.compile(r"[+-]?0[0-9]+")  # "007", "00", "-012"; not "0", "0.5" or "7"

# Text that often means "missing" but is kept as a value (only empty cells are missing).
NA_LIKE_TOKENS = frozenset({"na", "n/a", "nan", "null", "none", "nil", "?", "-", "--"})

EXAMPLE_LIMIT = 5


@dataclass(frozen=True)
class ProfileSettings:
    """Settings that decide a column's kind (provisional defaults; see docs/profiling.md)."""

    max_categories: int = 50  # more distinct text values than this -> kind "text"
    mixed_min_numeric_share: float = 0.5  # at least this share numeric (but not all) -> "mixed"
    top_n: int = 20  # most common values kept per column

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_categories": self.max_categories,
            "mixed_min_numeric_share": self.mixed_min_numeric_share,
            "top_n": self.top_n,
        }


DEFAULT_SETTINGS = ProfileSettings()


@dataclass(frozen=True)
class NumericStats:
    minimum: float
    maximum: float
    mean: float
    std: float | None  # sample standard deviation (n - 1); None when there is only one value
    median: float
    p25: float
    p75: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "p25": self.p25,
            "p75": self.p75,
        }


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    position: int  # 1-based
    kind: Kind
    count: int  # non-missing cells
    missing: int
    missing_pct: float
    unique: int  # distinct non-missing text values
    top_values: tuple[tuple[str, int], ...]  # (value, count), most common first
    numeric: NumericStats | None  # only for kinds integer and float
    numeric_share: float  # share of non-missing cells that look like numbers
    non_numeric_examples: tuple[tuple[str, int], ...]  # for kind mixed
    na_like_count: int  # cells holding NA-like text (all of them, not just the examples)
    na_like_tokens: tuple[tuple[str, int], ...]  # most common NA-like spellings (kept as values)
    leading_zero_count: int
    leading_zero_examples: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "kind": self.kind,
            "count": self.count,
            "missing": self.missing,
            "missing_pct": self.missing_pct,
            "unique": self.unique,
            "top_values": [{"value": v, "count": c} for v, c in self.top_values],
            "numeric": self.numeric.to_dict() if self.numeric else None,
            "numeric_share": self.numeric_share,
            "non_numeric_examples": [
                {"value": v, "count": c} for v, c in self.non_numeric_examples
            ],
            "na_like_count": self.na_like_count,
            "na_like_tokens": [{"value": v, "count": c} for v, c in self.na_like_tokens],
            "leading_zero_count": self.leading_zero_count,
            "leading_zero_examples": [
                {"value": v, "count": c} for v, c in self.leading_zero_examples
            ],
        }


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    column_count: int
    columns: tuple[ColumnProfile, ...]
    duplicate_rows: int  # extra copies of exact duplicate rows (first copy not counted)
    rows_in_duplicate_groups: int
    rows_with_missing: int
    missing_cells: int
    settings: ProfileSettings

    def column(self, name: str) -> ColumnProfile | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_format": PROFILE_FORMAT,
            "settings": self.settings.to_dict(),
            "rows": self.rows,
            "columns": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "rows_in_duplicate_groups": self.rows_in_duplicate_groups,
            "rows_with_missing": self.rows_with_missing,
            "missing_cells": self.missing_cells,
            "column_profiles": [c.to_dict() for c in self.columns],
        }


@dataclass(frozen=True)
class VersionProfile:
    """A profile together with the exact dataset version it describes."""

    version: DatasetVersion
    profile: DatasetProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_ref": self.version.ref,
            "sha256": self.version.sha256,
            **self.profile.to_dict(),
        }


# --------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------


def read_text_frame(data: bytes) -> pd.DataFrame:
    """Read canonical CSV bytes with every cell as text. Only an empty cell becomes missing."""
    return pd.read_csv(
        io.BytesIO(data),
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------------------------


def significant(value: float, digits: int = 12) -> float:
    """Round to `digits` significant digits so results do not depend on last-bit noise."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    return float(f"{value:.{digits}g}")


def _is_finite_number(text: str) -> bool:
    if not NUMBER_RE.fullmatch(text):
        return False
    return math.isfinite(float(text))


def _ranked(items: dict[str, int], limit: int) -> tuple[tuple[str, int], ...]:
    ordered = sorted(items.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ordered[:limit])


def _numeric_stats(values: list[str], counts: list[int]) -> NumericStats:
    array = np.repeat(np.array([float(v) for v in values], dtype="float64"), counts)
    std = float(np.std(array, ddof=1)) if array.size > 1 else None
    return NumericStats(
        minimum=significant(float(array.min())),
        maximum=significant(float(array.max())),
        mean=significant(float(array.mean())),
        std=None if std is None else significant(std),
        median=significant(float(np.percentile(array, 50))),
        p25=significant(float(np.percentile(array, 25))),
        p75=significant(float(np.percentile(array, 75))),
    )


def profile_column(
    series: pd.Series, name: str, position: int, rows: int, settings: ProfileSettings
) -> ColumnProfile:
    counted = series.value_counts(dropna=True, sort=False)
    values = [str(v) for v in counted.index]
    counts = [int(c) for c in counted.to_numpy()]
    count = sum(counts)
    missing = rows - count
    freq = dict(zip(values, counts, strict=True))

    numeric_flags = [_is_finite_number(v) for v in values]
    numeric_count = sum(c for c, flag in zip(counts, numeric_flags, strict=True) if flag)
    integer_only = all(
        INTEGER_RE.fullmatch(v) for v, flag in zip(values, numeric_flags, strict=True) if flag
    )
    leading_zero = {v: c for v, c in freq.items() if LEADING_ZERO_RE.fullmatch(v)}
    na_like = {v: c for v, c in freq.items() if v.strip().lower() in NA_LIKE_TOKENS}
    non_numeric = {
        v: c for v, c, flag in zip(values, counts, numeric_flags, strict=True) if not flag
    }
    share = numeric_count / count if count else 0.0

    kind: Kind
    if count == 0:
        kind = "empty"
    elif numeric_count == count:
        kind = "code" if leading_zero else ("integer" if integer_only else "float")
    elif share >= settings.mixed_min_numeric_share:
        kind = "mixed"
    else:
        kind = "categorical" if len(values) <= settings.max_categories else "text"

    stats = _numeric_stats(values, counts) if kind in ("integer", "float") else None
    return ColumnProfile(
        name=name,
        position=position,
        kind=kind,
        count=count,
        missing=missing,
        missing_pct=significant(100.0 * missing / rows) if rows else 0.0,
        unique=len(values),
        top_values=_ranked(freq, settings.top_n),
        numeric=stats,
        numeric_share=significant(share),
        non_numeric_examples=_ranked(non_numeric, EXAMPLE_LIMIT) if kind == "mixed" else (),
        na_like_count=sum(na_like.values()),
        na_like_tokens=_ranked(na_like, EXAMPLE_LIMIT),
        leading_zero_count=sum(leading_zero.values()),
        leading_zero_examples=_ranked(leading_zero, EXAMPLE_LIMIT),
    )


def profile_frame(
    frame: pd.DataFrame, settings: ProfileSettings = DEFAULT_SETTINGS
) -> DatasetProfile:
    """Profile a frame of text cells (as returned by `read_text_frame`)."""
    rows = len(frame)
    columns = tuple(
        profile_column(frame[name], str(name), position, rows, settings)
        for position, name in enumerate(frame.columns, start=1)
    )
    missing_mask = frame.isna()
    duplicated_any = frame.duplicated(keep=False)
    return DatasetProfile(
        rows=rows,
        column_count=len(frame.columns),
        columns=columns,
        duplicate_rows=int(frame.duplicated(keep="first").sum()),
        rows_in_duplicate_groups=int(duplicated_any.sum()),
        rows_with_missing=int(missing_mask.any(axis=1).sum()),
        missing_cells=int(missing_mask.to_numpy().sum()),
        settings=settings,
    )


def profile_csv_bytes(data: bytes, settings: ProfileSettings = DEFAULT_SETTINGS) -> DatasetProfile:
    return profile_frame(read_text_frame(data), settings)


def read_version_bytes(store: DatasetStore, ref: str) -> tuple[DatasetVersion, bytes]:
    """Read a stored version's file, refusing it if it no longer matches its record."""
    version = store.get_version(ref)
    data = store.version_path(ref).read_bytes()
    if sha256_bytes(data) != version.sha256:
        raise DatasetStoreError(
            f"{ref} does not match its recorded SHA-256 (the stored file was changed); "
            "run `verify` and do not use this version"
        )
    return version, data


def read_version_frame(store: DatasetStore, ref: str) -> tuple[DatasetVersion, pd.DataFrame]:
    """Read a stored version as a text frame (all cells text), after the SHA-256 check."""
    version, data = read_version_bytes(store, ref)
    return version, read_text_frame(data)


def profile_version(
    store: DatasetStore, ref: str, settings: ProfileSettings = DEFAULT_SETTINGS
) -> VersionProfile:
    version, data = read_version_bytes(store, ref)
    return VersionProfile(version=version, profile=profile_csv_bytes(data, settings))
