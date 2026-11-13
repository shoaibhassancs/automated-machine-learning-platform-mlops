"""Feature schema (BUILD_PLAN Step 4, chunk 4B).

A `SchemaSpec` records what the model's **input (feature) columns** looked like in a training
dataset: which columns exist and their profile kind, plus what was *observed* there (numeric range,
category set, whether blanks occurred). It is built from the Step 4A profile and the Step 5
`TaskConfig`, is plain JSON-friendly data (`to_dict` / `from_dict`) and is **never stored by this
module**: results are computed on demand.

`check_frame` / `check_csv_bytes` compare new data with a spec and report two separate things:

* **Structural failures** (`errors`, make `ok` false): a required feature column is missing, a
  column name is duplicated in the header, or a value in a numeric-kind column is not a clean finite
  number (`abc`, `1_000`, `nan`, `1e999`). These are checked with Pandera.
* **Observations** (`info`, never change `ok`): values outside the numeric range seen in training,
  categories not seen in training, blanks in a column that had none, non-whole values in a
  whole-number column, and extra columns. They are valid inputs that may or may not matter; they
  are **not** drift detection (that is Step 24).

No domain rules are built in: a general CSV platform cannot know that an age may not be negative.
Ranges and category sets here are only what one training dataset happened to contain. User-declared
domain rules are a deferred feature.

Data is checked **as text, as read** (`read_text_frame`); nothing is converted or rewritten. Numbers
are parsed only to test them. Pandera is imported lazily inside `check_frame`, so commands that do
not use it do not pay its import time.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from automl_ops.profile import (
    DEFAULT_SETTINGS,
    INTEGER_RE,
    NUMBER_RE,
    DatasetProfile,
    Kind,
    ProfileSettings,
    profile_frame,
    read_text_frame,
    significant,
)
from automl_ops.task import TaskConfig
from automl_ops.validate import Finding

SCHEMA_FORMAT = 1
NUMERIC_KINDS: tuple[str, ...] = ("integer", "float")
EXAMPLE_LIMIT = 5

# Category sets need every distinct value, so profile with top_n = max_categories (50 by default).
SCHEMA_PROFILE_SETTINGS = ProfileSettings(top_n=DEFAULT_SETTINGS.max_categories)


class SchemaError(ValueError):
    """The schema could not be built, loaded or applied (as opposed to data that fails a check)."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    position: int  # 1-based position in the training dataset
    kind: Kind
    observed_min: float | None  # integer/float kinds only
    observed_max: float | None
    observed_categories: tuple[str, ...] | None  # categorical kind only (the complete set)
    had_missing: bool  # the training data had at least one blank in this column

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "kind": self.kind,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "observed_categories": (
                None if self.observed_categories is None else list(self.observed_categories)
            ),
            "had_missing": self.had_missing,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> ColumnSpec:
        categories = raw["observed_categories"]
        return ColumnSpec(
            name=str(raw["name"]),
            position=int(raw["position"]),
            kind=raw["kind"],
            observed_min=None if raw["observed_min"] is None else float(raw["observed_min"]),
            observed_max=None if raw["observed_max"] is None else float(raw["observed_max"]),
            observed_categories=None if categories is None else tuple(str(c) for c in categories),
            had_missing=bool(raw["had_missing"]),
        )


@dataclass(frozen=True)
class SchemaSpec:
    columns: tuple[ColumnSpec, ...]  # feature columns only
    target: str | None  # known but not a feature: not reported as an extra column
    excluded: tuple[str, ...]  # likewise
    training_rows: int

    def column(self, name: str) -> ColumnSpec | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_format": SCHEMA_FORMAT,
            "target": self.target,
            "excluded": list(self.excluded),
            "training_rows": self.training_rows,
            "columns": [c.to_dict() for c in self.columns],
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> SchemaSpec:
        if raw.get("schema_format") != SCHEMA_FORMAT:
            raise SchemaError(f"unsupported schema_format {raw.get('schema_format')!r}")
        return SchemaSpec(
            columns=tuple(ColumnSpec.from_dict(c) for c in raw["columns"]),
            target=raw["target"],
            excluded=tuple(str(c) for c in raw["excluded"]),
            training_rows=int(raw["training_rows"]),
        )


@dataclass(frozen=True)
class SchemaReport:
    rows: int  # rows in the data that was checked
    failures: tuple[Finding, ...]  # structural, severity "error"
    observations: tuple[Finding, ...]  # about the training data, severity "info"

    @property
    def ok(self) -> bool:
        """True when there are no structural failures. Observations never change this."""
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": self.rows,
            "structural_failures": [f.to_dict() for f in self.failures],
            "observations": [f.to_dict() for f in self.observations],
        }


# --------------------------------------------------------------------------------------------
# Building a spec
# --------------------------------------------------------------------------------------------


def profile_for_schema(frame: pd.DataFrame) -> DatasetProfile:
    """Profile a text frame with settings that keep every category (see SCHEMA_PROFILE_SETTINGS)."""
    return profile_frame(frame, SCHEMA_PROFILE_SETTINGS)


def build_schema(profile: DatasetProfile, config: TaskConfig) -> SchemaSpec:
    """Build the feature schema from a profile and a task configuration.

    Only feature columns (not the target, not excluded ones) are included. A category set is
    recorded only when the profile holds every distinct value (`top_n` at least as large as the
    number of categories); otherwise it is left out rather than recorded incompletely.
    """
    features = set(config.feature_columns(profile))
    if not features:
        raise SchemaError("no feature columns: the target and excluded columns cover every column")
    specs: list[ColumnSpec] = []
    for column in profile.columns:
        if column.name not in features:
            continue
        numeric = column.numeric if column.kind in NUMERIC_KINDS else None
        complete = column.kind == "categorical" and len(column.top_values) == column.unique
        specs.append(
            ColumnSpec(
                name=column.name,
                position=column.position,
                kind=column.kind,
                observed_min=None if numeric is None else numeric.minimum,
                observed_max=None if numeric is None else numeric.maximum,
                observed_categories=(
                    tuple(sorted(value for value, _ in column.top_values)) if complete else None
                ),
                had_missing=column.missing > 0,
            )
        )
    return SchemaSpec(
        columns=tuple(specs),
        target=config.target,
        excluded=tuple(config.excluded),
        training_rows=profile.rows,
    )


# --------------------------------------------------------------------------------------------
# Number syntax (the same strict rule as the profile)
# --------------------------------------------------------------------------------------------


def _clean_numbers(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """For each cell: is it a clean finite number (strict ASCII syntax), and its float value.

    Missing cells are neither (mask False, value NaN). Nothing is changed.
    """
    text = series.astype("string")
    matched = text.str.fullmatch(NUMBER_RE.pattern).fillna(False).to_numpy(dtype=bool)
    values = np.full(len(text), np.nan, dtype="float64")
    if matched.any():
        values[matched] = pd.to_numeric(text[matched], errors="coerce").to_numpy(dtype="float64")
    finite = matched & np.isfinite(values)
    values[~finite] = np.nan
    return finite, values


def _is_clean_number(series: pd.Series) -> pd.Series:
    """Pandera check: True where a cell is a clean finite number (missing cells are skipped)."""
    finite, _ = _clean_numbers(series)
    return pd.Series(finite, index=series.index)


# --------------------------------------------------------------------------------------------
# Structural failures (Pandera)
# --------------------------------------------------------------------------------------------


def _examples(values: Sequence[str]) -> list[dict[str, Any]]:
    ranked = sorted(Counter(values).items(), key=lambda item: (-item[1], item[0]))
    return [{"value": v, "count": c} for v, c in ranked[:EXAMPLE_LIMIT]]


def _structural_failures(
    spec: SchemaSpec, frame: pd.DataFrame, header: Sequence[str] | None
) -> list[Finding]:
    import pandera.pandas as pa  # lazy: about 1.5 s to import

    found: list[Finding] = []
    if header is not None:
        repeated = sorted(name for name, n in Counter(header).items() if n > 1)
        if repeated:
            found.append(
                Finding(
                    "duplicate_column_names",
                    "error",
                    None,
                    "the header repeats column name(s) " + ", ".join(repr(n) for n in repeated),
                    {"names": repeated},
                )
            )

    columns = {
        c.name: pa.Column(
            checks=(
                [pa.Check(_is_clean_number, name="clean_number")] if c.kind in NUMERIC_KINDS else []
            ),
            nullable=True,
            required=True,
        )
        for c in spec.columns
    }
    schema = pa.DataFrameSchema(columns, strict=False, coerce=False)
    try:
        schema.validate(frame, lazy=True)
        return found
    except pa.errors.SchemaErrors as error:
        cases = error.failure_cases

    position = {c.name: c.position for c in spec.columns}
    malformed: dict[str, list[tuple[str, int]]] = {}
    for _, case in cases.iterrows():
        check = str(case["check"])
        if check == "column_in_dataframe":
            name = str(case["failure_case"])
            found.append(
                Finding(
                    "missing_required_column",
                    "error",
                    name,
                    f"the required column {name!r} is missing",
                    {"position_in_training": position.get(name)},
                )
            )
        elif check == "clean_number":
            index = case["index"]
            malformed.setdefault(str(case["column"]), []).append(
                (str(case["failure_case"]), int(index) + 1 if pd.notna(index) else 0)
            )
        else:  # pragma: no cover - only the two checks above are defined
            found.append(
                Finding(
                    "schema_check_failed",
                    "error",
                    None if pd.isna(case["column"]) else str(case["column"]),
                    f"schema check {check!r} failed",
                    {"check": check, "failure_case": str(case["failure_case"])},
                )
            )
    for name, entries in malformed.items():
        spec_column = spec.column(name)
        kind = spec_column.kind if spec_column else "numeric"
        first_row = min(row for _, row in entries)
        found.append(
            Finding(
                "malformed_numeric",
                "error",
                name,
                f"{len(entries)} value(s) in this {kind} column are not clean numbers (for "
                "example "
                + ", ".join(repr(e["value"]) for e in _examples([v for v, _ in entries])[:3])
                + ")",
                {
                    "cells": len(entries),
                    "first_data_row": first_row,
                    "examples": _examples([v for v, _ in entries]),
                },
            )
        )
    return found


# --------------------------------------------------------------------------------------------
# Observations (plain pandas/numpy; never failures)
# --------------------------------------------------------------------------------------------


def _pct(part: int, whole: int) -> float:
    return significant(100.0 * part / whole) if whole else 0.0


def _observations(spec: SchemaSpec, frame: pd.DataFrame) -> list[Finding]:
    found: list[Finding] = []
    rows = len(frame)
    for column in spec.columns:
        if column.name not in frame.columns:
            continue
        series = frame[column.name]
        present = series.dropna()
        blanks = rows - len(present)
        name = column.name

        if blanks and not column.had_missing:
            found.append(
                Finding(
                    "new_blanks",
                    "info",
                    name,
                    f"{blanks} blank value(s) ({_pct(blanks, rows):g}% of rows) in a column that "
                    "had none in the training data",
                    {"cells": blanks, "percent_of_rows": _pct(blanks, rows)},
                )
            )

        if column.observed_categories is not None and len(present):
            known = set(column.observed_categories)
            unseen = present[~present.isin(known)]
            if len(unseen):
                counted = Counter(str(v) for v in unseen)
                found.append(
                    Finding(
                        "unseen_categories",
                        "info",
                        name,
                        f"{len(counted)} category value(s) not seen in the training data "
                        f"({len(unseen)} cell(s), {_pct(len(unseen), len(present)):g}% of "
                        "non-blank values), for example "
                        + ", ".join(repr(e["value"]) for e in _examples(list(unseen))[:3]),
                        {
                            "cells": len(unseen),
                            "distinct": len(counted),
                            "percent_of_values": _pct(len(unseen), len(present)),
                            "examples": _examples(list(unseen)),
                        },
                    )
                )

        if column.kind in NUMERIC_KINDS and len(present):
            finite, values = _clean_numbers(present)
            good = values[finite]
            if good.size:
                low = column.observed_min
                high = column.observed_max
                below = above = 0
                if low is not None and high is not None:
                    # Profile limits are rounded to 12 significant digits, so round the same way
                    # before comparing; otherwise a value equal to the maximum could look outside.
                    for value in good[(good < low) | (good > high)]:
                        rounded = significant(float(value))
                        below += rounded < low
                        above += rounded > high
                if below or above:
                    out = below + above
                    found.append(
                        Finding(
                            "out_of_observed_range",
                            "info",
                            name,
                            f"{out} value(s) ({_pct(out, int(good.size)):g}% of numbers) lie "
                            f"outside the range seen in training ({low:g} to {high:g}): "
                            f"{below} below, {above} above. This may be valid data, not an error",
                            {
                                "below": int(below),
                                "above": int(above),
                                "percent_of_values": _pct(out, int(good.size)),
                                "observed_min": low,
                                "observed_max": high,
                                "data_min": significant(float(good.min())),
                                "data_max": significant(float(good.max())),
                            },
                        )
                    )
                if column.kind == "integer":
                    texts = present.astype("string")[finite]
                    fractional = int((~texts.str.fullmatch(INTEGER_RE.pattern)).sum())
                    if fractional:
                        found.append(
                            Finding(
                                "non_integer_values",
                                "info",
                                name,
                                f"{fractional} value(s) are not written as whole numbers in a "
                                "column that held only whole numbers in training",
                                {"cells": fractional},
                            )
                        )

    known_names = {c.name for c in spec.columns} | set(spec.excluded)
    if spec.target is not None:
        known_names.add(spec.target)
    extras = [str(c) for c in frame.columns if str(c) not in known_names]
    if extras:
        found.append(
            Finding(
                "extra_columns",
                "info",
                None,
                f"{len(extras)} column(s) are not part of the schema and are ignored: "
                + ", ".join(repr(c) for c in extras[:5]),
                {"columns": extras},
            )
        )
    return found


# --------------------------------------------------------------------------------------------
# Checking data against a spec
# --------------------------------------------------------------------------------------------


def check_frame(
    spec: SchemaSpec, frame: pd.DataFrame, header: Sequence[str] | None = None
) -> SchemaReport:
    """Check a text frame (from `read_text_frame`) against a spec. The frame is not changed.

    Pass the raw `header` to also catch duplicate column names (pandas renames duplicates when it
    reads a file, so they cannot be seen in the frame itself).
    """
    failures = _structural_failures(spec, frame, header)
    position = {c.name: c.position for c in spec.columns}
    failures.sort(key=lambda f: (position.get(f.column or "", 0), f.code, f.column or ""))
    observations = _observations(spec, frame)
    observations.sort(key=lambda f: (position.get(f.column or "", 0), f.code))
    return SchemaReport(rows=len(frame), failures=tuple(failures), observations=tuple(observations))


def check_csv_bytes(spec: SchemaSpec, data: bytes) -> SchemaReport:
    """Check canonical-or-not CSV bytes against a spec (reading every cell as text)."""
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8")))
        header = next(reader, [])
        # pandas would silently reshape a ragged file (extra leading fields become an index), so
        # insist that every non-blank row has exactly as many fields as the header.
        for number, row in enumerate(reader, start=1):
            if row and len(row) != len(header):
                raise SchemaError(
                    f"the data could not be read as CSV: data row {number} has {len(row)} "
                    f"field(s) but the header has {len(header)}"
                )
        frame = read_text_frame(data)
    except (UnicodeDecodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise SchemaError(f"the data could not be read as CSV: {e}") from e
    return check_frame(spec, frame, header)
