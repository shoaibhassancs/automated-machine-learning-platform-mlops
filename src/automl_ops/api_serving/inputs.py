"""Validating prediction input against the model's feature schema (P4-7, BUILD_PLAN Step 21).

A row is a JSON object with **exactly** the model's input columns: a missing column and an extra
column are both refused (a silent default or a silently ignored column would hide a caller's
mistake). Values follow the training data's text rules: `null` or an empty string is a missing
value (the model imputes it); a numeric column takes a JSON number or a clean number as text; a
categorical column takes text. A bad value is refused with its row and column, never turned into a
missing value. An unseen category is *not* an error (the model handles it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from automl_ops.profile import NUMBER_RE

MAX_PROBLEMS = 20
SHOW = 40  # characters of an offending value shown in a message


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "numeric" | "categorical"


@dataclass(frozen=True)
class Problem:
    row: int | None
    column: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"row": self.row, "column": self.column, "message": self.message}


class InvalidInput(ValueError):
    def __init__(self, problems: list[Problem], truncated: bool = False) -> None:
        super().__init__("the input does not match the model's feature schema")
        self.problems = problems
        self.truncated = truncated


def _shown(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= SHOW else text[: SHOW - 1] + "…"


def _numeric_text(value: Any) -> tuple[str | None, str | None]:
    """(text for the pipeline, problem) for one value of a numeric column."""
    if isinstance(value, bool):
        return None, f"{_shown(value)} is a boolean, not a number"
    if isinstance(value, int):
        return str(value), None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, f"{_shown(value)} is not a finite number"
        return repr(value), None
    if isinstance(value, str):
        if not NUMBER_RE.fullmatch(value) or not math.isfinite(float(value)):
            return None, f"{_shown(value)} is not a clean number"
        return value, None
    return None, f"{_shown(value)} is not a number or text"


def prepare_rows(columns: list[Column], rows: list[dict[str, Any]]) -> pd.DataFrame:
    """The rows as the text frame the pipeline expects, or `InvalidInput` listing what is wrong."""
    expected = [c.name for c in columns]
    wanted = set(expected)
    problems: list[Problem] = []
    cells: list[list[str | None]] = []
    for index, row in enumerate(rows):
        have = set(row)
        missing = [c for c in expected if c not in have]
        extra = sorted(have - wanted)
        if missing:
            problems.append(Problem(index, None, f"missing column(s): {missing}"))
        if extra:
            problems.append(Problem(index, None, f"unknown column(s): {extra}"))
        if missing or extra:
            continue
        values: list[str | None] = []
        for column in columns:
            value = row[column.name]
            if value is None or value == "":
                values.append(None)
            elif column.kind == "numeric":
                text, problem = _numeric_text(value)
                if problem is not None:
                    problems.append(Problem(index, column.name, problem))
                values.append(text)
            elif isinstance(value, str):
                values.append(value)
            else:
                problems.append(
                    Problem(
                        index,
                        column.name,
                        f"{_shown(value)} is not text (give categorical values as strings)",
                    )
                )
                values.append(None)
        cells.append(values)
    if problems:
        raise InvalidInput(problems[:MAX_PROBLEMS], truncated=len(problems) > MAX_PROBLEMS)
    return pd.DataFrame(cells, columns=expected, dtype="string")
