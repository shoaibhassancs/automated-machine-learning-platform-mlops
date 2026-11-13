"""Preprocessing pipeline builder (BUILD_PLAN Step 7).

One place builds the preprocessing for training **and** later inference, so serving cannot differ
from training. The pipeline takes the data **as text** (a frame from `read_text_frame`, which is
what a stored dataset version or a prediction request becomes) and does every conversion inside:

* `integer` / `float` columns: strict number parsing, median imputation, and standard scaling for
  algorithms that need it (linear models);
* `categorical` columns: most-frequent-value imputation, then one-hot encoding that ignores
  categories not seen in fitting;
* `text`, `code`, `mixed` and `empty` columns are **not model inputs**. `ColumnPlan.dropped` names
  each one with the reason, so nothing is discarded silently.

Everything learned from data (medians, scaling, category lists) is learned by `fit`, inside a
scikit-learn `Pipeline`. Cross-validation therefore fits the preprocessing separately on each fold's
training part (see `train.py`); it is never fitted on the whole dataset first.

A malformed non-empty value in a numeric column (`abc`, `1_000`, `nan`) is **not** turned into a
missing value: it raises `MalformedValueError`. Data is expected to pass the Step 4B schema check
first; this is the second line of defence. An empty cell is missing and is imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from automl_ops.profile import NUMBER_RE, DatasetProfile
from automl_ops.task import TaskConfig

Array = NDArray[Any]

PIPELINE_FORMAT = 1


class PipelineError(ValueError):
    """The preprocessing cannot be built (for example, no usable input columns)."""


class MalformedValueError(ValueError):
    """A non-empty value in a numeric column is not a clean finite number."""


@dataclass(frozen=True)
class DroppedColumn:
    name: str
    kind: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind, "reason": self.reason}


@dataclass(frozen=True)
class ColumnPlan:
    """Which feature columns are used, how, and which are left out (with the reason)."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dropped: tuple[DroppedColumn, ...]

    @property
    def used(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "dropped": [d.to_dict() for d in self.dropped],
        }


def build_column_plan(profile: DatasetProfile, config: TaskConfig) -> ColumnPlan:
    """Decide each feature column's treatment from its profile kind (target and excluded left out).

    Raises `PipelineError` if no column can be used as a model input.
    """
    limit = profile.settings.max_categories
    reasons = {
        "text": f"more than {limit} distinct text values; one-hot encoding would create too many "
        "columns",
        "code": "values with leading zeros look like codes or identifiers, not quantities or "
        "categories",
        "mixed": "some values look like numbers and others do not",
        "empty": "every value is missing",
    }
    features = set(config.feature_columns(profile))
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: list[DroppedColumn] = []
    for column in profile.columns:
        if column.name not in features:
            continue
        if column.kind in ("integer", "float"):
            numeric.append(column.name)
        elif column.kind == "categorical":
            categorical.append(column.name)
        else:
            dropped.append(DroppedColumn(column.name, column.kind, reasons[column.kind]))
    plan = ColumnPlan(tuple(numeric), tuple(categorical), tuple(dropped))
    if not plan.used:
        why = "; ".join(f"{d.name!r} ({d.kind}): {d.reason}" for d in dropped)
        raise PipelineError(
            "no usable input columns: every feature column is left out"
            + (f" ({why})" if why else "")
        )
    return plan


# --------------------------------------------------------------------------------------------
# Conversion steps (module-level functions, so a fitted pipeline can be saved and loaded)
# --------------------------------------------------------------------------------------------


def text_to_number(frame: pd.DataFrame) -> Array:
    """Strictly parse text columns to floats: an empty cell becomes NaN, anything else must be a
    clean finite number or `MalformedValueError` is raised (never silently turned into NaN)."""
    out = np.full(frame.shape, np.nan, dtype="float64")
    for j, name in enumerate(frame.columns):
        column = frame[name]
        present = column.notna().to_numpy(dtype=bool)
        if not present.any():
            continue
        text = column[present].astype(str)
        matched = text.str.fullmatch(NUMBER_RE.pattern).to_numpy(dtype=bool)
        values = np.full(len(text), np.nan, dtype="float64")
        if matched.any():
            values[matched] = pd.to_numeric(text[matched], errors="coerce").to_numpy(
                dtype="float64"
            )
        good = matched & np.isfinite(values)
        if not good.all():
            examples = ", ".join(repr(v) for v in text[~good].unique()[:3])
            raise MalformedValueError(
                f"column {name!r} holds value(s) that are not clean numbers (for example "
                f"{examples}); validate the data against the schema first"
            )
        out[present, j] = values
    return out


def text_to_category(frame: pd.DataFrame) -> Array:
    """Text columns as an object array with NaN for empty cells (original text otherwise)."""
    converted: Array = frame.astype(object).where(frame.notna(), np.nan).to_numpy()
    return converted


def build_preprocessor(plan: ColumnPlan, *, scale_numeric: bool) -> ColumnTransformer:
    """The preprocessing step. Nothing is learned until `fit` is called on a training part."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if plan.numeric:
        steps: list[tuple[str, Any]] = [
            (
                "parse",
                FunctionTransformer(text_to_number, validate=False, feature_names_out="one-to-one"),
            ),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), list(plan.numeric)))
    if plan.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "parse",
                            FunctionTransformer(
                                text_to_category, validate=False, feature_names_out="one-to-one"
                            ),
                        ),
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False, dtype=np.float64
                            ),
                        ),
                    ]
                ),
                list(plan.categorical),
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(plan: ColumnPlan, estimator: Any, *, scale_numeric: bool) -> Pipeline:
    """Preprocessing followed by the estimator, as one unfitted scikit-learn `Pipeline`.

    Fitting this pipeline on a training part learns the imputation values, scaling and category
    lists from that part only, which is what keeps cross-validation free of preprocessing leakage.
    """
    return Pipeline(
        [("prep", build_preprocessor(plan, scale_numeric=scale_numeric)), ("model", estimator)]
    )
