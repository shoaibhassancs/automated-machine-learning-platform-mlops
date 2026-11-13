"""Metrics and chart data (BUILD_PLAN Step 10).

Plain functions over arrays that return JSON-ready numbers (no plotting library, no NaN or
infinity): a metric that cannot be defined for the rows given is `None`, and `undefined` says why.

**Class order.** Classification functions take `classes`, the one sorted list of class labels; the
confusion matrix, the per-class tables and the probability columns all follow it. For two classes
the **positive class is `classes[1]`**.

**Macro averages** (precision, recall, F1) average over the classes that occur in `y_true` for the
rows being scored, with 0 where a class is never predicted (`zero_division=0`). **ROC-AUC** is the
binary area for two classes and the **one-vs-rest macro average** for more; it needs every class to
occur in `y_true`, otherwise it is undefined. For the pooled out-of-fold rows every class occurs.

**Curves.** ROC and precision-recall points come from predicted probabilities: one curve for a
binary task, one one-vs-rest curve per class (the 10 largest classes) otherwise, each thinned to
at most 200 points. A curve needs both positive and negative rows, otherwise it is skipped and
noted.

**Regression.** RMSE, MAE and R². The residual is `actual - predicted`, listed per observation
with the row number it came from. R² is undefined when the actual values are all equal.

Nothing here knows about a test set: it scores whatever predictions it is given.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from automl_ops.profile import significant

Array = NDArray[Any]

CURVE_MAX_POINTS = 200
MAX_CURVE_CLASSES = 10
RESIDUAL_MAX_POINTS = 1000

CLASSIFICATION_METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc")
REGRESSION_METRICS = ("rmse", "mae", "r2")


class EvaluationError(ValueError):
    """The inputs cannot be scored (mismatched lengths, labels outside the class list, ...)."""


def number(value: float) -> float | None:
    """A JSON-safe number: 12 significant digits, or None when not finite."""
    value = float(value)
    return significant(value) if math.isfinite(value) else None


def _thin(indices: int, limit: int) -> Array:
    """Up to `limit` evenly spaced positions out of `indices`, always keeping the first and last."""
    if indices <= limit:
        return np.arange(indices)
    chosen: Array = np.unique(np.round(np.linspace(0, indices - 1, limit)).astype(int))
    return chosen


def _check_lengths(**arrays: Array) -> int:
    lengths = {name: len(a) for name, a in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise EvaluationError(f"inputs have different lengths: {lengths}")
    n = next(iter(lengths.values()))
    if n == 0:
        raise EvaluationError("nothing to score: no rows")
    return n


# --------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------


def _check_labels(y_true: Array, y_pred: Array, classes: Sequence[str]) -> None:
    known = set(classes)
    stray = (set(y_true.tolist()) | set(y_pred.tolist())) - known
    if stray:
        raise EvaluationError(f"labels outside the class list: {sorted(stray)[:5]}")


def classification_scalars(
    y_true: Array, y_pred: Array, proba: Array | None, classes: Sequence[str]
) -> tuple[dict[str, float | None], dict[str, str]]:
    """The five headline metrics and, for any that is undefined, the reason."""
    _check_lengths(y_true=y_true, y_pred=y_pred, **({} if proba is None else {"proba": proba}))
    _check_labels(y_true, y_pred, classes)
    present = [c for c in classes if (y_true == c).any()]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="macro", zero_division=0
    )
    scalars: dict[str, float | None] = {
        "accuracy": number(accuracy_score(y_true, y_pred)),
        "precision_macro": number(precision),
        "recall_macro": number(recall),
        "f1_macro": number(f1),
        "roc_auc": None,
    }
    undefined: dict[str, str] = {}
    if proba is None:
        undefined["roc_auc"] = "no predicted probabilities were given"
    elif len(present) < len(classes):
        missing = [c for c in classes if c not in present]
        undefined["roc_auc"] = f"class {missing[0]!r} does not occur in these rows"
    elif len(classes) < 2:
        undefined["roc_auc"] = "fewer than two classes"
    else:
        if len(classes) == 2:
            area = roc_auc_score(y_true == classes[1], proba[:, 1])
        else:
            area = roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=list(classes)
            )
        scalars["roc_auc"] = number(area)
    return scalars, undefined


def _curve_points(x: Array, y: Array) -> tuple[list[float | None], list[float | None]]:
    keep = _thin(len(x), CURVE_MAX_POINTS)
    return [number(v) for v in x[keep]], [number(v) for v in y[keep]]


def classification_curves(
    y_true: Array, proba: Array, classes: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """ROC and precision-recall points per curve, and notes about anything skipped."""
    notes: list[str] = []
    if len(classes) == 2:
        targets = [(classes[1], 1)]
    else:
        counts = {c: int((y_true == c).sum()) for c in classes}
        order = sorted(classes, key=lambda c: (-counts[c], c))
        shown = order[:MAX_CURVE_CLASSES]
        if len(order) > len(shown):
            notes.append(
                f"curves are shown for the {MAX_CURVE_CLASSES} largest of {len(order)} classes"
            )
        targets = [(c, list(classes).index(c)) for c in shown]
    roc_curves: list[dict[str, Any]] = []
    pr_curves: list[dict[str, Any]] = []
    for label, column in targets:
        positive = y_true == label
        if positive.all() or not positive.any():
            notes.append(f"no curve for class {label!r}: it needs both positive and negative rows")
            continue
        fpr, tpr, _ = roc_curve(positive, proba[:, column])
        precision, recall, _ = precision_recall_curve(positive, proba[:, column])
        fx, ty = _curve_points(fpr, tpr)
        rx, py = _curve_points(recall, precision)
        roc_curves.append({"class": label, "fpr": fx, "tpr": ty})
        pr_curves.append({"class": label, "recall": rx, "precision": py})
    return roc_curves, pr_curves, notes


def evaluate_classification(
    y_true: Array,
    y_pred: Array,
    proba: Array | None,
    classes: Sequence[str],
    *,
    curves: bool = True,
) -> dict[str, Any]:
    """Headline metrics, the confusion matrix, per-class tables and (optionally) curve points."""
    scalars, undefined = classification_scalars(y_true, y_pred, proba, classes)
    labels = list(classes)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    result: dict[str, Any] = {
        "rows": int(len(y_true)),
        "classes": labels,
        "metrics": scalars,
        "undefined": undefined,
        "confusion_matrix": [[int(v) for v in row] for row in matrix],
        "per_class": [
            {
                "class": c,
                "precision": number(precision[i]),
                "recall": number(recall[i]),
                "f1": number(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(labels)
        ],
    }
    if curves:
        if proba is None:
            result.update({"roc_curves": [], "pr_curves": [], "curve_notes": ["no probabilities"]})
        else:
            roc, pr, notes = classification_curves(y_true, proba, classes)
            result.update({"roc_curves": roc, "pr_curves": pr, "curve_notes": notes})
    return result


# --------------------------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------------------------


def regression_scalars(
    y_true: Array, y_pred: Array
) -> tuple[dict[str, float | None], dict[str, str]]:
    _check_lengths(y_true=y_true, y_pred=y_pred)
    scalars: dict[str, float | None] = {
        "rmse": number(math.sqrt(float(np.mean((y_true - y_pred) ** 2)))),
        "mae": number(mean_absolute_error(y_true, y_pred)),
        "r2": None,
    }
    undefined: dict[str, str] = {}
    if float(np.ptp(y_true)) == 0.0:
        undefined["r2"] = "the actual values are all equal, so R² is undefined"
    else:
        scalars["r2"] = number(r2_score(y_true, y_pred))
    return scalars, undefined


def evaluate_regression(
    y_true: Array, y_pred: Array, rows: Array, *, residuals: bool = True
) -> dict[str, Any]:
    """RMSE, MAE, R² and residual points (`actual - predicted`), each with its row number."""
    _check_lengths(y_true=y_true, y_pred=y_pred, rows=rows)
    scalars, undefined = regression_scalars(y_true, y_pred)
    result: dict[str, Any] = {"rows": int(len(y_true)), "metrics": scalars, "undefined": undefined}
    if residuals:
        keep = _thin(len(y_true), RESIDUAL_MAX_POINTS)
        result["residual_points"] = [
            {
                "row": int(rows[i]),
                "actual": number(y_true[i]),
                "predicted": number(y_pred[i]),
                "residual": number(y_true[i] - y_pred[i]),
            }
            for i in keep
        ]
        result["residual_points_note"] = (
            None
            if len(keep) == len(y_true)
            else f"{len(keep)} evenly spaced points shown out of {len(y_true)}"
        )
    return result


# --------------------------------------------------------------------------------------------
# Summaries across folds
# --------------------------------------------------------------------------------------------


def summarize_folds(
    fold_scalars: Sequence[dict[str, float | None]], names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Mean and sample standard deviation of each metric over the folds where it is defined."""
    summary: dict[str, dict[str, Any]] = {}
    for name in names:
        values = [s[name] for s in fold_scalars if s.get(name) is not None]
        floats = np.array([v for v in values if v is not None], dtype="float64")
        summary[name] = {
            "mean": number(float(floats.mean())) if floats.size else None,
            "std": number(float(floats.std(ddof=1))) if floats.size > 1 else None,
            "folds_defined": int(floats.size),
            "folds": len(fold_scalars),
        }
    return summary
