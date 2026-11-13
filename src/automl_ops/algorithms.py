"""Algorithm catalogue and parameter schemas (BUILD_PLAN Step 8).

A closed, typed registry of the selectable algorithms. Each has editable parameters with a default
and bounds. `validate_params` rejects unknown keys, wrong types and out-of-range values; there is
no way to name an arbitrary estimator or pass executable code. Bounds are **provisional** and
chosen to be sensible for a local single-user machine (see docs/training.md).

Classification: logistic regression, random forest, histogram gradient boosting.
Regression: ridge, random forest, histogram gradient boosting.

Fixed choices (not editable): logistic regression uses the L2 penalty and the `lbfgs` solver; the
random forest runs on one thread; gradient boosting has early stopping off (so its result does not
depend on an internal validation split) and keeps scikit-learn's other defaults.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from automl_ops.task import TASK_TYPES, TaskType

ParamKind = Literal["int", "float", "optional_int", "choice"]


class AlgorithmError(ValueError):
    """An unknown algorithm, or parameters that fail validation."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: ParamKind
    default: Any
    description: str
    minimum: float | None = None  # inclusive; for int, float and optional_int
    maximum: float | None = None  # inclusive
    choices: tuple[str, ...] = ()  # for kind "choice"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": list(self.choices),
            "description": self.description,
        }

    def check(self, value: Any) -> str | None:
        """None if `value` is acceptable, otherwise a short reason."""
        if self.kind == "choice":
            if not isinstance(value, str) or value not in self.choices:
                return f"must be one of {list(self.choices)}"
            return None
        if self.kind == "optional_int" and value is None:
            return None
        is_int = isinstance(value, int) and not isinstance(value, bool)
        if self.kind in ("int", "optional_int"):
            if not is_int:
                return "must be a whole number" + (
                    " or null" if self.kind == "optional_int" else ""
                )
        else:  # float
            if isinstance(value, bool) or not isinstance(value, int | float):
                return "must be a number"
            if not math.isfinite(value):
                return "must be a finite number"
        assert self.minimum is not None and self.maximum is not None
        if not self.minimum <= value <= self.maximum:
            return f"must be between {self.minimum:g} and {self.maximum:g}"
        return None


Builder = Callable[[dict[str, Any], int], Any]


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    task: TaskType
    label: str
    params: tuple[ParamSpec, ...]
    scale_numeric: bool  # linear models need scaled numeric inputs; tree models do not
    builder: Builder

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def validate_params(self, raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Defaults filled in; unknown keys, wrong types and out-of-range values are rejected."""
        given = dict(raw or {})
        known = {p.name: p for p in self.params}
        problems = [
            f"unknown parameter {key!r} (allowed: {sorted(known)})"
            for key in given
            if key not in known
        ]
        merged = self.defaults()
        for name, value in given.items():
            spec = known.get(name)
            if spec is None:
                continue
            reason = spec.check(value)
            if reason:
                problems.append(f"{name}: {reason} (got {value!r})")
            else:
                merged[name] = value
        if problems:
            raise AlgorithmError(f"{self.task} algorithm {self.key!r}: " + "; ".join(problems))
        return merged

    def build(self, params: Mapping[str, Any] | None, seed: int) -> Any:
        """A new, unfitted scikit-learn estimator with validated parameters."""
        return self.builder(self.validate_params(params), seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "task": self.task,
            "label": self.label,
            "scale_numeric": self.scale_numeric,
            "params": [p.to_dict() for p in self.params],
        }


# --------------------------------------------------------------------------------------------
# Parameter definitions (provisional bounds)
# --------------------------------------------------------------------------------------------

_C = ParamSpec("C", "float", 1.0, "inverse regularisation strength (smaller = stronger)", 1e-4, 1e4)
_MAX_ITER = ParamSpec("max_iter", "int", 1000, "maximum solver iterations", 50, 5000)
_CLASS_WEIGHT = ParamSpec(
    "class_weight",
    "choice",
    "none",
    "'balanced' weights classes inversely to their frequency",
    choices=("none", "balanced"),
)
_ALPHA = ParamSpec("alpha", "float", 1.0, "regularisation strength", 1e-4, 1e4)

_TREES = ParamSpec("n_estimators", "int", 100, "number of trees", 10, 500)
_FOREST_DEPTH = ParamSpec(
    "max_depth", "optional_int", None, "maximum tree depth (null = no limit)", 1, 50
)
_FOREST_LEAF = ParamSpec("min_samples_leaf", "int", 1, "minimum rows in a leaf", 1, 50)


def _forest_features(default: str) -> ParamSpec:
    return ParamSpec(
        "max_features",
        "choice",
        default,
        "features considered per split ('all' = every feature)",
        choices=("sqrt", "log2", "all"),
    )


_BOOST_ITER = ParamSpec("max_iter", "int", 100, "number of boosting rounds", 10, 500)
_BOOST_RATE = ParamSpec("learning_rate", "float", 0.1, "shrinkage per round", 0.01, 1.0)
_BOOST_DEPTH = ParamSpec(
    "max_depth", "optional_int", None, "maximum tree depth (null = no limit)", 1, 30
)
_BOOST_LEAF = ParamSpec("min_samples_leaf", "int", 20, "minimum rows in a leaf", 1, 200)
_BOOST_L2 = ParamSpec("l2_regularization", "float", 0.0, "L2 penalty on leaf values", 0.0, 100.0)


def _max_features(value: str) -> str | float:
    return 1.0 if value == "all" else value


def _logistic(p: dict[str, Any], seed: int) -> Any:
    return LogisticRegression(
        C=p["C"],
        max_iter=p["max_iter"],
        class_weight=None if p["class_weight"] == "none" else "balanced",
        solver="lbfgs",
        random_state=seed,
    )


def _ridge(p: dict[str, Any], seed: int) -> Any:
    return Ridge(alpha=p["alpha"])


def _forest_classifier(p: dict[str, Any], seed: int) -> Any:
    return RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        max_features=_max_features(p["max_features"]),
        n_jobs=1,
        random_state=seed,
    )


def _forest_regressor(p: dict[str, Any], seed: int) -> Any:
    return RandomForestRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        max_features=_max_features(p["max_features"]),
        n_jobs=1,
        random_state=seed,
    )


def _boost_classifier(p: dict[str, Any], seed: int) -> Any:
    return HistGradientBoostingClassifier(
        max_iter=p["max_iter"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        l2_regularization=p["l2_regularization"],
        early_stopping=False,
        random_state=seed,
    )


def _boost_regressor(p: dict[str, Any], seed: int) -> Any:
    return HistGradientBoostingRegressor(
        max_iter=p["max_iter"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        l2_regularization=p["l2_regularization"],
        early_stopping=False,
        random_state=seed,
    )


_BOOST_PARAMS = (_BOOST_ITER, _BOOST_RATE, _BOOST_DEPTH, _BOOST_LEAF, _BOOST_L2)

_CATALOGUE: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        "logistic_regression",
        "classification",
        "Logistic regression",
        (_C, _MAX_ITER, _CLASS_WEIGHT),
        True,
        _logistic,
    ),
    AlgorithmSpec(
        "random_forest",
        "classification",
        "Random forest",
        (_TREES, _FOREST_DEPTH, _FOREST_LEAF, _forest_features("sqrt")),
        False,
        _forest_classifier,
    ),
    AlgorithmSpec(
        "hist_gradient_boosting",
        "classification",
        "Histogram gradient boosting",
        _BOOST_PARAMS,
        False,
        _boost_classifier,
    ),
    AlgorithmSpec("ridge", "regression", "Ridge regression", (_ALPHA,), True, _ridge),
    AlgorithmSpec(
        "random_forest",
        "regression",
        "Random forest",
        (_TREES, _FOREST_DEPTH, _FOREST_LEAF, _forest_features("all")),
        False,
        _forest_regressor,
    ),
    AlgorithmSpec(
        "hist_gradient_boosting",
        "regression",
        "Histogram gradient boosting",
        _BOOST_PARAMS,
        False,
        _boost_regressor,
    ),
)


def algorithms_for(task: TaskType) -> tuple[AlgorithmSpec, ...]:
    """The selectable algorithms for a task, in a fixed order."""
    if task not in TASK_TYPES:
        raise AlgorithmError(f"unknown task {task!r} (use one of {list(TASK_TYPES)})")
    return tuple(a for a in _CATALOGUE if a.task == task)


def get_algorithm(task: TaskType, key: str) -> AlgorithmSpec:
    for spec in algorithms_for(task):
        if spec.key == key:
            return spec
    allowed = [a.key for a in algorithms_for(task)]
    raise AlgorithmError(f"unknown {task} algorithm {key!r} (available: {allowed})")
