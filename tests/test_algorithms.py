"""Algorithm catalogue tests: the closed list, defaults, and rejection of anything not allowed."""

from __future__ import annotations

import json
import math
from typing import Any, cast

import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from automl_ops.algorithms import AlgorithmError, algorithms_for, get_algorithm
from automl_ops.task import TaskType

TASKS: tuple[TaskType, TaskType] = ("classification", "regression")


def test_the_catalogue_is_the_six_approved_algorithms_in_a_fixed_order() -> None:
    assert [a.key for a in algorithms_for("classification")] == [
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]
    assert [a.key for a in algorithms_for("regression")] == [
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    ]


def test_only_linear_models_ask_for_scaled_inputs() -> None:
    scaled = {(a.task, a.key) for t in TASKS for a in algorithms_for(t) if a.scale_numeric}
    assert scaled == {("classification", "logistic_regression"), ("regression", "ridge")}


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_default_configuration_is_valid_and_builds(task: TaskType) -> None:
    for spec in algorithms_for(task):
        assert spec.validate_params({}) == spec.defaults()
        assert spec.build({}, seed=7) is not None
        assert all(p.description for p in spec.params)


def test_built_estimators_have_the_documented_fixed_settings() -> None:
    forest = get_algorithm("classification", "random_forest").build({}, 3)
    assert isinstance(forest, RandomForestClassifier)
    assert (forest.n_jobs, forest.random_state, forest.max_features) == (1, 3, "sqrt")
    forest_r = get_algorithm("regression", "random_forest").build({}, 3)
    assert isinstance(forest_r, RandomForestRegressor) and forest_r.max_features == 1.0
    boosters: list[tuple[TaskType, type]] = [
        ("classification", HistGradientBoostingClassifier),
        ("regression", HistGradientBoostingRegressor),
    ]
    for task, cls in boosters:
        boost = get_algorithm(task, "hist_gradient_boosting").build({}, 5)
        settings = boost.get_params()  # read before the type check narrows `boost`
        assert type(boost) is cls
        assert settings["early_stopping"] is False and settings["random_state"] == 5
    logistic = get_algorithm("classification", "logistic_regression").build(
        {"C": 2.0, "class_weight": "balanced"}, 1
    )
    assert isinstance(logistic, LogisticRegression)
    assert (logistic.C, logistic.class_weight, logistic.solver) == (2.0, "balanced", "lbfgs")
    assert isinstance(get_algorithm("regression", "ridge").build({"alpha": 0.5}, 1), Ridge)


def test_valid_overrides_are_applied_and_the_rest_keep_their_defaults() -> None:
    spec = get_algorithm("classification", "random_forest")
    merged = spec.validate_params({"n_estimators": 30, "max_depth": None, "max_features": "all"})
    assert merged["n_estimators"] == 30 and merged["max_features"] == "all"
    assert merged["min_samples_leaf"] == spec.defaults()["min_samples_leaf"]
    assert spec.validate_params({"max_depth": 7})["max_depth"] == 7  # optional int accepts a number


@pytest.mark.parametrize(
    ("key", "name", "low", "high"),
    [
        ("logistic_regression", "C", 1e-4, 1e4),
        ("logistic_regression", "max_iter", 50, 5000),
        ("random_forest", "n_estimators", 10, 500),
        ("random_forest", "max_depth", 1, 50),
        ("random_forest", "min_samples_leaf", 1, 50),
        ("hist_gradient_boosting", "max_iter", 10, 500),
        ("hist_gradient_boosting", "learning_rate", 0.01, 1.0),
        ("hist_gradient_boosting", "max_depth", 1, 30),
        ("hist_gradient_boosting", "min_samples_leaf", 1, 200),
        ("hist_gradient_boosting", "l2_regularization", 0.0, 100.0),
    ],
)
def test_bounds_are_inclusive_and_anything_beyond_is_rejected(
    key: str, name: str, low: float, high: float
) -> None:
    spec = get_algorithm("classification", key)
    param = next(p for p in spec.params if p.name == name)
    cast = int if param.kind in ("int", "optional_int") else float
    assert spec.validate_params({name: cast(low)})[name] == cast(low)
    assert spec.validate_params({name: cast(high)})[name] == cast(high)
    step = 1 if cast is int else low / 10 or 0.5
    for outside in (cast(low - step), cast(high + step)):
        with pytest.raises(AlgorithmError, match=name):
            spec.validate_params({name: outside})


def test_ridge_alpha_bounds() -> None:
    spec = get_algorithm("regression", "ridge")
    assert spec.validate_params({"alpha": 1e-4})["alpha"] == 1e-4
    for bad in (0.0, 1e-5, 1e5, -1.0):
        with pytest.raises(AlgorithmError, match="alpha"):
            spec.validate_params({"alpha": bad})


@pytest.mark.parametrize(
    "bad",
    [
        {"unknown_key": 1},
        {"n_estimators": True},  # a bool is not a whole number
        {"n_estimators": 10.5},
        {"n_estimators": "100"},
        {"max_depth": 0},
        {"max_features": "auto"},
        {"max_features": 3},
        {"min_samples_leaf": None},  # not an optional parameter
        {"n_estimators": float("nan")},
    ],
)
def test_unknown_keys_wrong_types_and_bad_values_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(AlgorithmError):
        get_algorithm("classification", "random_forest").validate_params(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "1.0", None, True])
def test_float_parameters_must_be_finite_numbers(bad: Any) -> None:
    with pytest.raises(AlgorithmError):
        get_algorithm("classification", "logistic_regression").validate_params({"C": bad})


def test_nothing_executable_or_arbitrary_can_be_selected() -> None:
    spec = get_algorithm("classification", "logistic_regression")
    for payload in ("__import__('os').system('x')", "sklearn.svm.SVC", "lambda: 1"):
        with pytest.raises(AlgorithmError):
            spec.validate_params({"class_weight": payload})
    for key in ("svm", "sklearn.svm.SVC", "__import__", ""):
        with pytest.raises(AlgorithmError, match="unknown"):
            get_algorithm("classification", key)
    with pytest.raises(AlgorithmError, match="unknown"):
        get_algorithm("regression", "logistic_regression")  # right name, wrong task
    with pytest.raises(AlgorithmError, match="unknown task"):
        algorithms_for(cast(TaskType, "clustering"))


def test_all_problems_are_reported_together_and_nothing_is_changed() -> None:
    spec = get_algorithm("classification", "random_forest")
    given = {"n_estimators": 9999, "bogus": 1, "max_features": "x"}
    with pytest.raises(AlgorithmError) as info:
        spec.validate_params(given)
    message = str(info.value)
    assert "n_estimators" in message and "bogus" in message and "max_features" in message
    assert given == {"n_estimators": 9999, "bogus": 1, "max_features": "x"}


def test_specs_are_json_ready_for_a_future_form() -> None:
    for task in TASKS:
        for spec in algorithms_for(task):
            payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
            assert payload["key"] == spec.key and payload["params"]
