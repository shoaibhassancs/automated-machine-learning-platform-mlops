"""Metric tests. Expected values are worked out by hand or with explicit formulas (pairwise AUC,
NumPy), not by calling scikit-learn, which the module itself uses."""

from __future__ import annotations

import json
import math
from itertools import product
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from automl_ops.evaluate import (
    CURVE_MAX_POINTS,
    MAX_CURVE_CLASSES,
    RESIDUAL_MAX_POINTS,
    EvaluationError,
    classification_scalars,
    evaluate_classification,
    evaluate_regression,
    summarize_folds,
)

Array = NDArray[Any]


def arr(values: list[Any]) -> Array:
    return np.array(values, dtype=object if isinstance(values[0], str) else "float64")


def auc_by_pairs(is_positive: Array, score: Array) -> float:
    """ROC area as the probability that a positive outranks a negative (ties count one half)."""
    pos, neg = score[is_positive], score[~is_positive]
    wins = sum((p > n) + 0.5 * (p == n) for p, n in product(pos, neg))
    return float(wins) / (len(pos) * len(neg))


def strict_json(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


# --- binary classification, worked by hand ---------------------------------------------------

Y_TRUE = arr(["a", "a", "a", "b", "b", "b"])
Y_PRED = arr(["a", "a", "b", "b", "b", "a"])
P_B = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4])  # probability of class "b" (the positive class)
PROBA = np.column_stack([1 - P_B, P_B])


def test_binary_metrics_by_hand() -> None:
    result = evaluate_classification(Y_TRUE, Y_PRED, PROBA, ["a", "b"])
    m = result["metrics"]
    # confusion: true a -> (a 2, b 1); true b -> (a 1, b 2). Each class: precision = recall = 2/3.
    assert result["confusion_matrix"] == [[2, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-11)
    for name in ("precision_macro", "recall_macro", "f1_macro"):
        assert m[name] == pytest.approx(2 / 3, abs=1e-11)
    # positives (b) scores 0.7, 0.8, 0.4 versus negatives 0.1, 0.2, 0.6: wins 3 + 3 + 2 = 8 of 9
    assert m["roc_auc"] == pytest.approx(8 / 9, abs=1e-11)
    assert m["roc_auc"] == pytest.approx(auc_by_pairs(Y_TRUE == "b", P_B), abs=1e-11)
    assert result["undefined"] == {} and result["rows"] == 6
    assert [c["support"] for c in result["per_class"]] == [3, 3]


def test_positive_class_is_the_second_class_in_the_given_order() -> None:
    flipped = evaluate_classification(Y_TRUE, Y_PRED, PROBA[:, ::-1], ["b", "a"])
    # with order ["b", "a"] the positive class is "a" and its probability column is P(a)
    assert flipped["classes"] == ["b", "a"]
    assert flipped["confusion_matrix"] == [[2, 1], [1, 2]]  # rows follow the given class order
    assert flipped["metrics"]["roc_auc"] == pytest.approx(
        auc_by_pairs(Y_TRUE == "a", PROBA[:, 0]), abs=1e-11
    )


def test_confusion_matrix_total_equals_the_rows_scored() -> None:
    rng = np.random.default_rng(1)
    classes = ["a", "b", "c", "d"]
    y_true = arr(list(rng.choice(classes, 500)))
    y_pred = arr(list(rng.choice(classes, 500)))
    matrix = evaluate_classification(y_true, y_pred, None, classes, curves=False)[
        "confusion_matrix"
    ]
    assert sum(sum(row) for row in matrix) == 500
    for i, c in enumerate(classes):  # each row sums to that class's true count
        assert sum(matrix[i]) == int((y_true == c).sum())


# --- multiclass --------------------------------------------------------------------------------


def three_class_example() -> tuple[Array, Array, Array, list[str]]:
    classes = ["a", "b", "c"]
    y_true = arr(["a", "a", "b", "b", "c", "c", "a", "b", "c"])
    proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.4, 0.3],
            [0.1, 0.3, 0.6],
            [0.4, 0.2, 0.4],
            [0.2, 0.5, 0.3],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
        ]
    )
    y_pred = arr([classes[i] for i in proba.argmax(axis=1)])
    return y_true, y_pred, proba, classes


def test_multiclass_auc_is_the_one_vs_rest_macro_average() -> None:
    y_true, y_pred, proba, classes = three_class_example()
    expected = np.mean(
        [auc_by_pairs(y_true == c, proba[:, i]) for i, c in enumerate(classes)]
    )  # explicit OvR macro
    scalars, undefined = classification_scalars(y_true, y_pred, proba, classes)
    assert scalars["roc_auc"] == pytest.approx(float(expected), abs=1e-11) and undefined == {}


def test_multiclass_macro_metrics_by_hand() -> None:
    y_true, y_pred, _, classes = three_class_example()
    assert list(y_pred) == ["a", "a", "b", "b", "c", "a", "b", "b", "c"]
    # a: 2 correct of 3 predicted, 3 actual; b: 3 correct of 4 predicted, 3 actual;
    # c: 2 correct of 2 predicted, 3 actual. Seven of the nine rows are correct.
    prec = [2 / 3, 3 / 4, 2 / 2]
    rec = [2 / 3, 3 / 3, 2 / 3]
    f1s = [2 * p * r / (p + r) for p, r in zip(prec, rec, strict=True)]
    scalars, _ = classification_scalars(y_true, y_pred, None, classes)
    assert scalars["precision_macro"] == pytest.approx(float(np.mean(prec)), abs=1e-11)
    assert scalars["recall_macro"] == pytest.approx(float(np.mean(rec)), abs=1e-11)
    assert scalars["f1_macro"] == pytest.approx(float(np.mean(f1s)), abs=1e-11)
    assert scalars["accuracy"] == pytest.approx(7 / 9, abs=1e-11)


def test_macro_averages_cover_only_classes_present_in_the_scored_rows() -> None:
    y_true = arr(["a", "a", "b", "b"])
    y_pred = arr(["a", "b", "b", "b"])
    scalars, undefined = classification_scalars(y_true, y_pred, None, ["a", "b", "c"])
    # class c never occurs: averaged over a and b only (precision a 1, b 2/3 ; recall a 1/2, b 1)
    assert scalars["precision_macro"] == pytest.approx((1 + 2 / 3) / 2, abs=1e-11)
    assert scalars["recall_macro"] == pytest.approx((0.5 + 1) / 2, abs=1e-11)
    assert "roc_auc" in undefined


# --- undefined metrics and invalid input --------------------------------------------------------


def test_undefined_roc_auc_is_none_with_a_reason_and_the_output_is_strict_json() -> None:
    only_a = arr(["a", "a", "a"])
    result = evaluate_classification(only_a, only_a, np.array([[1.0, 0.0]] * 3), ["a", "b"])
    assert result["metrics"]["roc_auc"] is None
    assert "does not occur" in result["undefined"]["roc_auc"]
    assert result["roc_curves"] == [] and result["curve_notes"]  # a curve needs both kinds of rows
    strict_json(result)
    no_proba = evaluate_classification(Y_TRUE, Y_PRED, None, ["a", "b"])
    assert (
        no_proba["metrics"]["roc_auc"] is None
        and "no predicted probabilities" in (no_proba["undefined"]["roc_auc"])
    )
    strict_json(no_proba)


@pytest.mark.parametrize(
    "call",
    [
        lambda: classification_scalars(arr(["a", "z"]), arr(["a", "a"]), None, ["a", "b"]),
        lambda: classification_scalars(arr(["a", "b"]), arr(["a"]), None, ["a", "b"]),
        lambda: classification_scalars(arr(["a"]), arr(["a"]), np.ones((2, 2)), ["a", "b"]),
        lambda: evaluate_regression(arr([1.0]), arr([1.0, 2.0]), arr([0.0])),
        lambda: evaluate_regression(np.array([]), np.array([]), np.array([])),
    ],
)
def test_invalid_inputs_raise_a_clear_error(call: Any) -> None:
    with pytest.raises(EvaluationError):
        call()


# --- curves --------------------------------------------------------------------------------------


def test_binary_roc_and_pr_curves_are_consistent_with_the_auc() -> None:
    result = evaluate_classification(Y_TRUE, Y_PRED, PROBA, ["a", "b"])
    (roc,) = result["roc_curves"]
    (pr,) = result["pr_curves"]
    assert roc["class"] == "b" and (roc["fpr"][0], roc["tpr"][0]) == (0.0, 0.0)
    assert (roc["fpr"][-1], roc["tpr"][-1]) == (1.0, 1.0)
    area = np.trapezoid(roc["tpr"], roc["fpr"])  # few points: nothing was thinned
    assert float(area) == pytest.approx(result["metrics"]["roc_auc"], abs=1e-9)
    assert (pr["recall"][-1], pr["precision"][-1]) == (0.0, 1.0)  # the curve ends at recall 0
    assert pr["recall"][0] == 1.0


def test_large_curves_are_thinned_keeping_the_endpoints() -> None:
    rng = np.random.default_rng(2)
    y = arr(list(rng.choice(["n", "p"], 3000)))
    score = np.clip(rng.normal(0.5, 0.25, 3000) + (y == "p") * 0.2, 0, 1)
    result = evaluate_classification(y, y, np.column_stack([1 - score, score]), ["n", "p"])
    (roc,) = result["roc_curves"]
    assert len(roc["fpr"]) <= CURVE_MAX_POINTS
    assert (roc["fpr"][0], roc["tpr"][0], roc["fpr"][-1], roc["tpr"][-1]) == (0.0, 0.0, 1.0, 1.0)
    assert roc["fpr"] == sorted(roc["fpr"])


def test_multiclass_curves_are_one_vs_rest_and_limited_to_the_largest_classes() -> None:
    y_true, y_pred, proba, classes = three_class_example()
    result = evaluate_classification(y_true, y_pred, proba, classes)
    assert [c["class"] for c in result["roc_curves"]] == ["a", "b", "c"]
    assert result["curve_notes"] == []
    many = [f"c{i:02d}" for i in range(12)]
    rng = np.random.default_rng(3)
    y = arr([many[i % 12] for i in range(240)])
    proba12 = rng.dirichlet(np.ones(12), 240)
    big = evaluate_classification(y, y, proba12, many)
    assert len(big["roc_curves"]) == MAX_CURVE_CLASSES
    assert any("10 largest of 12" in note for note in big["curve_notes"])
    strict_json(big)


# --- regression ----------------------------------------------------------------------------------


def test_regression_metrics_by_hand() -> None:
    y = arr([3.0, -0.5, 2.0, 7.0])
    pred = arr([2.5, 0.0, 2.0, 8.0])
    result = evaluate_regression(y, pred, arr([10.0, 11.0, 12.0, 13.0]))
    # errors 0.5, -0.5, 0, -1: mse 1.5/4, rmse sqrt(0.375), mae 2/4;
    # SS_tot 29.1875, so r2 = 1 - 1.5/29.1875
    assert result["metrics"]["rmse"] == pytest.approx(math.sqrt(0.375), abs=1e-11)
    assert result["metrics"]["mae"] == pytest.approx(0.5, abs=1e-11)
    assert result["metrics"]["r2"] == pytest.approx(1 - 1.5 / 29.1875, abs=1e-11)
    assert [p["residual"] for p in result["residual_points"]] == [
        0.5,
        -0.5,
        0.0,
        -1.0,
    ]  # actual - predicted
    assert [p["row"] for p in result["residual_points"]] == [10, 11, 12, 13]
    assert result["residual_points_note"] is None


def test_residuals_stay_attached_to_the_right_observation_when_thinned() -> None:
    rng = np.random.default_rng(4)
    n = 5 * RESIDUAL_MAX_POINTS
    actual = rng.normal(size=n)
    predicted = actual + rng.normal(scale=0.1, size=n)
    rows = np.arange(n)[::-1] * 3.0 + 100  # arbitrary, unsorted row ids
    result = evaluate_regression(actual, predicted, rows)
    points = result["residual_points"]
    assert len(points) <= RESIDUAL_MAX_POINTS and "evenly spaced" in result["residual_points_note"]
    position = {float(r): i for i, r in enumerate(rows)}
    for p in points:
        i = position[float(p["row"])]
        assert p["actual"] == pytest.approx(actual[i], rel=1e-10, abs=1e-12)
        assert p["residual"] == pytest.approx(actual[i] - predicted[i], rel=1e-10, abs=1e-12)


def test_r2_is_undefined_for_constant_actual_values() -> None:
    result = evaluate_regression(arr([2.0, 2.0, 2.0]), arr([1.0, 2.0, 3.0]), arr([0.0, 1.0, 2.0]))
    assert result["metrics"]["r2"] is None and "all equal" in result["undefined"]["r2"]
    assert result["metrics"]["rmse"] is not None
    strict_json(result)


# --- summaries across folds ---------------------------------------------------------------


def test_fold_summary_is_mean_and_sample_std_over_defined_folds() -> None:
    folds: list[dict[str, float | None]] = [{"m": 1.0}, {"m": 3.0}, {"m": None}, {"m": 5.0}]
    summary = summarize_folds(folds, ["m"])["m"]
    assert summary["mean"] == pytest.approx(3.0) and summary["std"] == pytest.approx(2.0)
    assert (summary["folds_defined"], summary["folds"]) == (3, 4)
    one = summarize_folds([{"m": 4.0}], ["m"])["m"]
    assert one["mean"] == 4.0 and one["std"] is None
    none = summarize_folds([{"m": None}], ["m"])["m"]
    assert none["mean"] is None and none["folds_defined"] == 0
