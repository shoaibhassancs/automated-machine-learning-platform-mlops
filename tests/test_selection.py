"""Model selection from cross-validation results only (BUILD_PLAN Step 11).

The training results here are written by hand, so every expected winner follows from numbers that
can be read in the test itself.
"""

from __future__ import annotations

import dataclasses
import inspect
import random
from typing import Any

import numpy as np
import pytest

from automl_ops.pipeline import ColumnPlan
from automl_ops.selection import (
    TIE_TOLERANCE,
    FrozenSelection,
    SelectionError,
    freeze_selection,
    select_model,
)
from automl_ops.train import AlgorithmResult, Split, TrainingResult, TrainSettings

CLASSES = ("neg", "pos")
LR, RF, HGB = "logistic_regression", "random_forest", "hist_gradient_boosting"


def entry(mean: float | None, std: float | None = 0.01, defined: int = 3) -> dict[str, Any]:
    return {"mean": mean, "std": std, "folds_defined": defined, "folds": 3}


def clf(
    key: str,
    f1: float | None,
    *,
    oof_f1: float | None = None,
    accuracy: float = 0.8,
    std: float | None = 0.01,
    roc_defined: int = 3,
    status: str = "ok",
) -> AlgorithmResult:
    if status != "ok":
        return AlgorithmResult(
            key,
            key,
            {},
            "failed",
            {"type": "Boom", "message": "x"},
            0.1,
            (),
            (),
            {},
            None,
            None,
            None,
        )
    summary = {
        "f1_macro": entry(f1, std),
        "accuracy": entry(accuracy, 0.01),
        "roc_auc": entry(0.9 if roc_defined else None, 0.01, roc_defined),
    }
    metrics = {
        "f1_macro": f1 if oof_f1 is None else oof_f1,
        "accuracy": accuracy,
        "roc_auc": 0.9 if roc_defined == 3 else None,
    }
    oof = {"metrics": metrics, "undefined": {"roc_auc": "class missing"}}
    return AlgorithmResult(
        key, key, {"p": key}, "ok", None, 0.1, (), (), summary, oof, np.array(["neg"]), None
    )


def reg(
    key: str, rmse: float, *, oof_rmse: float | None = None, mae: float = 1.0
) -> AlgorithmResult:
    summary = {"rmse": entry(rmse), "mae": entry(mae), "r2": entry(0.5)}
    metrics = {"rmse": rmse if oof_rmse is None else oof_rmse, "mae": mae, "r2": 0.5}
    return AlgorithmResult(
        key,
        key,
        {},
        "ok",
        None,
        0.1,
        (),
        (),
        summary,
        {"metrics": metrics, "undefined": {}},
        np.array([0.0]),
        None,
    )


def training(results: list[AlgorithmResult], task: str = "classification") -> TrainingResult:
    split = Split(np.arange(8), np.arange(8, 10), True)
    return TrainingResult(
        task=task,  # type: ignore[arg-type]
        target="y",
        settings=TrainSettings(),
        split=split,
        folds_used=3,
        warnings=(),
        plan=ColumnPlan(("a",), (), ()),
        classes=CLASSES if task == "classification" else None,
        train_class_counts={"neg": 4, "pos": 4} if task == "classification" else None,
        results=tuple(results),
        seconds=1.0,
    )


# --- ranking ---------------------------------------------------------------------------------


def test_the_best_fold_mean_of_macro_f1_wins_and_the_report_says_why() -> None:
    report = select_model(training([clf(LR, 0.70), clf(RF, 0.80), clf(HGB, 0.75)])).to_dict()
    assert report["metric"] == "f1_macro" and report["direction"] == "higher"
    assert [r["algorithm"] for r in report["ranking"]] == [RF, HGB, LR]
    assert report["selected"]["algorithm"] == RF
    assert report["margin"]["runner_up"] == HGB
    assert report["margin"]["difference"] == pytest.approx(0.05)
    assert (
        "best mean f1_macro" in report["explanation"]
        and "cross-validation" in report["explanation"]
    )
    assert report["positive_class"] == "pos"  # the second of the sorted labels
    assert report["classes"] == ["neg", "pos"]


def test_regression_defaults_to_rmse_where_lower_is_better() -> None:
    report = select_model(
        training(
            [reg("ridge", 3.0), reg("random_forest", 2.0), reg("hist_gradient_boosting", 2.5)],
            "regression",
        )
    ).to_dict()
    assert report["metric"] == "rmse" and report["direction"] == "lower"
    assert report["selected"]["algorithm"] == "random_forest"
    assert report["positive_class"] is None and report["classes"] is None


def test_a_metric_override_from_the_closed_list_changes_the_winner() -> None:
    results = [clf(LR, 0.70, accuracy=0.9), clf(RF, 0.80, accuracy=0.7)]
    assert select_model(training(results)).selected_key == RF
    assert select_model(training(results), "accuracy").selected_key == LR
    with pytest.raises(SelectionError, match="not a classification selection metric"):
        select_model(training(results), "rmse")
    with pytest.raises(SelectionError, match="not a regression selection metric"):
        select_model(training([reg("ridge", 1.0)], "regression"), "f1_macro")


# --- ties --------------------------------------------------------------------------------------


def test_a_difference_just_inside_the_tolerance_is_a_tie_and_just_outside_is_not() -> None:
    inside = select_model(
        training([clf(LR, 0.8, oof_f1=0.9), clf(RF, 0.8 + TIE_TOLERANCE / 2, oof_f1=0.7)])
    )
    assert inside.selected_key == LR  # tied on the fold mean, so the pooled value decides
    assert inside.to_dict()["ranking"][0]["decided_by"] == "tie-break: pooled_oof"
    outside = select_model(
        training([clf(LR, 0.8, oof_f1=0.9), clf(RF, 0.8 + 3 * TIE_TOLERANCE, oof_f1=0.7)])
    )
    assert outside.selected_key == RF
    assert outside.to_dict()["ranking"][0]["decided_by"] == "fold_mean"


def test_tie_break_order_is_pooled_oof_then_secondary_then_fold_std_then_catalogue_order() -> None:
    def winner(a: AlgorithmResult, b: AlgorithmResult) -> tuple[str, str]:
        report = select_model(training([a, b])).to_dict()
        return report["selected"]["algorithm"], report["ranking"][0]["decided_by"]

    # 1. pooled out-of-fold value (RF's is higher although every later criterion favours LR)
    assert winner(
        clf(LR, 0.8, oof_f1=0.80, accuracy=0.9, std=0.001),
        clf(RF, 0.8, oof_f1=0.85, accuracy=0.5, std=0.1),
    ) == (RF, "tie-break: pooled_oof")
    # 2. secondary metric (accuracy)
    assert winner(clf(LR, 0.8, accuracy=0.7, std=0.001), clf(RF, 0.8, accuracy=0.9, std=0.1)) == (
        RF,
        "tie-break: secondary_metric",
    )
    # 3. smaller fold standard deviation
    assert winner(clf(LR, 0.8, std=0.05), clf(RF, 0.8, std=0.01)) == (RF, "tie-break: fold_std")
    # 4. fixed catalogue order, whatever order the results come in
    assert winner(clf(RF, 0.8), clf(LR, 0.8)) == (LR, "tie-break: catalogue_order")
    assert winner(clf(LR, 0.8), clf(RF, 0.8)) == (LR, "tie-break: catalogue_order")


def test_selection_is_the_same_for_every_input_order() -> None:
    results = [clf(LR, 0.8), clf(RF, 0.8), clf(HGB, 0.8, std=0.2)]
    expected = select_model(training(results)).to_dict()
    shuffler = random.Random(3)
    for _ in range(10):
        shuffled = list(results)
        shuffler.shuffle(shuffled)
        assert select_model(training(shuffled)).to_dict() == expected


# --- failures and incomplete results ----------------------------------------------------------


def test_failed_and_undefined_candidates_are_excluded_with_reasons() -> None:
    results = [
        clf(LR, 0.95, status="failed"),
        clf(RF, 0.80),
        clf(HGB, 0.99, roc_defined=1),  # roc_auc is undefined in 2 of 3 folds
    ]
    report = select_model(training(results), "roc_auc").to_dict()
    assert [r["algorithm"] for r in report["ranking"]] == [RF]
    reasons = {e["algorithm"]: e["reason"] for e in report["excluded"]}
    assert "failed" in reasons[LR] and "undefined in 2 of 3 folds" in reasons[HGB]
    assert any("only algorithm that completed" in w for w in report["warnings"])


def test_a_candidate_undefined_on_the_pooled_rows_is_excluded() -> None:
    odd = clf(RF, 0.9)
    assert odd.oof is not None
    odd.oof["metrics"]["f1_macro"] = None
    report = select_model(training([odd, clf(LR, 0.5)])).to_dict()
    assert report["selected"]["algorithm"] == LR
    assert "pooled out-of-fold" in report["excluded"][0]["reason"]


def test_nothing_to_select_is_an_error_naming_the_reasons() -> None:
    with pytest.raises(SelectionError, match="no algorithm can be selected.*logistic_regression"):
        select_model(training([clf(LR, 0.1, status="failed")]))


def test_a_close_race_is_reported_as_not_a_clear_winner() -> None:
    report = select_model(training([clf(LR, 0.800, std=0.05), clf(RF, 0.790, std=0.05)])).to_dict()
    assert report["margin"]["within_fold_std"] is True
    assert any("not a clear winner" in w for w in report["warnings"])
    clear = select_model(training([clf(LR, 0.90, std=0.01), clf(RF, 0.70, std=0.01)])).to_dict()
    assert clear["margin"]["within_fold_std"] is False and not clear["warnings"]


def test_disagreement_between_fold_means_and_pooled_values_is_reported() -> None:
    report = select_model(
        training([clf(LR, 0.80, oof_f1=0.70), clf(RF, 0.79, oof_f1=0.90)])
    ).to_dict()
    assert report["selected"]["algorithm"] == LR and report["rankings_agree"] is False
    assert any("pooled" in w for w in report["warnings"])


# --- what the selection can see, and the frozen record --------------------------------------


def test_selection_takes_only_the_training_result_and_never_a_data_frame() -> None:
    assert list(inspect.signature(select_model).parameters) == ["training", "metric"]
    assert list(inspect.signature(freeze_selection).parameters) == ["training", "report"]


def test_freezing_records_the_configuration_and_a_verifiable_hash() -> None:
    run = training([clf(LR, 0.7), clf(RF, 0.8)])
    frozen = freeze_selection(run, select_model(run))
    frozen.verify()
    data = frozen.data
    assert data["algorithm"] == RF and data["params"] == {"p": RF} and data["seed"] == 42
    assert data["positive_class"] == "pos" and data["classes"] == ["neg", "pos"]
    assert data["plan"]["numeric"] == ["a"] and data["split"]["test_sha256"]
    assert data["selection"]["metric"] == "f1_macro"
    again = freeze_selection(run, select_model(run))
    assert again == frozen  # the same input gives the same bytes and hash


def test_a_changed_frozen_selection_fails_verification() -> None:
    run = training([clf(LR, 0.7), clf(RF, 0.8)])
    frozen = freeze_selection(run, select_model(run))
    forged = FrozenSelection(frozen.payload.replace(RF, LR, 1), frozen.sha256)
    with pytest.raises(SelectionError, match="SHA-256"):
        forged.verify()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.sha256 = "0" * 64  # type: ignore[misc]
