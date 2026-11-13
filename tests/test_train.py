"""Split, fold and training-runner tests on small invented data.

The important properties are tested directly: preprocessing is fitted inside each fold from that
fold's training rows only; out-of-fold predictions cover the training part exactly once; the
held-out test part never influences anything; small classes adjust or stop the run cleanly.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import automl_ops.train as train_module
from automl_ops.algorithms import AlgorithmError
from automl_ops.pipeline import build_column_plan, build_pipeline
from automl_ops.profile import profile_frame, read_text_frame
from automl_ops.task import TaskConfig
from automl_ops.train import (
    AlgorithmRequest,
    FoldFit,
    ProgressEvent,
    Split,
    TrainingError,
    TrainSettings,
    fit_fold,
    make_folds,
    make_split,
    plan_folds,
    run_training,
)

Array = NDArray[Any]

FAST_CLASSIFIERS = [
    AlgorithmRequest("logistic_regression"),
    AlgorithmRequest("random_forest", {"n_estimators": 20}),
    AlgorithmRequest("hist_gradient_boosting", {"max_iter": 20}),
]
FAST_REGRESSORS = [
    AlgorithmRequest("ridge"),
    AlgorithmRequest("random_forest", {"n_estimators": 20}),
    AlgorithmRequest("hist_gradient_boosting", {"max_iter": 20}),
]


def frame_of(columns: dict[str, list[str]]) -> pd.DataFrame:
    names = list(columns)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    for i in range(len(columns[names[0]])):
        writer.writerow([columns[name][i] for name in names])
    return read_text_frame(out.getvalue().encode("utf-8"))


def classification_frame(n: int = 300, n_classes: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    cuts = np.quantile(x1, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.digitize(x1 + rng.normal(scale=0.5, size=n), cuts)
    x2 = [f"{v:.4f}" if i % 17 else "" for i, v in enumerate(rng.normal(size=n))]
    cat = [
        ("red", "green", "blue")[int(v)] if i % 23 else ""
        for i, v in enumerate(rng.integers(0, 3, n))
    ]
    return frame_of(
        {
            "id": [str(i) for i in range(n)],
            "x1": [f"{v:.4f}" for v in x1],
            "x2": x2,
            "cat": cat,
            "y": [str("abcdefgh"[k]) for k in y],
        }
    )


def labelled_frame(sizes: dict[str, int], seed: int = 0) -> pd.DataFrame:
    """A frame whose target classes have exactly the given sizes (features are random noise)."""
    rng = np.random.default_rng(seed)
    labels = [k for k, n in sizes.items() for _ in range(n)]
    n = len(labels)
    order = rng.permutation(n)
    return frame_of(
        {
            "id": [str(i) for i in range(n)],
            "x1": [f"{v:.4f}" for v in rng.normal(size=n)],
            "cat": [("red", "green", "blue")[int(v)] for v in rng.integers(0, 3, n)],
            "y": [labels[i] for i in order],
        }
    )


def regression_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    cat = rng.integers(0, 3, n)
    y = 3 * x1 + cat + rng.normal(scale=0.3, size=n)
    return frame_of(
        {
            "id": [str(i) for i in range(n)],
            "x1": [f"{v:.4f}" for v in x1],
            "cat": [("p", "q", "r")[int(c)] for c in cat],
            "y": [f"{v:.4f}" for v in y],
        }
    )


def labels_of(sizes: dict[str, int]) -> Array:
    return np.array([k for k, n in sizes.items() for _ in range(n)], dtype=object)


def strip_times(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_times(v) for k, v in value.items() if k not in ("seconds", "fold_seconds")}
    if isinstance(value, list):
        return [strip_times(v) for v in value]
    return value


CLF = TaskConfig("y", "classification")
REG = TaskConfig("y", "regression")


# --- settings ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        TrainSettings(folds=1),
        TrainSettings(folds=11),
        TrainSettings(test_fraction=0.04),
        TrainSettings(test_fraction=0.51),
        TrainSettings(seed=-1),
        TrainSettings(folds=True),
    ],
)
def test_invalid_settings_are_rejected(bad: TrainSettings) -> None:
    with pytest.raises(TrainingError):
        bad.validate()


def test_defaults_are_the_approved_ones() -> None:
    assert TrainSettings().to_dict() == {"test_fraction": 0.2, "seed": 42, "folds": 5}


# --- the holdout split ---------------------------------------------------------------------------


def test_classification_split_is_stratified_seeded_and_disjoint() -> None:
    y = labels_of({"a": 100, "b": 100, "c": 100})
    split = make_split(y, "classification", TrainSettings())
    assert (len(split.train_rows), len(split.test_rows)) == (240, 60) and split.stratified
    assert set(split.train_rows).isdisjoint(split.test_rows)
    assert sorted(set(split.train_rows) | set(split.test_rows)) == list(range(300))
    assert list(split.train_rows) == sorted(split.train_rows)
    for label in "abc":  # stratified: each class contributes 20% to the test part
        assert int((y[split.test_rows] == label).sum()) == 20
    again = make_split(y, "classification", TrainSettings())
    assert np.array_equal(split.train_rows, again.train_rows)
    other = make_split(y, "classification", TrainSettings(seed=7))
    assert not np.array_equal(split.test_rows, other.test_rows)


def test_regression_split_is_shuffled_not_stratified() -> None:
    values = np.arange(100, dtype="float64")
    split = make_split(values, "regression", TrainSettings())
    assert (len(split.train_rows), len(split.test_rows)) == (80, 20) and not split.stratified
    assert not np.array_equal(split.test_rows, np.arange(80, 100))  # not simply the last rows


@pytest.mark.parametrize(
    ("sizes", "match"),
    [
        ({"a": 50, "b": 1}, "fewer than 2 rows"),
        ({f"c{i:02d}": 2 for i in range(40)}, "cannot hold all 40 classes"),
        (
            {"a": 50, "b": 50, "c": 2},
            "fewer than 2 training rows",
        ),  # 2 rows: 1 lands in the test part
        ({"a": 20, "b": 20, "c": 2}, "fewer than 2 training rows"),
    ],
)
def test_infeasible_classification_splits_stop_with_an_actionable_error(
    sizes: dict[str, int], match: str
) -> None:
    with pytest.raises(TrainingError, match=match):
        make_split(labels_of(sizes), "classification", TrainSettings())


def test_a_class_with_three_rows_leaves_two_for_training() -> None:
    y = labels_of({"a": 50, "b": 50, "c": 3})
    split = make_split(y, "classification", TrainSettings())
    assert int((y[split.train_rows] == "c").sum()) == 2


def test_too_few_rows_for_the_test_fraction_is_an_error() -> None:
    with pytest.raises(TrainingError, match="too few"):
        make_split(np.arange(1, dtype="float64"), "regression", TrainSettings())


# --- folds ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smallest", "folds", "warns"),
    [(6, 5, False), (5, 5, False), (4, 4, True), (3, 3, True), (2, 2, True)],
)
def test_folds_are_reduced_to_the_smallest_class_with_a_warning(
    smallest: int, folds: int, warns: bool
) -> None:
    used, warnings = plan_folds(labels_of({"a": 40, "b": smallest}), "classification", 5)
    assert used == folds and bool(warnings) == warns
    if warns:
        assert f"from 5 to {folds}" in warnings[0] and "'b'" in warnings[0]


def test_a_class_with_one_training_row_cannot_be_cross_validated() -> None:
    with pytest.raises(TrainingError, match="needs at least 2"):
        plan_folds(labels_of({"a": 40, "b": 1}), "classification", 5)


def test_regression_needs_at_least_as_many_rows_as_folds() -> None:
    assert plan_folds(np.arange(5.0), "regression", 5) == (5, ())
    with pytest.raises(TrainingError, match="too few"):
        plan_folds(np.arange(4.0), "regression", 5)


def test_folds_partition_the_training_part_and_stratified_folds_keep_every_class() -> None:
    y = labels_of({"a": 30, "b": 30, "c": 10})
    folds = make_folds(y, "classification", 5, seed=42)
    validation = np.concatenate([val for _, val in folds])
    assert sorted(validation) == list(range(70))  # every row validated exactly once
    for fit, val in folds:
        assert set(fit).isdisjoint(val) and len(fit) + len(val) == 70
        assert set(y[val]) == {"a", "b", "c"} and set(y[fit]) == {"a", "b", "c"}
    again = make_folds(y, "classification", 5, seed=42)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again, strict=True))
    reg = make_folds(np.arange(50.0), "regression", 5, seed=42)
    assert sorted(np.concatenate([v for _, v in reg])) == list(range(50))


# --- preprocessing is fitted inside each fold ----------------------------------------------------


def test_fit_fold_learns_preprocessing_from_the_fit_rows_only() -> None:
    frame = frame_of(
        {
            "a": ["1", "2", "3", "", "5", "1000", "2000", "3000"],
            "c": ["x", "x", "y", "y", "x", "val-only", "val-only", "z"],
            "y": ["p", "q", "p", "q", "p", "q", "p", "q"],
        }
    )
    plan = build_column_plan(profile_frame(frame), CLF)
    template = build_pipeline(plan, LogisticRegression(), scale_numeric=True)
    fit_rows, val_rows = np.arange(5), np.arange(5, 8)
    x = frame[["a", "c"]]
    y = frame["y"].astype(str).to_numpy(dtype=object)
    fitted = fit_fold(template, x.iloc[fit_rows], y[fit_rows], x.iloc[val_rows], ("p", "q"))
    numeric = fitted.pipeline.named_steps["prep"].named_transformers_["numeric"]
    encoder = (
        fitted.pipeline.named_steps["prep"].named_transformers_["categorical"].named_steps["encode"]
    )
    assert numeric.named_steps["impute"].statistics_[0] == np.median([1.0, 2.0, 3.0, 5.0])
    assert list(encoder.categories_[0]) == ["x", "y"]
    assert fitted.proba is not None and fitted.proba.shape == (3, 2)
    # the template itself is never fitted: each fold gets a fresh clone
    assert not hasattr(template.named_steps["prep"], "transformers_")


def test_the_runner_fits_preprocessing_separately_in_every_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = classification_frame(n=240)
    calls: list[tuple[pd.DataFrame, pd.DataFrame, FoldFit]] = []
    original = train_module.fit_fold

    def spy(
        template: Any, x_fit: pd.DataFrame, y_fit: Array, x_val: pd.DataFrame, classes: Any
    ) -> FoldFit:
        fitted = original(template, x_fit, y_fit, x_val, classes)
        calls.append((x_fit, x_val, fitted))
        return fitted

    monkeypatch.setattr(train_module, "fit_fold", spy)
    result = run_training(
        frame, CLF, [AlgorithmRequest("logistic_regression")], TrainSettings(folds=5)
    )
    assert result.ok and len(calls) == 5
    global_median = float(np.median(frame["x1"].astype(float)))
    differs = 0
    for x_fit, x_val, fitted in calls:
        assert set(x_fit.index).isdisjoint(x_val.index)  # validation rows are not fit rows
        prep = fitted.pipeline.named_steps["prep"]
        numeric = prep.named_transformers_["numeric"]
        names = list(prep.transformers_[0][2])
        for j, name in enumerate(names):
            values = x_fit[name].dropna().astype(float).to_numpy()
            imputed = np.where(pd.isna(x_fit[name]), np.median(values), x_fit[name].astype(float))
            assert numeric.named_steps["impute"].statistics_[j] == pytest.approx(np.median(values))
            assert numeric.named_steps["scale"].mean_[j] == pytest.approx(imputed.mean())
            assert numeric.named_steps["scale"].scale_[j] == pytest.approx(imputed.std())
        if numeric.named_steps["impute"].statistics_[names.index("x1")] != global_median:
            differs += 1
    assert differs >= 1  # sensitivity: a global fit would have given different numbers


def test_categories_only_in_the_validation_rows_are_not_learned_but_still_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 120
    base = classification_frame(n=n)
    cat = base["cat"].astype(object).tolist()
    cat[7] = "rare-one"  # appears in exactly one row: in whichever fold validates it, it is unseen
    frame = base.assign(cat=pd.array(cat, dtype="string"))
    seen: list[Any] = []
    original = train_module.fit_fold

    def spy(*args: Any) -> FoldFit:
        fitted = original(*args)
        seen.append((args[1].index, args[2], fitted))
        return fitted

    monkeypatch.setattr(train_module, "fit_fold", spy)
    result = run_training(frame, CLF, [AlgorithmRequest("logistic_regression")])
    assert result.ok
    learned = [
        "rare-one"
        in fitted.pipeline.named_steps["prep"]
        .named_transformers_["categorical"]
        .named_steps["encode"]
        .categories_[0]
        for _, _, fitted in seen
    ]
    assert sorted(learned) == [
        False,
        True,
        True,
        True,
        True,
    ]  # not learned in the fold that held it out


# --- out-of-fold predictions and results ---------------------------------------------------------


def test_out_of_fold_predictions_cover_the_training_part_exactly_once() -> None:
    frame = classification_frame(n=300)
    result = run_training(frame, CLF, FAST_CLASSIFIERS)
    assert result.ok and result.folds_used == 5 and result.warnings == ()
    n_train = len(result.split.train_rows)
    for r in result.results:
        assert r.oof_predicted is not None and r.oof_proba is not None and r.oof is not None
        assert len(r.oof_predicted) == n_train and r.oof["rows"] == n_train
        assert not pd.isna(pd.Series(r.oof_predicted)).any()  # every training row was predicted
        assert not np.isnan(r.oof_proba).any() and np.allclose(r.oof_proba.sum(axis=1), 1.0)
        assert sum(sum(row) for row in r.oof["confusion_matrix"]) == n_train
        assert len(r.fold_metrics) == 5
        assert r.oof["classes"] == ["a", "b", "c"]
        counts = result.train_class_counts
        assert counts is not None
        assert [c["support"] for c in r.oof["per_class"]] == [counts[c] for c in "abc"]
        assert (
            r.summary["accuracy"]["mean"] is not None and r.summary["roc_auc"]["mean"] is not None
        )


def test_regression_residuals_point_at_the_right_rows_of_the_original_data() -> None:
    frame = regression_frame(n=200)
    result = run_training(frame, REG, FAST_REGRESSORS)
    assert result.ok
    actual = frame["y"].astype(float).to_numpy()
    train_rows = set(result.split.train_rows.tolist())
    for r in result.results:
        assert r.oof is not None and r.oof["rows"] == len(train_rows)
        for point in r.oof["residual_points"]:
            assert point["row"] in train_rows  # a training row, never a test row
            assert point["actual"] == pytest.approx(actual[point["row"]], rel=1e-9)
            assert point["residual"] == pytest.approx(
                point["actual"] - point["predicted"], abs=1e-9
            )
        assert r.summary["rmse"]["mean"] is not None and r.summary["r2"]["mean"] > 0.8


def test_defaults_run_for_both_tasks_and_results_are_strict_json() -> None:
    for frame, config in ((classification_frame(240), CLF), (regression_frame(160), REG)):
        result = run_training(frame, config, FAST_CLASSIFIERS if config is CLF else FAST_REGRESSORS)
        payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert payload["train_format"] == 1 and len(payload["results"]) == 3
        assert payload["split"]["test_rows"] == len(result.split.test_rows)
        assert {r["status"] for r in payload["results"]} == {"ok"}


def test_every_default_parameter_set_trains() -> None:
    result = run_training(
        classification_frame(150),
        CLF,
        [
            AlgorithmRequest(k)
            for k in ("logistic_regression", "random_forest", "hist_gradient_boosting")
        ],
        TrainSettings(folds=3),
    )
    assert result.ok


# --- reproducibility -----------------------------------------------------------------------------


def test_same_data_configuration_and_seed_give_identical_results() -> None:
    frame = classification_frame(240)
    first = run_training(frame, CLF, FAST_CLASSIFIERS)
    second = run_training(frame, CLF, FAST_CLASSIFIERS)
    assert strip_times(first.to_dict()) == strip_times(second.to_dict())
    proba_a, proba_b = first.results[2].oof_proba, second.results[2].oof_proba
    assert proba_a is not None and proba_b is not None and np.array_equal(proba_a, proba_b)
    reg = regression_frame(160)
    assert strip_times(run_training(reg, REG, FAST_REGRESSORS).to_dict()) == strip_times(
        run_training(reg, REG, FAST_REGRESSORS).to_dict()
    )


def test_a_different_seed_changes_the_split_and_the_results() -> None:
    frame = classification_frame(240)
    a = run_training(frame, CLF, FAST_CLASSIFIERS[:1], TrainSettings(seed=1))
    b = run_training(frame, CLF, FAST_CLASSIFIERS[:1], TrainSettings(seed=2))
    assert not np.array_equal(a.split.test_rows, b.split.test_rows)
    assert strip_times(a.to_dict()) != strip_times(b.to_dict())


# --- small classes -------------------------------------------------------------------------------


def test_a_small_class_reduces_the_folds_and_warns() -> None:
    frame = labelled_frame({"a": 50, "b": 50, "c": 4})  # 4 rows: one goes to the test part
    result = run_training(frame, CLF, FAST_CLASSIFIERS[:1])
    counts = result.train_class_counts
    assert counts is not None and counts["c"] == 3
    assert result.folds_used == 3
    assert any("folds reduced from 5 to 3" in w and "'c'" in w for w in result.warnings)
    assert result.ok and len(result.results[0].fold_metrics) == 3


@pytest.mark.parametrize(("rows", "folds"), [(3, 2), (4, 3), (5, 4)])
def test_two_to_four_training_rows_in_the_smallest_class_set_the_fold_count(
    rows: int, folds: int
) -> None:
    result = run_training(labelled_frame({"a": 50, "b": 50, "c": rows}), CLF, FAST_CLASSIFIERS[:1])
    assert result.folds_used == folds and result.warnings and result.ok


def test_five_or_more_rows_in_the_smallest_class_keep_five_folds() -> None:
    result = run_training(labelled_frame({"a": 50, "b": 50, "c": 6}), CLF, FAST_CLASSIFIERS[:1])
    assert result.folds_used == 5 and result.warnings == ()


def test_a_class_that_cannot_be_split_or_cross_validated_stops_before_training() -> None:
    events: list[ProgressEvent] = []
    with pytest.raises(TrainingError, match="fewer than 2 training rows"):
        run_training(
            labelled_frame({"a": 50, "b": 50, "c": 2}),
            CLF,
            FAST_CLASSIFIERS,
            progress=events.append,
        )
    assert events == []  # nothing was trained


# --- failures are isolated, progress is reported -------------------------------------------------


def test_a_failing_algorithm_is_reported_and_the_others_still_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = train_module._make_estimator

    def make(spec: Any, params: Any, seed: int) -> Any:
        if spec.key == "random_forest":
            raise RuntimeError("deliberate failure while building")
        return original(spec, params, seed)

    monkeypatch.setattr(train_module, "_make_estimator", make)
    events: list[ProgressEvent] = []
    result = run_training(classification_frame(240), CLF, FAST_CLASSIFIERS, progress=events.append)
    assert not result.ok
    by_key = {r.key: r for r in result.results}
    assert by_key["random_forest"].status == "failed"
    assert by_key["random_forest"].error == {
        "type": "RuntimeError",
        "message": "deliberate failure while building",
    }
    assert (
        by_key["logistic_regression"].status == "ok"
        and by_key["hist_gradient_boosting"].status == "ok"
    )
    failed = [e for e in events if e.kind == "algorithm_failed"]
    assert [e.algorithm for e in failed] == ["random_forest"]
    json.dumps(result.to_dict(), allow_nan=False)


def test_a_failure_during_fitting_is_also_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    original = train_module._make_estimator

    def make(spec: Any, params: Any, seed: int) -> Any:
        if spec.key == "random_forest":
            return RandomForestClassifier(n_estimators=0)  # rejected by scikit-learn at fit time
        return original(spec, params, seed)

    monkeypatch.setattr(train_module, "_make_estimator", make)
    result = run_training(classification_frame(240), CLF, FAST_CLASSIFIERS)
    by_key = {r.key: r for r in result.results}
    assert by_key["random_forest"].status == "failed" and by_key["random_forest"].error is not None
    assert by_key["random_forest"].oof is None and by_key["random_forest"].summary == {}
    assert by_key["hist_gradient_boosting"].status == "ok"


def test_progress_events_follow_the_order_algorithms_then_folds() -> None:
    events: list[ProgressEvent] = []
    run_training(classification_frame(240), CLF, FAST_CLASSIFIERS, progress=events.append)
    kinds = [(e.algorithm, e.kind, e.fold) for e in events]
    expected: list[tuple[str, str, int | None]] = []
    for key in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        expected.append((key, "algorithm_started", None))
        expected += [(key, "fold_finished", k) for k in range(1, 6)]
        expected.append((key, "algorithm_finished", None))
    assert kinds == expected  # sequential: one algorithm completes before the next starts
    assert {(e.algorithm_index, e.algorithm_count) for e in events} == {(1, 3), (2, 3), (3, 3)}
    assert all(e.folds == 5 for e in events)
    assert events[1].to_dict()["fold"] == 1 and events[1].to_dict()["seconds"] is not None


def test_time_per_algorithm_and_per_fold_is_recorded() -> None:
    result = run_training(classification_frame(240), CLF, FAST_CLASSIFIERS[:1])
    r = result.results[0]
    assert r.seconds > 0 and len(r.fold_seconds) == 5 and result.seconds >= r.seconds


# --- request validation happens before anything is trained ---------------------------------------


def test_bad_requests_are_rejected_up_front() -> None:
    frame = classification_frame(120)
    with pytest.raises(TrainingError, match="no algorithms"):
        run_training(frame, CLF, [])
    with pytest.raises(TrainingError, match="more than once"):
        run_training(frame, CLF, [AlgorithmRequest("ridge"), AlgorithmRequest("ridge")])
    with pytest.raises(AlgorithmError, match="unknown"):
        run_training(frame, CLF, [AlgorithmRequest("svm")])
    with pytest.raises(AlgorithmError, match="n_estimators"):
        run_training(frame, CLF, [AlgorithmRequest("random_forest", {"n_estimators": 10_000})])
    with pytest.raises(TrainingError, match="not a column"):
        run_training(frame, TaskConfig("nope", "classification"), FAST_CLASSIFIERS)
    with pytest.raises(TrainingError, match="not usable"):
        run_training(
            frame, TaskConfig("y", "classification", ("id", "x1", "x2", "cat")), FAST_CLASSIFIERS
        )
    with pytest.raises(TrainingError, match="not usable.*target_not_numeric"):
        run_training(frame, TaskConfig("y", "regression"), FAST_REGRESSORS)


def test_columns_that_are_left_out_are_reported_with_reasons_and_training_still_works() -> None:
    n = 200
    base = classification_frame(n=n)
    frame = base.assign(
        free=pd.array([f"free text {i}" for i in range(n)], dtype="string"),
        code=pd.array([("007", "012", "5", "9")[i % 4] for i in range(n)], dtype="string"),
        mixed=pd.array([("1", "2", "3", "x")[i % 4] for i in range(n)], dtype="string"),
        empty=pd.array([None] * n, dtype="string"),
    )
    result = run_training(frame, CLF, FAST_CLASSIFIERS[:1])
    assert result.ok
    assert {d.name: d.kind for d in result.plan.dropped} == {
        "free": "text",
        "code": "code",
        "mixed": "mixed",
        "empty": "empty",
    }
    payload = result.to_dict()["columns"]
    assert all(d["reason"] for d in payload["dropped"])
    assert set(payload["numeric"]) == {"id", "x1", "x2"} and payload["categorical"] == ["cat"]


def test_the_fitted_fold_pipeline_scores_missing_values_and_unseen_categories() -> None:
    frame = classification_frame(240)
    plan = build_column_plan(profile_frame(frame), CLF)
    template = build_pipeline(plan, LogisticRegression(max_iter=300), scale_numeric=True)
    x = frame[list(plan.used)]
    y = frame["y"].astype(str).to_numpy(dtype=object)
    new = pd.DataFrame(
        {"id": [None], "x1": ["0.3"], "x2": [None], "cat": ["purple"]}, dtype="string"
    )
    fitted = fit_fold(template, x.iloc[:200], y[:200], new, ("a", "b", "c"))
    assert fitted.proba is not None and np.isclose(fitted.proba.sum(), 1.0)


# --- the held-out test part is untouched ---------------------------------------------------------


def damage_test_rows(frame: pd.DataFrame, rows: Array, config: TaskConfig) -> pd.DataFrame:
    """Rewrite the held-out rows: other labels and garbage features (even malformed numbers)."""
    damaged = frame.copy()
    for column in damaged.columns:
        values = damaged[column].astype(object).to_numpy().copy()
        if column == config.target:
            if config.task == "classification":
                existing = sorted(set(values.tolist()))
                values[rows] = [
                    existing[(existing.index(v) + 1) % len(existing)] for v in values[rows]
                ]
            else:
                values[rows] = [f"{9999.5 + i}" for i in range(len(rows))]
        elif column == "cat":
            values[rows] = "category-only-in-the-test-part"
        else:
            values[rows] = "???garbage"
        damaged[column] = pd.array(values, dtype="string")
    return damaged


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_changing_only_the_test_part_changes_no_result(task: str) -> None:
    config = CLF if task == "classification" else REG
    frame = classification_frame(240) if task == "classification" else regression_frame(200)
    requests = FAST_CLASSIFIERS if task == "classification" else FAST_REGRESSORS
    target = frame[config.target].astype(str).to_numpy(dtype=object)
    labels = target if task == "classification" else target.astype(float)
    split = make_split(labels, config.task, TrainSettings())
    original = run_training(frame, config, requests, split=split)
    damaged = damage_test_rows(frame, split.test_rows, config)
    assert not damaged.iloc[split.test_rows].equals(frame.iloc[split.test_rows])  # really changed
    changed = run_training(damaged, config, requests, split=split)
    assert original.ok and changed.ok
    assert strip_times(original.to_dict()) == strip_times(changed.to_dict())
    for a, b in zip(original.results, changed.results, strict=True):
        assert np.array_equal(np.asarray(a.oof_predicted), np.asarray(b.oof_predicted))


def test_only_training_rows_ever_reach_fitting_or_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = classification_frame(240)
    seen_ids: set[str] = set()
    original = train_module.fit_fold

    def spy(
        template: Any, x_fit: pd.DataFrame, y_fit: Array, x_val: pd.DataFrame, classes: Any
    ) -> FoldFit:
        seen_ids.update(x_fit["id"].tolist())
        seen_ids.update(x_val["id"].tolist())
        return original(template, x_fit, y_fit, x_val, classes)

    monkeypatch.setattr(train_module, "fit_fold", spy)
    result = run_training(frame, CLF, FAST_CLASSIFIERS[:2])
    train_ids = set(frame["id"].iloc[result.split.train_rows].tolist())
    test_ids = set(frame["id"].iloc[result.split.test_rows].tolist())
    assert seen_ids == train_ids and seen_ids.isdisjoint(test_ids)


def test_results_contain_no_test_metrics_or_test_predictions() -> None:
    result = run_training(classification_frame(240), CLF, FAST_CLASSIFIERS[:1])
    payload = result.to_dict()
    assert payload["split"]["test_rows"] > 0  # only a count of reserved rows
    assert not any("test" in key for r in payload["results"] for key in _all_keys(r))
    assert all(k not in payload for k in ("test_metrics", "test_predictions", "test_score"))
    assert (
        isinstance(result.split, Split)
        and len(result.split.test_rows) == payload["split"]["test_rows"]
    )


def _all_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(k) for k in value] + [k for v in value.values() for k in _all_keys(v)]
    if isinstance(value, list):
        return [k for v in value for k in _all_keys(v)]
    return []
