"""Cross-validation on the prepared Adult and Abalone demo files (skipped until `make demo-data`).

All default algorithms run sequentially. Adult uses a deterministic 5,000-row sample of its training
pool to keep the test suite quick; the full pools are measured separately (docs/training.md). The
demo files are only read, never modified."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from automl_ops.algorithms import algorithms_for
from automl_ops.profile import read_text_frame
from automl_ops.task import TaskConfig
from automl_ops.train import AlgorithmRequest, TrainSettings, make_split, run_training

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"

pytestmark = pytest.mark.skipif(
    not (DEMO / "manifest.json").exists(),
    reason="demo data not prepared; run `make demo-data` first",
)

ADULT = TaskConfig("income", "classification")
ABALONE = TaskConfig("Rings", "regression")


def load(name: str, every: int = 1) -> pd.DataFrame:
    frame = read_text_frame((DEMO / name).read_bytes())
    return frame.iloc[::every].reset_index(drop=True) if every > 1 else frame


def defaults(config: TaskConfig) -> list[AlgorithmRequest]:
    return [AlgorithmRequest(a.key) for a in algorithms_for(config.task)]


def strip_times(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_times(v) for k, v in value.items() if k not in ("seconds", "fold_seconds")}
    if isinstance(value, list):
        return [strip_times(v) for v in value]
    return value


def test_abalone_all_default_regressors() -> None:
    frame = load("abalone_train_pool.csv")
    result = run_training(frame, ABALONE, defaults(ABALONE))
    assert result.ok and [r.key for r in result.results] == [
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    ]
    assert result.split.to_dict() == {
        "rows": 2924,
        "train_rows": 2339,
        "test_rows": 585,
        "stratified": False,
    }
    assert result.plan.numeric == (
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
    )
    assert result.plan.categorical == ("Sex",) and result.plan.dropped == ()
    for r in result.results:
        assert r.oof is not None and r.oof["rows"] == 2339 and len(r.fold_metrics) == 5
        r2 = r.summary["r2"]["mean"]
        assert r2 is not None and 0.4 < r2 < 0.7  # abalone age is only partly predictable
        assert r.summary["rmse"]["mean"] < 3.0
        # every residual is attached to a training row of the original data
        actual = frame["Rings"].astype(float).to_numpy()
        train_rows = set(result.split.train_rows.tolist())
        for p in r.oof["residual_points"]:
            assert p["row"] in train_rows and p["actual"] == pytest.approx(actual[p["row"]])


def test_abalone_results_repeat_exactly_and_ignore_the_test_part() -> None:
    frame = load("abalone_train_pool.csv")
    labels = frame["Rings"].astype(float).to_numpy()
    split = make_split(labels, "regression", TrainSettings())
    requests = defaults(ABALONE)
    first = run_training(frame, ABALONE, requests, split=split)
    second = run_training(frame, ABALONE, requests, split=split)
    assert strip_times(first.to_dict()) == strip_times(second.to_dict())

    damaged = frame.copy()
    for column in damaged.columns:
        values = damaged[column].astype(object).to_numpy().copy()
        values[split.test_rows] = "9999" if column == "Rings" else "???"
        damaged[column] = pd.array(values, dtype="string")
    changed = run_training(damaged, ABALONE, requests, split=split)
    assert changed.ok
    assert strip_times(first.to_dict()) == strip_times(changed.to_dict())


def test_adult_sample_all_default_classifiers() -> None:
    frame = load("adult_train_pool.csv", every=7)
    result = run_training(frame, ADULT, defaults(ADULT))
    assert result.ok and len(frame) > 4500
    assert result.classes == ("<=50K", ">50K") and result.folds_used == 5
    assert len(result.plan.numeric) == 6 and len(result.plan.categorical) == 8
    assert result.plan.dropped == () and result.warnings == ()
    n_train = len(result.split.train_rows)
    for r in result.results:
        assert r.oof is not None and r.oof["rows"] == n_train
        assert sum(sum(row) for row in r.oof["confusion_matrix"]) == n_train
        assert not pd.isna(pd.Series(r.oof_predicted)).any()
        assert r.oof_proba is not None and np.allclose(r.oof_proba.sum(axis=1), 1.0)
        auc, accuracy = r.summary["roc_auc"]["mean"], r.summary["accuracy"]["mean"]
        assert auc is not None and auc > 0.8 and accuracy is not None and accuracy > 0.78
        assert len(r.oof["roc_curves"]) == 1 and r.oof["roc_curves"][0]["class"] == ">50K"


def test_demo_files_are_not_modified_by_training() -> None:
    def digest() -> dict[str, str]:
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEMO.glob("*.csv"))
        }

    before = digest()
    run_training(load("abalone_train_pool.csv"), ABALONE, [AlgorithmRequest("ridge")])
    assert digest() == before
