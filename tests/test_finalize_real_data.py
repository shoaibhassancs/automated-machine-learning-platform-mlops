"""Selection, the single test and packaging on the prepared demo files (skipped until
`make demo-data`). Adult uses a deterministic 1-in-7 sample of its training pool to keep the suite
quick; the full pools are measured separately (docs/selection-and-packaging.md)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from automl_ops.algorithms import algorithms_for
from automl_ops.finalize import DatasetInfo, finalize
from automl_ops.package import load_package, verify_package
from automl_ops.profile import read_text_frame
from automl_ops.selection import freeze_selection, select_model
from automl_ops.task import TaskConfig
from automl_ops.train import AlgorithmRequest, run_training

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"

pytestmark = pytest.mark.skipif(
    not (DEMO / "manifest.json").exists(),
    reason="demo data not prepared; run `make demo-data` first",
)


def workflow(name: str, config: TaskConfig, root: Path, every: int = 1):  # type: ignore[no-untyped-def]
    data = (DEMO / name).read_bytes()
    frame = read_text_frame(data)
    if every > 1:
        frame = frame.iloc[::every].reset_index(drop=True)
    info = DatasetInfo(
        f"{name}@v1", name, 1, hashlib.sha256(data).hexdigest(), len(frame), tuple(frame.columns)
    )
    requests = [AlgorithmRequest(a.key) for a in algorithms_for(config.task)]
    training = run_training(frame, config, requests)
    report = select_model(training)
    frozen = freeze_selection(training, report)
    result = finalize(frame, config, training, frozen, info, root, confirm_final_test=True)
    return frame, training, report, frozen, result


def test_abalone_workflow_selects_tests_once_packages_and_reloads_identically(
    tmp_path: Path,
) -> None:
    config = TaskConfig("Rings", "regression")
    frame, training, report, frozen, result = workflow(
        "abalone_train_pool.csv", config, tmp_path / "a"
    )
    assert training.ok and report.to_dict()["metric"] == "rmse"
    assert [r["algorithm"] for r in report.to_dict()["ranking"]] and not report.to_dict()[
        "excluded"
    ]
    metrics = result.test_result["evaluation"]["metrics"]
    assert 0.4 < metrics["r2"] < 0.7 and metrics["rmse"] < 3.0
    assert result.test_result["evaluation"]["rows"] == len(training.split.test_rows) == 585
    package = load_package(tmp_path / "a", result.package_id)
    assert verify_package(tmp_path / "a", result.package_id).ok
    assert package.feature_columns == list(training.plan.used)

    # the same run again gives the same selection and the same predictions (this machine only)
    _, _, _, frozen_again, result_again = workflow("abalone_train_pool.csv", config, tmp_path / "b")
    assert frozen_again == frozen
    again = load_package(tmp_path / "b", result_again.package_id)
    assert np.allclose(package.predict(frame), again.predict(frame), rtol=1e-12, atol=1e-12)
    assert result_again.test_result["evaluation"] == result.test_result["evaluation"]


def test_adult_sample_workflow_selects_tests_once_packages_and_reloads_identically(
    tmp_path: Path,
) -> None:
    config = TaskConfig("income", "classification")
    frame, training, report, frozen, result = workflow(
        "adult_train_pool.csv", config, tmp_path / "a", every=7
    )
    data = report.to_dict()
    assert training.ok and data["metric"] == "f1_macro" and data["positive_class"] == ">50K"
    metrics = result.test_result["evaluation"]["metrics"]
    assert metrics["accuracy"] > 0.78 and metrics["roc_auc"] > 0.8
    package = load_package(tmp_path / "a", result.package_id)
    assert package.classes == ["<=50K", ">50K"]
    assert verify_package(tmp_path / "a", result.package_id).ok
    sample = frame.head(200)
    assert set(package.predict(sample)) <= {"<=50K", ">50K"}
    assert np.allclose(package.predict_proba(sample).sum(axis=1), 1.0)
    again = workflow("adult_train_pool.csv", config, tmp_path / "b", every=7)
    assert again[3] == frozen
    assert list(load_package(tmp_path / "b", again[4].package_id).predict(sample)) == list(
        package.predict(sample)
    )
