"""The single final test: protocol, isolation and the one-attempt rule (BUILD_PLAN Steps 11-12)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import automl_ops.finalize as finalize_module
from automl_ops.algorithms import get_algorithm
from automl_ops.finalize import FinalizeError, finalize, run_final_test
from automl_ops.package import (
    PackageSpentError,
    begin_test_attempt,
    list_packages,
    load_package,
    package_state,
    verify_package,
)
from automl_ops.pipeline import build_pipeline
from automl_ops.selection import (
    FrozenSelection,
    SelectionError,
    freeze_selection,
    select_model,
)
from automl_ops.train import make_split, run_training, target_values
from conftest import (
    CLF,
    FAST,
    SETTINGS,
    Run,
    make_run,
    toy_bytes,
    toy_dataset,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "automl_ops"


# --- the protocol, end to end -----------------------------------------------------------------


def test_the_selected_model_is_refit_on_the_training_part_only_and_tested_once(
    clf_run: Run,
) -> None:
    run = clf_run
    result = run.result.test_result
    package = load_package(run.root, run.result.package_id)
    split = run.training.split
    train_rows, test_rows = split.train_rows, split.test_rows
    assert set(train_rows).isdisjoint(test_rows)

    # the learned statistics are those of the training rows, not of all rows
    scaler = package.pipeline.named_steps["prep"].named_transformers_["numeric"].named_steps
    train_x = np.array([float(v) for v in run.frame.iloc[train_rows]["x"]])
    all_x = np.array([float(v) for v in run.frame["x"]])
    if run.frozen.data["scale_numeric"]:
        assert scaler["scale"].mean_[0] == pytest.approx(train_x.mean())
        assert scaler["scale"].mean_[0] != pytest.approx(all_x.mean())

    # the recorded test score is what the packaged model scores on exactly the test rows
    test_frame = run.frame.iloc[test_rows]
    expected = accuracy_score(test_frame["y"], package.predict(test_frame))
    assert result["evaluation"]["metrics"]["accuracy"] == pytest.approx(expected)
    assert result["evaluation"]["rows"] == len(test_rows)
    assert result["algorithm"] == run.frozen.data["algorithm"]
    assert result["selection_sha256"] == run.frozen.sha256
    assert result["dataset"]["sha256"] == run.dataset.sha256
    assert result["test_split"]["sha256"] == split.fingerprints()["test_sha256"]
    assert "not independent or leakage-proof" in result["limitation"]
    assert package_state(run.result.path) == "sealed"


def test_the_packaged_model_equals_an_independent_refit_on_the_training_rows(clf_run: Run) -> None:
    run = clf_run
    spec = get_algorithm("classification", run.frozen.data["algorithm"])
    train = run.frame.iloc[run.training.split.train_rows]
    pipe = build_pipeline(
        run.training.plan,
        spec.build(run.frozen.data["params"], run.frozen.data["seed"]),
        scale_numeric=spec.scale_numeric,
    )
    pipe.fit(train[list(run.training.plan.used)], target_values(train, run.config))
    package = load_package(run.root, run.result.package_id)
    assert list(package.predict(run.frame)) == list(
        pipe.predict(run.frame[list(package.feature_columns)])
    )
    assert np.allclose(
        package.predict_proba(run.frame),
        pipe.predict_proba(run.frame[list(package.feature_columns)]),
        rtol=1e-12,
    )


def test_regression_is_finalized_with_residual_points_and_no_positive_class(reg_run: Run) -> None:
    result = reg_run.result.test_result
    assert result["positive_class"] is None and result["selection_metric"] == "rmse"
    assert set(result["evaluation"]["metrics"]) == {"rmse", "mae", "r2"}
    test_rows = set(reg_run.training.split.test_rows.tolist())
    assert {p["row"] for p in result["evaluation"]["residual_points"]} == test_rows
    assert verify_package(reg_run.root, reg_run.result.package_id).ok


def test_the_binary_positive_class_is_recorded_the_same_way_everywhere(clf_run: Run) -> None:
    manifest = json.loads((clf_run.result.path / "manifest.json").read_text())
    result = clf_run.result.test_result
    frozen = clf_run.frozen.data
    positives = {
        frozen["positive_class"],
        frozen["selection"]["positive_class"],
        manifest["positive_class"],
        manifest["selection"]["positive_class"],
        result["positive_class"],
    }
    assert positives == {manifest["classes"][1]} == {"pos"}


def test_finalization_without_explicit_confirmation_is_refused(tmp_path: Path) -> None:
    frame, info = toy_dataset(toy_bytes())
    training = run_training(frame, CLF, FAST, SETTINGS)
    frozen = freeze_selection(training, select_model(training))
    with pytest.raises(FinalizeError, match="confirm"):
        finalize(frame, CLF, training, frozen, info, tmp_path / "m", confirm_final_test=False)
    assert not (tmp_path / "m").exists()


# --- order of events and the one-attempt rule --------------------------------------------------


def test_the_test_part_is_read_once_after_the_freeze_check_and_the_attempt_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    real = finalize_module._read_test_part

    def spy(frame: Any, split: Any, config: Any, columns: Any, token: Any) -> Any:
        marker = token.path
        seen.append(
            {
                "attempt_exists": marker.exists(),
                "marker": json.loads(marker.read_text()),
                "package_files": sorted(p.name for p in marker.parent.iterdir()),
            }
        )
        return real(frame, split, config, columns, token)

    monkeypatch.setattr(finalize_module, "_read_test_part", spy)
    run = make_run(tmp_path)
    assert len(seen) == 1 and seen[0]["attempt_exists"]
    assert seen[0]["marker"]["selection_sha256"] == run.frozen.sha256
    # the package was already complete and reloadable when the test part was first read
    assert {"model.joblib", "manifest.json", "parity.json", "attempt.json"} <= set(
        seen[0]["package_files"]
    )
    assert "seal.json" not in seen[0]["package_files"]


def test_the_test_part_cannot_be_read_without_the_attempt_token() -> None:
    frame, _ = toy_dataset(toy_bytes())
    training = run_training(frame, CLF, FAST, SETTINGS)
    with pytest.raises(FinalizeError, match="before the attempt marker"):
        finalize_module._read_test_part(frame, training.split, CLF, ["x"], object())  # type: ignore[arg-type]


def test_a_second_attempt_on_the_same_package_is_refused_before_any_test_row_is_read(
    clf_run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[int] = []
    monkeypatch.setattr(finalize_module, "_read_test_part", lambda *a: reads.append(1))
    with pytest.raises(PackageSpentError):
        run_final_test(
            clf_run.result.path,
            clf_run.frame,
            clf_run.config,
            clf_run.training,
            clf_run.frozen,
            clf_run.dataset,
        )
    assert reads == []


def test_only_the_selected_model_ever_predicts_on_the_test_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    after_read: list[int] = []  # ids of the pipelines that predicted after the test part was read
    state = {"read": False}
    real_read = finalize_module._read_test_part
    real_predict = Pipeline.predict
    real_proba = Pipeline.predict_proba

    def read(*args: Any) -> Any:
        state["read"] = True
        return real_read(*args)

    def predict(self: Pipeline, x: Any, **kw: Any) -> Any:
        if state["read"]:
            after_read.append(id(self))
        return real_predict(self, x, **kw)

    def proba(self: Pipeline, x: Any, **kw: Any) -> Any:
        if state["read"]:
            after_read.append(id(self))
        return real_proba(self, x, **kw)

    monkeypatch.setattr(finalize_module, "_read_test_part", read)
    monkeypatch.setattr(Pipeline, "predict", predict)
    monkeypatch.setattr(Pipeline, "predict_proba", proba)
    make_run(tmp_path)
    assert len(after_read) == 2 and len(set(after_read)) == 1  # predict + predict_proba, one model


def test_a_failure_after_the_attempt_marker_spends_the_package_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any) -> Any:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(finalize_module, "_read_test_part", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        make_run(tmp_path)
    (package_id,) = list_packages(tmp_path)
    path = tmp_path / package_id
    assert package_state(path) == "spent" and not (path / "test_result.json").exists()
    assert not verify_package(tmp_path, package_id).ok
    with pytest.raises(PackageSpentError):
        begin_test_attempt(path, {})
    assert list_packages(tmp_path) == [package_id]  # no second package, no runner-up fallback


def test_a_refit_failure_stops_without_a_package_or_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def broken(*_: Any, **__: Any) -> Any:
        calls.append(1)
        raise RuntimeError("cannot fit")

    monkeypatch.setattr(finalize_module, "build_pipeline", broken)
    with pytest.raises(RuntimeError, match="cannot fit"):
        make_run(tmp_path)
    assert calls == [1] and list(tmp_path.iterdir()) == []


# --- the frozen selection and the training run must agree --------------------------------------


def test_a_tampered_frozen_selection_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    frame, info = toy_dataset(toy_bytes())
    training = run_training(frame, CLF, FAST, SETTINGS)
    frozen = freeze_selection(training, select_model(training))
    other = (
        "random_forest"
        if frozen.data["algorithm"] == "logistic_regression"
        else "logistic_regression"
    )
    forged = FrozenSelection(
        frozen.payload.replace(frozen.data["algorithm"], other, 1), frozen.sha256
    )
    with pytest.raises(SelectionError, match="SHA-256"):
        finalize(frame, CLF, training, forged, info, tmp_path / "m", confirm_final_test=True)
    assert not (tmp_path / "m").exists()


def test_a_selection_from_another_run_is_refused(tmp_path: Path) -> None:
    frame, info = toy_dataset(toy_bytes())
    training = run_training(frame, CLF, FAST, SETTINGS)
    different = run_training(frame, CLF, FAST, SETTINGS.__class__(folds=3, seed=7))
    frozen = freeze_selection(different, select_model(different))
    with pytest.raises(FinalizeError, match="do not match"):
        finalize(frame, CLF, training, frozen, info, tmp_path / "m", confirm_final_test=True)
    assert not (tmp_path / "m").exists()
    short = frame.iloc[:-5]
    own = freeze_selection(training, select_model(training))
    with pytest.raises(FinalizeError, match="row count"):
        finalize(short, CLF, training, own, info, tmp_path / "m", confirm_final_test=True)


# --- test information cannot influence selection ----------------------------------------------


def test_rewriting_the_test_rows_changes_neither_the_selection_nor_the_fitted_model(
    tmp_path: Path,
) -> None:
    frame, info = toy_dataset(toy_bytes())
    split = make_split(target_values(frame, CLF), "classification", SETTINGS)
    altered = frame.copy()
    rows = split.test_rows
    altered.loc[rows, "x"] = "999"
    altered.loc[rows, "z"] = "-999"
    altered.loc[rows, "color"] = "never-seen"
    altered.loc[rows[::2], "y"] = np.where(altered.loc[rows[::2], "y"] == "pos", "neg", "pos")

    runs = []
    for index, data in enumerate((frame, altered)):
        training = run_training(data, CLF, FAST, SETTINGS, split=split)
        frozen = freeze_selection(training, select_model(training))
        result = finalize(
            data, CLF, training, frozen, info, tmp_path / str(index), confirm_final_test=True
        )
        runs.append((frozen, load_package(tmp_path / str(index), result.package_id), result))
    (frozen_a, package_a, result_a), (frozen_b, package_b, result_b) = runs
    assert frozen_a == frozen_b  # identical payload and hash
    stats_a = package_a.pipeline.named_steps["prep"].named_transformers_["numeric"]
    stats_b = package_b.pipeline.named_steps["prep"].named_transformers_["numeric"]
    for name in stats_a.named_steps:
        step_a, step_b = stats_a.named_steps[name], stats_b.named_steps[name]
        for attribute in ("statistics_", "mean_", "scale_"):
            if hasattr(step_a, attribute):
                assert np.array_equal(getattr(step_a, attribute), getattr(step_b, attribute))
    categories_a = package_a.pipeline.named_steps["prep"].named_transformers_["categorical"]
    assert list(categories_a.named_steps["encode"].categories_[0]) == ["blue", "green", "red"]
    # sensitivity control: the test results themselves DO differ
    assert result_a.test_result["evaluation"] != result_b.test_result["evaluation"]


def test_malformed_values_in_the_test_rows_do_not_disturb_training_or_selection() -> None:
    frame, _ = toy_dataset(toy_bytes())
    split = make_split(target_values(frame, CLF), "classification", SETTINGS)
    broken = frame.copy()
    broken.loc[split.test_rows, "x"] = "not-a-number"
    plain = run_training(frame, CLF, FAST, SETTINGS, split=split)
    other = run_training(broken, CLF, FAST, SETTINGS, split=split)
    assert freeze_selection(plain, select_model(plain)) == freeze_selection(
        other, select_model(other)
    )


def test_the_test_rows_are_indexed_in_exactly_one_function_of_the_whole_code_base() -> None:
    """Apart from counting them (`len(...)`) and hashing their positions for the fingerprint,
    `test_rows` is used only by `_read_test_part`; selection, packaging and the scripts never
    touch it."""
    found: dict[str, set[str]] = {}
    files = [*SRC.glob("*.py"), *(SRC.parents[1] / "scripts").glob("*.py")]
    for path in files:
        tree = ast.parse(path.read_text())
        counted = {
            id(call.args[0])
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "len"
            and call.args
        }
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "test_rows"
                    and id(node) not in counted
                ):
                    found.setdefault(path.name, set()).add(function.name)
    assert found == {"finalize.py": {"_read_test_part"}, "train.py": {"fingerprints"}}, found
