"""Model packages: lifecycle, saving, loading, integrity and prediction parity (Step 12)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import automl_ops.package as package_module
from automl_ops.package import (
    ModelPackage,
    PackageError,
    PackageSpentError,
    begin_test_attempt,
    create_package,
    leftover_staging,
    list_packages,
    load_package,
    package_path,
    package_state,
    parity_rows,
    seal_package,
    verify_package,
    write_test_result,
)
from automl_ops.pipeline import MalformedValueError, build_column_plan, build_pipeline
from automl_ops.profile import profile_frame
from conftest import CLF, Run, is_root, read_only, toy_bytes, toy_dataset, writable_copy


def open_package(root: Path) -> tuple[Path, pd.DataFrame]:
    """A small open (created, not finalized) package, built without the training runner."""
    frame, _ = toy_dataset(toy_bytes())
    plan = build_column_plan(profile_frame(frame), CLF)
    pipe = build_pipeline(plan, LogisticRegression(max_iter=200), scale_numeric=True)
    x = frame[list(plan.used)]
    pipe.fit(x, frame["y"])
    kinds = {n: "numeric" for n in plan.numeric} | {n: "categorical" for n in plan.categorical}
    manifest = {
        "classes": ["neg", "pos"],
        "features": {"input_columns": [{"name": n, "kind": kinds[n]} for n in plan.used]},
        "selection_sha256": "abc",
    }
    return create_package(root, pipe, manifest, parity_rows(x)), frame


# --- lifecycle ----------------------------------------------------------------------------------


def test_a_new_package_is_open_complete_and_read_only_with_no_staging_left(tmp_path: Path) -> None:
    path, _ = open_package(tmp_path)
    assert package_state(path) == "open" and path.parent == tmp_path
    assert sorted(p.name for p in path.iterdir()) == [
        "manifest.json",
        "model.joblib",
        "parity.json",
    ]
    assert all(read_only(p) for p in path.iterdir())
    assert leftover_staging(tmp_path) == [] and list_packages(tmp_path) == [path.name]
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["package_format"] == 1 and manifest["model_file"]["sha256"]
    assert set(manifest["libraries"]) >= {"python", "scikit-learn", "pandas", "numpy", "joblib"}


def test_a_failed_parity_check_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_module, "_parity_problems", lambda *_: ["predictions differ"])
    with pytest.raises(PackageError, match="failed its parity check"):
        open_package(tmp_path)
    assert list(tmp_path.iterdir()) == []  # no staging folder, no package


def test_an_interrupted_creation_is_visible_but_never_a_package(tmp_path: Path) -> None:
    (tmp_path / ".staging-m-20260101T000000Z-deadbeef").mkdir()
    assert list_packages(tmp_path) == [] and len(leftover_staging(tmp_path)) == 1


def test_the_attempt_marker_and_the_test_result_can_each_be_created_only_once(
    tmp_path: Path,
) -> None:
    path, _ = open_package(tmp_path)
    with pytest.raises(PackageError, match="only follow the attempt marker"):
        write_test_result(path, {"x": 1})
    begin_test_attempt(path, {"n": 1})
    assert package_state(path) == "spent"
    with pytest.raises(PackageSpentError, match="never retried"):
        begin_test_attempt(path, {"n": 2})
    assert json.loads((path / "attempt.json").read_text()) == {"n": 1}  # not overwritten
    write_test_result(path, {"a": 1})
    with pytest.raises(PackageSpentError):
        write_test_result(path, {"a": 2})
    assert json.loads((path / "test_result.json").read_text()) == {"a": 1}


def test_sealing_needs_an_attempt_and_a_result_and_makes_everything_read_only(
    tmp_path: Path,
) -> None:
    path, _ = open_package(tmp_path)
    with pytest.raises(PackageError, match="only a package with a test attempt"):
        seal_package(path)
    begin_test_attempt(path, {})
    with pytest.raises(PackageError, match="no test result"):
        seal_package(path)
    write_test_result(path, {"selection_sha256": "abc"})
    seal = seal_package(path)
    assert package_state(path) == "sealed" and set(seal["files"]) == {
        "model.joblib",
        "manifest.json",
        "parity.json",
        "attempt.json",
        "test_result.json",
    }
    assert all(read_only(p) for p in path.iterdir()) and read_only(path)
    if not is_root():
        with pytest.raises(PermissionError):
            (path / "extra.txt").write_text("x")
        with pytest.raises(PermissionError):
            (path / "manifest.json").write_text("{}")
    with pytest.raises(PackageSpentError):
        begin_test_attempt(path, {})


def test_an_open_or_spent_package_cannot_be_loaded(tmp_path: Path) -> None:
    path, _ = open_package(tmp_path)
    with pytest.raises(PackageError, match="not been finalized"):
        load_package(tmp_path, path.name)
    begin_test_attempt(path, {})
    with pytest.raises(PackageError, match="never retried"):
        load_package(tmp_path, path.name)
    check = verify_package(tmp_path, path.name)
    assert check.state == "spent" and not check.ok


def test_package_ids_resolve_only_inside_the_models_folder(tmp_path: Path) -> None:
    for bad in ("../etc", "m-1", "/tmp", "m-20260101T000000Z-zzzzzzzz"):
        with pytest.raises(PackageError):
            package_path(tmp_path, bad)
    (tmp_path / "elsewhere").mkdir()
    link = tmp_path / "m-20260101T000000Z-deadbeef"
    link.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(PackageError, match="no package"):
        package_path(tmp_path, link.name)


# --- loading, parity, feature order -------------------------------------------------------------


def test_a_sealed_package_loads_and_predicts_like_the_pipeline_that_was_saved(clf_run: Run) -> None:
    loaded = load_package(clf_run.root, clf_run.result.package_id)
    assert isinstance(loaded, ModelPackage)
    assert loaded.feature_columns == list(clf_run.training.plan.used)  # order preserved
    assert loaded.classes == ["neg", "pos"]
    parity = json.loads((loaded.path / "parity.json").read_text())
    frame = pd.DataFrame(parity["rows"], columns=parity["columns"], dtype="string")
    assert [str(v) for v in loaded.predict(frame)] == parity["predictions"]
    assert np.allclose(loaded.predict_proba(frame), parity["probabilities"], rtol=1e-12, atol=1e-12)
    assert verify_package(clf_run.root, loaded.package_id).ok


def test_extra_columns_are_ignored_and_a_missing_column_is_named(clf_run: Run) -> None:
    loaded = load_package(clf_run.root, clf_run.result.package_id)
    rows = clf_run.frame.head(8)
    assert list(loaded.predict(rows.assign(surprise="x"))) == list(loaded.predict(rows))
    with pytest.raises(PackageError, match=r"missing column.*'color'"):
        loaded.predict(rows.drop(columns=["color"]))
    with pytest.raises(MalformedValueError):
        loaded.predict(rows.assign(x="abc"))


def test_a_regression_package_has_no_probabilities(reg_run: Run) -> None:
    loaded = load_package(reg_run.root, reg_run.result.package_id)
    assert loaded.classes is None and loaded.predict(reg_run.frame.head(5)).shape == (5,)
    with pytest.raises(PackageError, match="regression"):
        loaded.predict_proba(reg_run.frame.head(5))


# --- integrity and versions ---------------------------------------------------------------------


def test_a_changed_model_file_is_caught_before_it_is_deserialized(
    clf_run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = writable_copy(clf_run, tmp_path)
    model = root / clf_run.result.package_id / "model.joblib"
    data = bytearray(model.read_bytes())
    data[len(data) // 2] ^= 0xFF
    model.write_bytes(bytes(data))
    loads: list[int] = []
    monkeypatch.setattr(joblib, "load", lambda *_: loads.append(1))
    with pytest.raises(PackageError, match="model.joblib does not match"):
        load_package(root, clf_run.result.package_id)
    assert loads == []  # joblib.load was never reached
    check = verify_package(root, clf_run.result.package_id)
    assert not check.ok and any("SHA-256" in p for p in check.problems)


def test_a_changed_manifest_or_a_missing_file_fails_verification(
    clf_run: Run, tmp_path: Path
) -> None:
    root = writable_copy(clf_run, tmp_path)
    folder = root / clf_run.result.package_id
    (folder / "manifest.json").write_text("{}")
    (folder / "parity.json").unlink()
    check = verify_package(root, clf_run.result.package_id)
    assert not check.ok
    assert any("manifest.json does not match" in p for p in check.problems)
    assert any("parity.json is missing" in p for p in check.problems)
    with pytest.raises(PackageError, match="failed verification"):
        load_package(root, clf_run.result.package_id)


def test_a_writable_sealed_package_is_reported(clf_run: Run, tmp_path: Path) -> None:
    root = writable_copy(clf_run, tmp_path)  # same bytes, but writable
    check = verify_package(root, clf_run.result.package_id)
    assert not check.ok and any("is writable" in p for p in check.problems)


def test_a_scikit_learn_version_mismatch_is_refused_unless_allowed(
    clf_run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = package_module.installed_version
    monkeypatch.setattr(
        package_module,
        "installed_version",
        lambda name: "0.0.1" if name == "scikit-learn" else real(name),
    )
    with pytest.raises(PackageError, match="saved with scikit-learn"):
        load_package(clf_run.root, clf_run.result.package_id)
    assert load_package(clf_run.root, clf_run.result.package_id, allow_version_mismatch=True)
    assert not verify_package(clf_run.root, clf_run.result.package_id).ok


def test_an_unsupported_package_or_pipeline_format_is_reported() -> None:
    problems = package_module._version_problems(
        {"package_format": 99, "formats": {"pipeline": 0, "train": 0}, "libraries": {}}
    )
    assert any("package format 99" in p for p in problems)
    assert any("pipeline or training format" in p for p in problems)


def test_the_model_file_must_hold_a_pipeline_with_the_manifest_feature_order(
    tmp_path: Path,
) -> None:
    path, _ = open_package(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["features"]["input_columns"].reverse()
    with pytest.raises(PackageError, match="feature order"):
        package_module._read_pipeline(path, manifest)


def test_the_manifest_records_what_a_later_registry_or_server_needs(clf_run: Run) -> None:
    manifest = json.loads((clf_run.result.path / "manifest.json").read_text())
    assert manifest["task"] == {"target": "y", "task": "classification", "excluded": []}
    assert [c["name"] for c in manifest["features"]["input_columns"]] == ["x", "z", "color"]
    assert {c["name"]: c["kind"] for c in manifest["features"]["input_columns"]} == {
        "x": "numeric",
        "z": "numeric",
        "color": "categorical",
    }
    assert manifest["classes"] == ["neg", "pos"] and manifest["positive_class"] == "pos"
    assert manifest["selection"]["positive_class"] == "pos"
    assert manifest["dataset"]["sha256"] == clf_run.dataset.sha256
    assert manifest["selection_sha256"] == clf_run.frozen.sha256
    assert manifest["model"]["seed"] == 42 and manifest["settings"]["folds"] == 3
    assert manifest["model"]["algorithm"] == clf_run.frozen.data["algorithm"]
    assert manifest["training"]["results"] and manifest["limitations"]
