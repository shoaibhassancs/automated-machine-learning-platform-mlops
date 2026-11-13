"""The single final test and packaging of the selected model (BUILD_PLAN Steps 11-12).

`finalize` takes the **same** `TrainingResult` that the selection was made from, plus the
`FrozenSelection`, and does, in this order:

1. require explicit confirmation, verify the frozen selection's SHA-256 and check that the frozen
   selection, the training result and the data all describe the same run (split fingerprints, column
   plan, row counts, dataset hash);
2. refit a **fresh** pipeline on the training part only (no test row is indexed);
3. build the package (model, manifest, parity sample) in a staging folder, verify it by reloading,
   and move it into place atomically (state *open*);
4. `run_final_test`: create the attempt marker **exclusively**, and only then read the test part
   (`_read_test_part`, the one place the test rows are indexed), score the packaged model once, and
   create the test result exclusively;
5. seal the package (hashes, read-only).

Only the selected model is ever scored on the test part, and nothing about the result can change the
parameters or the selection: there is no code path from a test score back to a candidate. If the
refit or the test fails there is **no fallback** to the runner-up. If the attempt marker exists and
anything later fails, the package is *spent*: it is never retried and never loadable.

**Open decision (BUILD_PLAN Step 11): reuse of the test part across jobs.** The same dataset version
and seed give the same test rows in every job. This module does not enforce or forbid that. It
records the dataset SHA-256 and the test-row fingerprint in the package so reuse can be recognised
later, and every test result carries the limitation text below.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from automl_ops.algorithms import get_algorithm
from automl_ops.dataset_store import DatasetVersion
from automl_ops.evaluate import evaluate_classification, evaluate_regression
from automl_ops.package import (
    PackageError,
    begin_test_attempt,
    create_package,
    load_pipeline_for_test,
    package_state,
    parity_rows,
    seal_package,
    write_test_result,
)
from automl_ops.pipeline import build_pipeline
from automl_ops.selection import FrozenSelection
from automl_ops.task import TaskConfig
from automl_ops.train import Split, TrainingResult, target_values

Array = NDArray[Any]

RESULT_FORMAT = 1
LIMITATION = (
    "This result comes from the held-out test part, read once for this package. The same test part "
    "can be reused by other jobs on the same dataset version (an open decision), and repeated use "
    "across jobs can influence model choices. It is not independent or leakage-proof."
)


class FinalizeError(ValueError):
    """Finalization was refused or failed (the message says why)."""


@dataclass(frozen=True)
class DatasetInfo:
    """Which stored dataset version the run used (provenance)."""

    ref: str
    name: str
    number: int
    sha256: str
    rows: int
    columns: tuple[str, ...]

    @staticmethod
    def from_version(version: DatasetVersion) -> DatasetInfo:
        return DatasetInfo(
            version.ref,
            version.dataset,
            version.number,
            version.sha256,
            version.rows,
            version.columns,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "name": self.name,
            "version": self.number,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": list(self.columns),
        }


@dataclass(frozen=True, eq=False)
class FinalizeResult:
    package_id: str
    path: Path
    test_result: dict[str, Any]
    seconds: float


class _AttemptToken:
    """Proof that the attempt marker was created; `_read_test_part` refuses to run without it."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path


def _read_test_part(
    frame: pd.DataFrame,
    split: Split,
    config: TaskConfig,
    columns: list[str],
    token: _AttemptToken,
) -> tuple[pd.DataFrame, Array, Array]:
    """The ONLY place the held-out test rows are indexed. Needs the attempt-marker token."""
    if not isinstance(token, _AttemptToken) or not token.path.exists():
        raise FinalizeError("the test part cannot be read before the attempt marker exists")
    test_frame = frame.iloc[split.test_rows].reset_index(drop=True)
    return test_frame[columns], target_values(test_frame, config), split.test_rows


def _check_same_run(
    frame: pd.DataFrame,
    config: TaskConfig,
    training: TrainingResult,
    frozen: FrozenSelection,
    dataset: DatasetInfo,
) -> None:
    """Refuse unless the frozen selection, the training result and the data are one run."""
    data = frozen.data
    problems: list[str] = []
    if data["task"] != training.task or data["task"] != config.task:
        problems.append("the task differs")
    if data["target"] != training.target or data["target"] != config.target:
        problems.append("the target differs")
    if data["split"] != training.split.fingerprints():
        problems.append("the split differs from the one the selection was made on")
    if data["plan"] != training.plan.to_dict():
        problems.append("the column plan differs")
    if data["settings"] != training.settings.to_dict():
        problems.append("the training settings differ")
    if len(frame) != data["split"]["rows"] or len(frame) != dataset.rows:
        problems.append("the data does not have the row count of the training run")
    if tuple(frame.columns) != dataset.columns:
        problems.append("the data's columns differ from the dataset version's")
    chosen = [r for r in training.results if r.key == data["algorithm"] and r.status == "ok"]
    if not chosen or chosen[0].params != data["params"]:
        problems.append("the selected algorithm or its parameters are not in the training result")
    if problems:
        raise FinalizeError(
            "the frozen selection and the training run do not match: " + "; ".join(problems)
        )


def _manifest(
    config: TaskConfig,
    training: TrainingResult,
    frozen: FrozenSelection,
    dataset: DatasetInfo,
) -> dict[str, Any]:
    data = frozen.data
    plan = training.plan
    kinds = {name: "numeric" for name in plan.numeric} | {
        name: "categorical" for name in plan.categorical
    }
    return {
        "task": config.to_dict(),
        "target": config.target,
        "classes": data["classes"],
        "positive_class": data["positive_class"],
        "features": {
            "input_columns": [{"name": n, "kind": kinds[n]} for n in plan.used],
            "dropped": [d.to_dict() for d in plan.dropped],
        },
        "model": {
            "algorithm": data["algorithm"],
            "label": data["label"],
            "params": data["params"],
            "scale_numeric": data["scale_numeric"],
            "seed": data["seed"],
        },
        "settings": data["settings"],
        "split": data["split"],
        "dataset": dataset.to_dict(),
        "training": training.to_dict(),
        "selection": data["selection"],
        "selection_sha256": frozen.sha256,
        "limitations": [LIMITATION],
    }


def run_final_test(
    package_dir: Path,
    frame: pd.DataFrame,
    config: TaskConfig,
    training: TrainingResult,
    frozen: FrozenSelection,
    dataset: DatasetInfo,
) -> dict[str, Any]:
    """The one test attempt of an open package. A second call on the same package raises
    `PackageSpentError` before any test row is touched."""
    frozen.verify()
    split = training.split
    columns = list(training.plan.used)
    token = _AttemptToken(
        begin_test_attempt(
            package_dir,
            {
                "attempt_format": 1,
                "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "selection_sha256": frozen.sha256,
                "dataset_sha256": dataset.sha256,
                "test_sha256": split.fingerprints()["test_sha256"],
            },
        )
    )
    # From here on the package is spent whatever happens.
    pipeline = load_pipeline_for_test(package_dir)
    x_test, y_test, test_rows = _read_test_part(frame, split, config, columns, token)
    predicted: Array = pipeline.predict(x_test)
    classes = frozen.data["classes"]
    if classes is not None:
        if list(pipeline.classes_) != list(classes):
            raise FinalizeError("the packaged model's classes differ from the frozen class order")
        evaluation = evaluate_classification(
            y_test, predicted, pipeline.predict_proba(x_test), classes
        )
    else:
        evaluation = evaluate_regression(y_test, predicted, test_rows)
    result = {
        "result_format": RESULT_FORMAT,
        "package_id": package_dir.name,
        "selection_sha256": frozen.sha256,
        "dataset": {"ref": dataset.ref, "sha256": dataset.sha256},
        "test_split": {
            "rows": split.fingerprints()["test_rows"],
            "sha256": split.fingerprints()["test_sha256"],
        },
        "algorithm": frozen.data["algorithm"],
        "selection_metric": frozen.data["selection"]["metric"],
        "positive_class": frozen.data["positive_class"],
        "evaluation": evaluation,
        "limitation": LIMITATION,
    }
    write_test_result(package_dir, result)
    return result


def finalize(
    frame: pd.DataFrame,
    config: TaskConfig,
    training: TrainingResult,
    frozen: FrozenSelection,
    dataset: DatasetInfo,
    models_root: Path,
    *,
    confirm_final_test: bool,
) -> FinalizeResult:
    """Refit the frozen selection on the training part, package it, test it once, seal it."""
    began = datetime.now(UTC)
    if not confirm_final_test:
        raise FinalizeError(
            "refused: this reads the held-out test part once. Pass confirm_final_test=True "
            "(command line: --confirm) to proceed."
        )
    frozen.verify()
    _check_same_run(frame, config, training, frozen, dataset)
    data = frozen.data
    spec = get_algorithm(training.task, data["algorithm"])

    # Refit on the training part only: the test rows are not indexed anywhere on this path.
    train_frame = frame.iloc[training.split.train_rows].reset_index(drop=True)
    columns = list(training.plan.used)
    x_train = train_frame[columns]
    y_train = target_values(train_frame, config)
    pipeline: Pipeline = build_pipeline(
        training.plan,
        spec.build(data["params"], data["seed"]),
        scale_numeric=spec.scale_numeric,
    )
    pipeline.fit(x_train, y_train)
    if data["classes"] is not None and list(pipeline.classes_) != list(data["classes"]):
        raise FinalizeError("the refitted model's classes differ from the frozen class order")

    package_dir = create_package(
        models_root, pipeline, _manifest(config, training, frozen, dataset), parity_rows(x_train)
    )
    if package_state(package_dir) != "open":  # pragma: no cover - defensive
        raise PackageError("a new package must start open")
    result = run_final_test(package_dir, frame, config, training, frozen, dataset)
    seal_package(package_dir)
    seconds = (datetime.now(UTC) - began).total_seconds()
    return FinalizeResult(package_dir.name, package_dir, result, seconds)
