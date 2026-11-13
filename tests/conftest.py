"""Shared fixtures for the selection, finalization and packaging tests (small invented data)."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_ops.finalize import DatasetInfo, FinalizeResult, finalize
from automl_ops.hashing import sha256_bytes
from automl_ops.profile import read_text_frame
from automl_ops.selection import FrozenSelection, freeze_selection, select_model
from automl_ops.task import TaskConfig
from automl_ops.train import AlgorithmRequest, TrainingResult, TrainSettings, run_training

CLF = TaskConfig("y", "classification")
REG = TaskConfig("y", "regression")
FAST = [
    AlgorithmRequest("logistic_regression"),
    AlgorithmRequest("random_forest", {"n_estimators": 10}),
]
FAST_REG = [AlgorithmRequest("ridge"), AlgorithmRequest("random_forest", {"n_estimators": 10})]
SETTINGS = TrainSettings(folds=3)


def toy_bytes(task: str = "classification", n: int = 150) -> bytes:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n).round(4)
    z = rng.normal(size=n).round(4)
    color = rng.choice(["red", "green", "blue"], n)
    frame = pd.DataFrame({"x": x, "z": z, "color": color})
    frame.loc[::17, "z"] = np.nan  # a few empty cells
    if task == "classification":
        frame["y"] = np.where(x + 0.5 * z.round(4) > 0, "pos", "neg")
    else:
        frame["y"] = (2 * x + 0.1 * rng.normal(size=n)).round(4)
    return frame.to_csv(index=False).encode()


def toy_dataset(data: bytes) -> tuple[pd.DataFrame, DatasetInfo]:
    frame = read_text_frame(data)
    info = DatasetInfo("toy@v1", "toy", 1, sha256_bytes(data), len(frame), tuple(frame.columns))
    return frame, info


@dataclass(frozen=True, eq=False)
class Run:
    frame: pd.DataFrame
    config: TaskConfig
    training: TrainingResult
    frozen: FrozenSelection
    dataset: DatasetInfo
    root: Path
    result: FinalizeResult


def make_run(root: Path, task: str = "classification") -> Run:
    frame, info = toy_dataset(toy_bytes(task))
    config = CLF if task == "classification" else REG
    training = run_training(frame, config, FAST if task == "classification" else FAST_REG, SETTINGS)
    frozen = freeze_selection(training, select_model(training))
    result = finalize(frame, config, training, frozen, info, root, confirm_final_test=True)
    return Run(frame, config, training, frozen, info, root, result)


@pytest.fixture(scope="session")
def clf_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return make_run(tmp_path_factory.mktemp("models-clf"))


@pytest.fixture(scope="session")
def reg_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return make_run(tmp_path_factory.mktemp("models-reg"), "regression")


def writable_copy(run: Run, destination: Path) -> Path:
    """A writable copy of the sealed package (so a test can tamper with it); returns its root."""
    target = destination / run.result.package_id
    shutil.copytree(run.result.path, target)
    target.chmod(0o755)
    for child in target.iterdir():
        child.chmod(0o644)
    return destination


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def read_only(path: Path) -> bool:
    return not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
