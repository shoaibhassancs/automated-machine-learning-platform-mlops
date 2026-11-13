"""Split, training runner and progress (BUILD_PLAN Step 9): cross-validation only.

`run_training` splits the rows into a **training part** and a **held-out test part**, then
cross-validates each requested algorithm on the training part and returns structured results
(`TrainingResult.to_dict()` is plain JSON-ready data; nothing here depends on the command line).

**The held-out test part is reserved by row indices and never read.** After the split, the runner
indexes only `split.train_rows`; no test row, label or statistic reaches preprocessing, fitting,
fold creation or scoring, and no test metric is computed. (The *split itself* is stratified, which
uses all labels to place rows, as any stratified split does, and the pre-flight feasibility checks
look at the whole dataset. After that, nothing touches the test rows.) Final refitting and scoring
on the test part is a later step (BUILD_PLAN Step 11).

**Preprocessing is fitted inside every fold.** Each fold clones the unfitted pipeline from
`pipeline.build_pipeline` and fits it on that fold's training rows only, so imputation values,
scaling and category lists never see the fold's validation rows.

**Feasibility.** Defaults: 20% test part, seed 42, 5 folds; stratified split for classification,
shuffled for regression. Classification needs each class to have at least 2 rows in the *training*
part after the holdout; if the smallest training class has fewer rows than the requested folds, the
fold count is reduced to that count with a warning. Anything that cannot work stops with a
`TrainingError` that says what to change, instead of proceeding with invalid cross-validation.

Algorithms run one at a time. A failing algorithm is reported and the others still finish.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from automl_ops.algorithms import AlgorithmSpec, get_algorithm
from automl_ops.evaluate import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    classification_scalars,
    evaluate_classification,
    evaluate_regression,
    regression_scalars,
    summarize_folds,
)
from automl_ops.hashing import sha256_bytes
from automl_ops.pipeline import ColumnPlan, build_column_plan, build_pipeline, text_to_number
from automl_ops.profile import profile_frame
from automl_ops.task import TaskConfig, TaskType, check_task_config, class_counts

Array = NDArray[Any]

TRAIN_FORMAT = 1
MIN_FOLDS = 2
MAX_FOLDS = 10
ERROR_MESSAGE_LIMIT = 500


class TrainingError(ValueError):
    """The requested training run cannot be carried out; the message says what to change."""


@dataclass(frozen=True)
class TrainSettings:
    """Provisional defaults (docs/training.md)."""

    test_fraction: float = 0.2
    seed: int = 42
    folds: int = 5

    def validate(self) -> None:
        if isinstance(self.test_fraction, bool) or not 0.05 <= self.test_fraction <= 0.5:
            raise TrainingError("test_fraction must be between 0.05 and 0.5")
        if isinstance(self.folds, bool) or not MIN_FOLDS <= self.folds <= MAX_FOLDS:
            raise TrainingError(f"folds must be a whole number from {MIN_FOLDS} to {MAX_FOLDS}")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise TrainingError("seed must be a whole number, zero or more")

    def to_dict(self) -> dict[str, Any]:
        return {"test_fraction": self.test_fraction, "seed": self.seed, "folds": self.folds}


DEFAULT_TRAIN_SETTINGS = TrainSettings()


@dataclass(frozen=True)
class AlgorithmRequest:
    key: str
    params: Mapping[str, Any] = field(default_factory=dict)


def _positions_hash(rows: Array) -> str:
    return sha256_bytes(np.ascontiguousarray(rows, dtype="<i8").tobytes())


@dataclass(frozen=True, eq=False)
class Split:
    """Row positions (in the dataset frame) of the training part and the reserved test part."""

    train_rows: Array
    test_rows: Array  # reserved by index only; never read in this step
    stratified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.train_rows) + len(self.test_rows)),
            "train_rows": int(len(self.train_rows)),
            "test_rows": int(len(self.test_rows)),
            "stratified": self.stratified,
        }

    def fingerprints(self) -> dict[str, Any]:
        """Counts plus a SHA-256 of each part's row positions (positions only, never row data).

        The test fingerprint is what lets a later step recognise that two jobs used the same test
        rows (the cross-job reuse question is an open decision, BUILD_PLAN Step 11).
        """
        return {
            **self.to_dict(),
            "train_sha256": _positions_hash(self.train_rows),
            "test_sha256": _positions_hash(self.test_rows),
        }


@dataclass(frozen=True)
class ProgressEvent:
    """One step of a run: an algorithm starting, a fold finishing, an algorithm done or failed."""

    kind: str  # algorithm_started | fold_finished | algorithm_finished | algorithm_failed
    algorithm: str
    algorithm_index: int  # 1-based
    algorithm_count: int
    fold: int | None = None  # 1-based, for fold_finished
    folds: int | None = None
    seconds: float | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "algorithm": self.algorithm,
            "algorithm_index": self.algorithm_index,
            "algorithm_count": self.algorithm_count,
            "fold": self.fold,
            "folds": self.folds,
            "seconds": None if self.seconds is None else round(self.seconds, 3),
            "message": self.message,
        }


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True, eq=False)
class AlgorithmResult:
    key: str
    label: str
    params: dict[str, Any]
    status: str  # "ok" or "failed"
    error: dict[str, str] | None
    seconds: float
    fold_seconds: tuple[float, ...]
    fold_metrics: tuple[dict[str, float | None], ...]
    summary: dict[str, dict[str, Any]]
    oof: dict[str, Any] | None  # evaluation of the pooled out-of-fold predictions
    oof_predicted: Array | None  # kept for later steps; not part of to_dict
    oof_proba: Array | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.key,
            "label": self.label,
            "params": self.params,
            "status": self.status,
            "error": self.error,
            "seconds": round(self.seconds, 3),
            "fold_seconds": [round(s, 3) for s in self.fold_seconds],
            "fold_metrics": list(self.fold_metrics),
            "summary": self.summary,
            "oof": self.oof,
        }


@dataclass(frozen=True, eq=False)
class TrainingResult:
    task: TaskType
    target: str
    settings: TrainSettings
    split: Split
    folds_used: int
    warnings: tuple[str, ...]
    plan: ColumnPlan
    classes: tuple[str, ...] | None
    train_class_counts: dict[str, int] | None
    results: tuple[AlgorithmResult, ...]
    seconds: float

    @property
    def ok(self) -> bool:
        """True when every requested algorithm finished."""
        return all(r.status == "ok" for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_format": TRAIN_FORMAT,
            "task": self.task,
            "target": self.target,
            "settings": self.settings.to_dict(),
            "split": self.split.to_dict(),
            "folds": {"requested": self.settings.folds, "used": self.folds_used},
            "warnings": list(self.warnings),
            "columns": self.plan.to_dict(),
            "classes": None if self.classes is None else list(self.classes),
            "train_class_counts": self.train_class_counts,
            "results": [r.to_dict() for r in self.results],
            "seconds": round(self.seconds, 3),
        }


# --------------------------------------------------------------------------------------------
# Split and folds
# --------------------------------------------------------------------------------------------


def make_split(labels: Array, task: TaskType, settings: TrainSettings) -> Split:
    """Seeded holdout split (stratified for classification). Checks feasibility first.

    `labels` are the target values of every row (strings for classification, floats for
    regression); they are used only to place rows in the two parts.
    """
    n = len(labels)
    n_test = math.ceil(settings.test_fraction * n)
    n_train = n - n_test
    rows: Array = np.arange(n)
    if n_test < 1 or n_train < 2:
        raise TrainingError(
            f"{n} rows are too few for a {settings.test_fraction:.0%} test part; add rows or "
            "lower the test fraction"
        )
    if task == "classification":
        counts = _count(labels)
        tiny = sorted(c for c, k in counts.items() if k < 2)
        if tiny:
            raise TrainingError(
                f"class {tiny[0]!r} has fewer than 2 rows, so a stratified split is impossible; "
                "remove it from the data or choose another target"
            )
        if n_test < len(counts) or n_train < len(counts):
            raise TrainingError(
                f"the {settings.test_fraction:.0%} test part ({n_test} rows) cannot hold all "
                f"{len(counts)} classes; use more rows, fewer classes, or a larger test fraction"
            )
        train, test = train_test_split(
            rows,
            test_size=settings.test_fraction,
            stratify=labels,
            shuffle=True,
            random_state=settings.seed,
        )
        short = sorted(c for c, k in _count(labels[train]).items() if k < 2)
        if short:
            raise TrainingError(
                f"after the holdout, class {short[0]!r} has fewer than 2 training rows, which "
                "cross-validation needs; use more data for that class, merge or remove it, or "
                "lower the test fraction"
            )
        return Split(np.sort(train), np.sort(test), stratified=True)
    train, test = train_test_split(
        rows, test_size=settings.test_fraction, shuffle=True, random_state=settings.seed
    )
    return Split(np.sort(train), np.sort(test), stratified=False)


def _count(labels: Array) -> dict[str, int]:
    values, counts = np.unique(labels.astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts, strict=True)}


def plan_folds(train_labels: Array, task: TaskType, requested: int) -> tuple[int, tuple[str, ...]]:
    """The number of folds to use and any warning. Classification: at most the smallest class."""
    if task == "regression":
        if len(train_labels) < requested:
            raise TrainingError(
                f"{len(train_labels)} training rows are too few for {requested}-fold "
                "cross-validation; use more rows or fewer folds"
            )
        return requested, ()
    counts = _count(train_labels)
    smallest_label, smallest = min(counts.items(), key=lambda item: (item[1], item[0]))
    if smallest < MIN_FOLDS:
        raise TrainingError(
            f"class {smallest_label!r} has only {smallest} training row(s); cross-validation "
            f"needs at least {MIN_FOLDS}"
        )
    if smallest < requested:
        return smallest, (
            f"cross-validation folds reduced from {requested} to {smallest}: the smallest class "
            f"({smallest_label!r}) has only {smallest} training rows",
        )
    return requested, ()


def make_folds(
    train_labels: Array, task: TaskType, folds: int, seed: int
) -> list[tuple[Array, Array]]:
    """(fit rows, validation rows) per fold, as positions within the training part."""
    rows: Array = np.arange(len(train_labels))
    if task == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        pairs = splitter.split(rows, train_labels)
    else:
        pairs = KFold(n_splits=folds, shuffle=True, random_state=seed).split(rows)
    return [(np.asarray(fit), np.asarray(val)) for fit, val in pairs]


# --------------------------------------------------------------------------------------------
# One fold, and the runner
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FoldFit:
    """A pipeline fitted on one fold's training rows, with its validation-row predictions."""

    pipeline: Pipeline
    predicted: Array
    proba: Array | None


def fit_fold(
    template: Pipeline,
    x_fit: pd.DataFrame,
    y_fit: Array,
    x_val: pd.DataFrame,
    classes: Sequence[str] | None,
) -> FoldFit:
    """Fit a fresh clone of the pipeline on the fold's training rows only, then predict the rest.

    The preprocessing inside the pipeline is fitted here, on `x_fit` alone.
    """
    pipeline: Pipeline = clone(template)
    pipeline.fit(x_fit, y_fit)
    predicted: Array = pipeline.predict(x_val)
    proba: Array | None = None
    if classes is not None:
        if list(pipeline.classes_) != list(classes):
            raise TrainingError("a fold's training rows do not contain every class")
        proba = pipeline.predict_proba(x_val)
    return FoldFit(pipeline, predicted, proba)


def _make_estimator(spec: AlgorithmSpec, params: Mapping[str, Any], seed: int) -> Any:
    return spec.build(params, seed)


def target_values(frame: pd.DataFrame, config: TaskConfig) -> Array:
    column = frame[[config.target]]
    if config.task == "classification":
        labels: Array = column[config.target].astype(str).to_numpy(dtype=object)
        return labels
    values: Array = text_to_number(column)[:, 0]
    return values


def _error_info(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)[:ERROR_MESSAGE_LIMIT]}


def run_training(
    frame: pd.DataFrame,
    config: TaskConfig,
    requests: Sequence[AlgorithmRequest],
    settings: TrainSettings = DEFAULT_TRAIN_SETTINGS,
    progress: ProgressCallback | None = None,
    split: Split | None = None,
) -> TrainingResult:
    """Cross-validate the requested algorithms on the training part of `frame`.

    `frame` is a text frame (`read_text_frame`). Pass a `split` to reuse a given holdout; otherwise
    one is made from the settings. Raises `TrainingError` (or `AlgorithmError` for bad algorithm
    names or parameters) before any training starts when the run cannot be carried out.
    """
    started = time.perf_counter()
    settings.validate()
    if not requests:
        raise TrainingError("no algorithms requested")
    keys = [r.key for r in requests]
    if len(set(keys)) != len(keys):
        raise TrainingError(f"an algorithm was requested more than once: {sorted(keys)}")
    specs = [get_algorithm(config.task, r.key) for r in requests]
    params = [s.validate_params(r.params) for s, r in zip(specs, requests, strict=True)]
    if config.target not in frame.columns:
        raise TrainingError(f"the target {config.target!r} is not a column of the data")

    # Pre-flight on the whole dataset: the same task checks as Step 5 (they look at all labels,
    # because feasibility of the split depends on them). Nothing here influences fitting.
    full_profile = profile_frame(frame)
    counts = class_counts(frame, config.target) if config.task == "classification" else {}
    report = check_task_config(full_profile, counts, config)
    if not report.ok:
        problems = "; ".join(
            f"{f.code}: {f.message}" for f in report.findings if f.severity == "error"
        )
        raise TrainingError(f"the task configuration is not usable: {problems}")
    warnings = [f"{f.code}: {f.message}" for f in report.findings if f.severity == "warning"]

    all_targets = target_values(frame, config)
    if split is None:
        split = make_split(all_targets, config.task, settings)
    elif len(split.train_rows) + len(split.test_rows) != len(frame):
        raise TrainingError("the given split does not match the data")
    # From here on only the training part is used. The test rows are not indexed again.
    train_frame = frame.iloc[split.train_rows].reset_index(drop=True)
    y_train = all_targets[split.train_rows]
    del all_targets

    plan = build_column_plan(profile_frame(train_frame), config)
    classification = config.task == "classification"
    classes = tuple(sorted(_count(y_train))) if classification else None
    train_counts = _count(y_train) if classification else None
    folds_used, fold_warnings = plan_folds(y_train, config.task, settings.folds)
    warnings.extend(fold_warnings)
    folds = make_folds(y_train, config.task, folds_used, settings.seed)
    x_train = train_frame[list(plan.used)]
    metric_names = CLASSIFICATION_METRICS if classification else REGRESSION_METRICS

    results: list[AlgorithmResult] = []
    total = len(specs)
    for index, (spec, algo_params) in enumerate(zip(specs, params, strict=True), start=1):

        def emit(
            kind: str,
            *,
            fold: int | None = None,
            seconds: float | None = None,
            message: str | None = None,
            _key: str = spec.key,
            _index: int = index,
        ) -> None:
            if progress is not None:
                progress(
                    ProgressEvent(kind, _key, _index, total, fold, folds_used, seconds, message)
                )

        emit("algorithm_started")
        began = time.perf_counter()
        fold_seconds: list[float] = []
        try:
            template = build_pipeline(
                plan,
                _make_estimator(spec, algo_params, settings.seed),
                scale_numeric=spec.scale_numeric,
            )
            oof_predicted: Array = np.empty(
                len(y_train), dtype=object if classification else "float64"
            )
            oof_proba: Array | None = (
                np.full((len(y_train), len(classes)), np.nan) if classes is not None else None
            )
            fold_scalars: list[dict[str, float | None]] = []
            for number, (fit_rows, val_rows) in enumerate(folds, start=1):
                fold_began = time.perf_counter()
                fitted = fit_fold(
                    template,
                    x_train.iloc[fit_rows],
                    y_train[fit_rows],
                    x_train.iloc[val_rows],
                    classes,
                )
                oof_predicted[val_rows] = fitted.predicted
                if oof_proba is not None and fitted.proba is not None:
                    oof_proba[val_rows] = fitted.proba
                if classes is not None:
                    scalars, _ = classification_scalars(
                        y_train[val_rows], fitted.predicted, fitted.proba, classes
                    )
                else:
                    scalars, _ = regression_scalars(y_train[val_rows], fitted.predicted)
                fold_scalars.append(scalars)
                fold_seconds.append(time.perf_counter() - fold_began)
                emit("fold_finished", fold=number, seconds=fold_seconds[-1])
            if classes is not None:
                oof = evaluate_classification(y_train, oof_predicted, oof_proba, classes)
            else:
                oof = evaluate_regression(y_train, oof_predicted, split.train_rows)
            elapsed = time.perf_counter() - began
            results.append(
                AlgorithmResult(
                    spec.key,
                    spec.label,
                    algo_params,
                    "ok",
                    None,
                    elapsed,
                    tuple(fold_seconds),
                    tuple(fold_scalars),
                    summarize_folds(fold_scalars, metric_names),
                    oof,
                    oof_predicted,
                    oof_proba,
                )
            )
            emit("algorithm_finished", seconds=elapsed)
        except Exception as error:  # one algorithm failing must not stop the others
            elapsed = time.perf_counter() - began
            info = _error_info(error)
            results.append(
                AlgorithmResult(
                    spec.key,
                    spec.label,
                    algo_params,
                    "failed",
                    info,
                    elapsed,
                    tuple(fold_seconds),
                    (),
                    {},
                    None,
                    None,
                    None,
                )
            )
            emit("algorithm_failed", seconds=elapsed, message=f"{info['type']}: {info['message']}")

    return TrainingResult(
        task=config.task,
        target=config.target,
        settings=settings,
        split=split,
        folds_used=folds_used,
        warnings=tuple(warnings),
        plan=plan,
        classes=classes,
        train_class_counts=train_counts,
        results=tuple(results),
        seconds=time.perf_counter() - started,
    )
