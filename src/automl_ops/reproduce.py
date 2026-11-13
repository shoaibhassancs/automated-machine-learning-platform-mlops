"""Reproducibility check of a tracked job (BUILD_PLAN Step 16).

What it does: rebuild the job's configuration from its MLflow record, re-run **cross-validation**
through `train.run_training` (the one training implementation) on the same dataset version, and
compare the result with what was logged.

What it establishes: the training/CV pipeline gives the recorded numbers again, on this machine and
these library versions (differences in the environment are reported, never hidden).

What it does **not** do: it does not reproduce the final packaged-model refit, and it never reads or
scores the held-out test part. For a finalized job it only checks, as JSON, that the recorded test
result in MLflow equals the sealed package's own `test_result.json`, and that the re-made split has
the same test-row fingerprint (a hash of row *positions*; no row is read). It writes nothing: no
MLflow run, tag or artifact, no dataset, no package. This module imports neither `mlflow` nor
`finalize`; a test enforces both.

The reference for the numbers is the full-precision `training.json` stored with the job. The MLflow
metric values are cross-checked against it separately.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from automl_ops.package import (
    MANIFEST_FILE,
    TEST_RESULT_FILE,
    PackageError,
    package_path,
    verify_package,
)
from automl_ops.selection import freeze_selection, select_model
from automl_ops.task import TaskConfig
from automl_ops.tracking import LoggedJob
from automl_ops.train import (
    AlgorithmRequest,
    ProgressCallback,
    TrainingResult,
    TrainSettings,
    run_training,
)

DEFAULT_ABS_TOLERANCE = 1e-9
DEFAULT_REL_TOLERANCE = 1e-6  # pass if |new - logged| <= abs + rel * |logged|  (provisional)
TIMING_KEYS = frozenset({"seconds", "fold_seconds"})  # wall-clock times are never compared
MAX_LISTED = 8


class ReproductionError(ValueError):
    """The check cannot be run (altered dataset, incomplete job, ...). Not a failed comparison."""


@dataclass
class _Walk:
    """Compares two JSON-like structures; numbers within tolerance, everything else exactly."""

    rel: float
    abs_tol: float = DEFAULT_ABS_TOLERANCE
    mismatches: list[str] = field(default_factory=list)
    max_deviation: float = 0.0
    compared: int = 0
    identical: bool = True

    def number(self, path: str, logged: float, new: float) -> None:
        self.compared += 1
        delta = abs(new - logged)
        if delta != 0:
            self.identical = False
        if math.isfinite(delta):
            self.max_deviation = max(self.max_deviation, delta)
        if not delta <= self.abs_tol + self.rel * abs(logged):
            self.mismatches.append(f"{path}: logged {logged!r}, recomputed {new!r}")

    def walk(self, path: str, logged: Any, new: Any, skip: frozenset[str] = frozenset()) -> None:
        if isinstance(logged, bool) or isinstance(new, bool) or logged is None or new is None:
            self._exact(path, logged, new)
        elif isinstance(logged, int | float) and isinstance(new, int | float):
            self.number(path, float(logged), float(new))
        elif isinstance(logged, dict) and isinstance(new, dict):
            for key in sorted(set(logged) | set(new)):
                if key in skip:
                    continue
                if key not in logged or key not in new:
                    self.identical = False
                    where = "recomputed run" if key not in logged else "record"
                    self.mismatches.append(f"{path}.{key}: missing from the {where}")
                    continue
                self.walk(f"{path}.{key}", logged[key], new[key], skip)
        elif isinstance(logged, list) and isinstance(new, list):
            if len(logged) != len(new):
                self.identical = False
                self.mismatches.append(f"{path}: {len(logged)} entries logged, {len(new)} now")
                return
            for i, (a, b) in enumerate(zip(logged, new, strict=True)):
                self.walk(f"{path}[{i}]", a, b, skip)
        else:
            self._exact(path, logged, new)

    def _exact(self, path: str, logged: Any, new: Any) -> None:
        self.compared += 1
        if logged != new or type(logged) is not type(new):
            self.identical = False
            self.mismatches.append(f"{path}: logged {logged!r}, recomputed {new!r}")


@dataclass(frozen=True)
class CheckItem:
    name: str
    status: str  # "pass", "fail" or "skipped"
    detail: str
    mismatches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "mismatches": list(self.mismatches[:MAX_LISTED]),
            "more_mismatches": max(0, len(self.mismatches) - MAX_LISTED),
        }


@dataclass(frozen=True)
class ReproductionReport:
    job_key: str
    kind: str
    ok: bool
    verdict: str
    tolerance: dict[str, float]
    items: tuple[CheckItem, ...]
    max_deviation: float
    bit_identical: bool
    environment_same: bool
    environment_differences: tuple[str, ...]
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_key": self.job_key,
            "kind": self.kind,
            "ok": self.ok,
            "verdict": self.verdict,
            "tolerance": self.tolerance,
            "max_deviation": self.max_deviation,
            "bit_identical": self.bit_identical,
            "environment_same": self.environment_same,
            "environment_differences": list(self.environment_differences),
            "items": [i.to_dict() for i in self.items],
            "scope": self.scope,
        }


SCOPE = (
    "Verifies the training/cross-validation pipeline and its recorded results. It does not "
    "reproduce the final packaged-model refit and does not read or re-score the held-out test "
    "part; a recorded test result is only compared, as JSON, with the sealed package's copy."
)


def sealed_test_result(models_root: Path, package_id: str) -> dict[str, Any]:
    """The sealed package's `test_result.json`, read as JSON after the seal hashes check out.
    The model file is not loaded. Raises `ReproductionError` if it cannot be trusted."""
    try:
        root = package_path(models_root, package_id)
        check = verify_package(models_root, package_id, deep=False)
    except PackageError as error:
        raise ReproductionError(str(error)) from error
    if not check.ok:
        raise ReproductionError(f"package {package_id} failed verification: {check.problems}")
    loaded: dict[str, Any] = json.loads((root / TEST_RESULT_FILE).read_text(encoding="utf-8"))
    return loaded


def sealed_manifest_selection_sha(models_root: Path, package_id: str) -> str | None:
    """The selection hash recorded in a sealed package's manifest (JSON only)."""
    try:
        loaded = json.loads(
            (package_path(models_root, package_id) / MANIFEST_FILE).read_text("utf-8")
        )
    except (OSError, ValueError, PackageError):
        return None
    value = loaded.get("selection_sha256")
    return str(value) if value else None


def derive_config(job: LoggedJob, frame_columns: list[str]) -> TaskConfig:
    """The task configuration of a logged job. `train` logs the column plan, not `--exclude`
    itself, so the excluded columns are the dataset columns that are neither the target nor in the
    logged plan. The derivation is checked afterwards: a wrong one produces a different plan."""
    plan = job.training["columns"]
    in_plan = set(plan["numeric"]) | set(plan["categorical"]) | {d["name"] for d in plan["dropped"]}
    target = str(job.training["target"])
    excluded = tuple(c for c in frame_columns if c != target and c not in in_plan)
    return TaskConfig(target=target, task=job.training["task"], excluded=excluded)


def _requests(job: LoggedJob) -> list[AlgorithmRequest]:
    return [AlgorithmRequest(r["algorithm"], dict(r["params"])) for r in job.training["results"]]


def _record_consistency(job: LoggedJob, rel: float) -> CheckItem:
    """MLflow metric values against the full-precision `training.json` of the same job. Values may
    be stored or displayed at lower precision, so this uses the tolerance, not equality."""
    walk = _Walk(rel)
    for result in job.training["results"]:
        logged = job.algorithms.get(result["algorithm"])
        if logged is None:
            walk.mismatches.append(f"{result['algorithm']}: no MLflow run")
            continue
        expected: dict[str, float] = {}
        for name, entry in (result.get("summary") or {}).items():
            for stat in ("mean", "std"):
                if isinstance(entry.get(stat), int | float):
                    expected[f"cv_{name}_{stat}"] = entry[stat]
        for name, value in ((result.get("oof") or {}).get("metrics") or {}).items():
            if isinstance(value, int | float):
                expected[f"oof_{name}"] = value
        for metric, value in expected.items():
            if metric not in logged.metrics:
                walk.mismatches.append(f"{result['algorithm']}.{metric}: missing in MLflow")
            else:
                walk.number(f"{result['algorithm']}.{metric}", value, logged.metrics[metric])
        for fold, scores in enumerate(result.get("fold_metrics") or []):
            for name, value in scores.items():
                if not isinstance(value, int | float):
                    continue
                series = logged.fold_metrics.get(f"fold_{name}", [])
                if fold >= len(series):
                    walk.mismatches.append(f"{result['algorithm']}.fold_{name}[{fold}]: missing")
                else:
                    walk.number(f"{result['algorithm']}.fold_{name}[{fold}]", value, series[fold])
    if walk.mismatches:
        return CheckItem(
            "record consistency",
            "fail",
            "the MLflow metrics differ from the job's own training.json",
            tuple(walk.mismatches),
        )
    return CheckItem(
        "record consistency",
        "pass",
        f"{walk.compared} MLflow metric values agree with training.json "
        f"(largest difference {walk.max_deviation:.3g})",
    )


def _compare_training(
    job: LoggedJob, new: dict[str, Any], rel: float
) -> tuple[list[CheckItem], _Walk]:
    """Logged `training.json` against the recomputed one: configuration first, then each algorithm.
    Timings are skipped; everything else (parameters, folds, per-fold, summary and pooled
    out-of-fold metrics) must agree within the tolerance."""
    total = _Walk(rel)
    items: list[CheckItem] = []
    logged = job.training
    config = _Walk(rel)
    config.walk(
        "config",
        {k: v for k, v in logged.items() if k != "results"},
        {k: v for k, v in new.items() if k != "results"},
        TIMING_KEYS,
    )
    items.append(_item("configuration, split sizes and column plan", config))
    total.mismatches += config.mismatches
    total.identical &= config.identical
    total.max_deviation = max(total.max_deviation, config.max_deviation)
    by_key = {r["algorithm"]: r for r in new["results"]}
    for old in logged["results"]:
        key = old["algorithm"]
        walk = _Walk(rel)
        fresh = by_key.get(key)
        if fresh is None:
            walk.mismatches.append(f"{key}: not run")
        else:
            walk.walk(key, old, fresh, TIMING_KEYS)
        word = "ok" if old["status"] == "ok" else f"failed ({(old.get('error') or {}).get('type')})"
        items.append(_item(f"algorithm {key} [{word}]", walk))
        total.mismatches += walk.mismatches
        total.identical &= walk.identical
        total.max_deviation = max(total.max_deviation, walk.max_deviation)
    return items, total


def _item(name: str, walk: _Walk) -> CheckItem:
    if walk.mismatches:
        return CheckItem(
            name, "fail", f"{len(walk.mismatches)} difference(s)", tuple(walk.mismatches)
        )
    note = "identical" if walk.identical else f"largest difference {walk.max_deviation:.3g}"
    return CheckItem(name, "pass", f"{walk.compared} values compared, {note}")


def _selection_items(
    job: LoggedJob,
    training: TrainingResult,
    package_test: dict[str, Any] | None,
    package_selection_sha: str | None,
    package_checked: bool,
) -> tuple[list[CheckItem], bool]:
    """Checks that only finalized jobs have. Returns the items and whether the selection hash was
    bit-identical. Nothing here touches test rows."""
    items: list[CheckItem] = []
    metric = job.tags.get("platform.selection_metric")
    report = select_model(training, metric)
    frozen = freeze_selection(training, report)
    logged_selection = job.selection or {}
    logged_order = [r["algorithm"] for r in logged_selection.get("ranking", [])]
    new_order = [r["algorithm"] for r in report.data["ranking"]]
    same_choice = (
        logged_order == new_order
        and job.tags.get("platform.selected_algorithm") == report.selected_key
    )
    items.append(
        CheckItem(
            "selection (from cross-validation only)",
            "pass" if same_choice else "fail",
            f"selected {report.selected_key}; ranking {new_order}"
            if same_choice
            else f"logged {logged_order} / {job.tags.get('platform.selected_algorithm')}, "
            f"recomputed {new_order} / {report.selected_key}",
        )
    )
    logged_hash = job.tags.get("platform.selection_sha256")
    identical = frozen.sha256 == logged_hash
    items.append(
        CheckItem(
            "frozen-selection hash",
            "pass",
            "identical"
            if identical
            else "differs (informational: the hash covers every metric digit, and the metric "
            "values themselves passed the tolerance check above)",
        )
    )
    new_test_hash = training.split.fingerprints()["test_sha256"]
    logged_test_hash = job.tags.get("platform.test_rows_sha256")
    items.append(
        CheckItem(
            "split fingerprint (row positions only)",
            "pass" if new_test_hash == logged_test_hash else "fail",
            "the re-made split holds the same test-row positions; no row was read"
            if new_test_hash == logged_test_hash
            else "the re-made split differs from the logged one",
        )
    )
    logged_test = next(
        (a.test_result for a in job.algorithms.values() if a.test_result is not None), None
    )
    if logged_test is None:
        items.append(CheckItem("recorded test result", "fail", "no test result is logged"))
    elif not package_checked:
        items.append(
            CheckItem(
                "recorded test result vs sealed package",
                "skipped",
                "NOT CHECKED: the sealed package was not available (JSON comparison only)",
            )
        )
    else:
        same = logged_test == package_test
        items.append(
            CheckItem(
                "recorded test result vs sealed package",
                "pass" if same else "fail",
                "the MLflow copy equals the sealed package's test_result.json (compared as JSON; "
                "nothing was re-scored)"
                if same
                else "the MLflow copy differs from the sealed package's test_result.json",
            )
        )
        sealed_hash = package_selection_sha
        if sealed_hash is not None:
            items.append(
                CheckItem(
                    "selection hash vs sealed package",
                    "pass" if sealed_hash == logged_hash else "fail",
                    "the logged hash equals the package manifest's"
                    if sealed_hash == logged_hash
                    else "the logged hash differs from the package manifest's",
                )
            )
        metrics = (logged_test.get("evaluation") or {}).get("metrics") or {}
        walk = _Walk(DEFAULT_REL_TOLERANCE)
        selected = job.algorithms.get(job.tags.get("platform.selected_algorithm", ""))
        for name, value in metrics.items():
            if isinstance(value, int | float):
                got = (selected.metrics if selected else {}).get(f"test_{name}")
                if got is None:
                    walk.mismatches.append(f"test_{name}: missing in MLflow")
                else:
                    walk.number(f"test_{name}", value, got)
        items.append(_item("MLflow test metrics vs recorded test result", walk))
    return items, identical


def reproduce_job(
    job: LoggedJob,
    frame: pd.DataFrame,
    dataset_sha256: str,
    current_libraries: dict[str, str],
    *,
    package_test_result: dict[str, Any] | None = None,
    package_selection_sha256: str | None = None,
    package_checked: bool = False,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
    progress: ProgressCallback | None = None,
) -> ReproductionReport:
    """Re-run the job's cross-validation and compare with the record. See the module docstring.

    `dataset_sha256` is the verified hash of the loaded version; it must equal the logged one."""
    if job.state != "complete":
        raise ReproductionError(f"{job.job_key} is not completely tracked (state: {job.state})")
    logged_sha = job.tags.get("platform.dataset_sha256")
    if dataset_sha256 != logged_sha:
        raise ReproductionError(
            f"the dataset version has SHA-256 {dataset_sha256[:12]}..., the job used "
            f"{str(logged_sha)[:12]}...; it cannot be reproduced on different data"
        )
    if rel_tolerance < 0:
        raise ReproductionError("the tolerance cannot be negative")

    items = [_record_consistency(job, rel_tolerance)]
    config = derive_config(job, [str(c) for c in frame.columns])
    settings = TrainSettings(**job.training["settings"])
    training = run_training(frame, config, _requests(job), settings, progress)
    fresh = json.loads(json.dumps(training.to_dict()))
    compared, total = _compare_training(job, fresh, rel_tolerance)
    items += compared
    bit_identical = total.identical and not total.mismatches
    if job.kind == "final":
        extra, same_hash = _selection_items(
            job, training, package_test_result, package_selection_sha256, package_checked
        )
        items += extra
        bit_identical = bit_identical and same_hash

    logged_libs = job.libraries
    differences = tuple(
        f"{name}: logged {logged_libs.get(name)}, now {current_libraries.get(name)}"
        for name in sorted(set(logged_libs) | set(current_libraries))
        if logged_libs.get(name) != current_libraries.get(name)
    )
    ok = all(i.status != "fail" for i in items)
    if not ok:
        failed = [i.name for i in items if i.status == "fail"]
        verdict = "NOT REPRODUCED: " + "; ".join(failed)
    elif differences:
        verdict = "REPRODUCED under a different environment (library versions differ)"
    else:
        verdict = "REPRODUCED"
    return ReproductionReport(
        job_key=job.job_key,
        kind=job.kind,
        ok=ok,
        verdict=verdict,
        tolerance={"relative": rel_tolerance, "absolute": DEFAULT_ABS_TOLERANCE},
        items=tuple(items),
        max_deviation=total.max_deviation,
        bit_identical=bit_identical and ok,
        environment_same=not differences,
        environment_differences=differences,
        scope=SCOPE,
    )
