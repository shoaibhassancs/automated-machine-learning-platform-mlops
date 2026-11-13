"""Cross-validation comparison of tracked jobs (BUILD_PLAN Step 15).

Pure logic over `LoggedJob` records (plain data from `tracking.read_job`); no MLflow import, no
data, no model. **Cross-validation numbers only**: the comparison never contains, ranks by or
highlights a test result (a test pins this). Jobs may differ in dataset, seed, folds or test
fraction; that is allowed but every difference is reported, because CV scores from different splits
are not directly comparable. Nothing here says anything about reuse of the held-out test part
(an open decision).
"""

from __future__ import annotations

from typing import Any

from automl_ops.selection import ALLOWED_METRICS, DEFAULT_METRIC, metric_direction
from automl_ops.tracking import LoggedJob


class CompareError(ValueError):
    """The jobs cannot be compared (different task or target, a bad metric, fewer than two)."""


# What decides whether two jobs' CV numbers come from the same splits.
SPLIT_FACTORS = ("dataset_sha256", "seed", "folds_used", "test_fraction")


def _factor(job: LoggedJob, name: str) -> str | None:
    if name == "dataset_sha256":
        return job.tags.get("platform.dataset_sha256")
    if name == "seed":
        return job.tags.get("platform.seed")
    return job.params.get(name)


def default_metric(jobs: list[LoggedJob], task: str) -> str:
    """The metric the jobs were selected on if they all agree, otherwise the task default."""
    chosen = {j.tags.get("platform.selection_metric") for j in jobs} - {None}
    if len(chosen) == 1:
        return str(next(iter(chosen)))
    return DEFAULT_METRIC[task]


def compare_jobs(jobs: list[LoggedJob], metric: str | None = None) -> dict[str, Any]:
    """Side-by-side CV scores of every algorithm of every job, ranked by the chosen metric."""
    if len(jobs) < 2:
        raise CompareError("compare needs at least two jobs")
    keys = [j.job_key for j in jobs]
    if len(set(keys)) != len(keys):
        raise CompareError("the same job was given twice")
    tasks = {j.tags.get("platform.task") for j in jobs}
    targets = {j.tags.get("platform.target") for j in jobs}
    if len(tasks) != 1 or len(targets) != 1:
        raise CompareError(
            "jobs with a different task or target cannot be compared "
            f"(tasks: {sorted(map(str, tasks))}, targets: {sorted(map(str, targets))})"
        )
    task = str(next(iter(tasks)))
    target = str(next(iter(targets)))
    metric = metric or default_metric(jobs, task)
    if metric not in ALLOWED_METRICS[task]:
        raise CompareError(
            f"{metric!r} is not a {task} metric (use one of {ALLOWED_METRICS[task]})"
        )
    direction = metric_direction(metric)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        selected = job.tags.get("platform.selected_algorithm")
        for name, algo in sorted(job.algorithms.items()):
            rows.append(
                {
                    "job_key": job.job_key,
                    "algorithm": name,
                    "status": algo.tags.get("platform.status", "?"),
                    "selected_in_job": name == selected,
                    "cv_mean": algo.metrics.get(f"cv_{metric}_mean"),
                    "cv_std": algo.metrics.get(f"cv_{metric}_std"),
                    "oof": algo.metrics.get(f"oof_{metric}"),
                    "params": {k[6:]: v for k, v in sorted(algo.params.items())},
                    "error": algo.tags.get("platform.error_type") or None,
                }
            )
    scored = sorted(
        (r for r in rows if r["cv_mean"] is not None),
        key=lambda r: (r["cv_mean"] if direction == "lower" else -r["cv_mean"], r["job_key"]),
    )
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    for row in rows:
        row.setdefault("rank", None)
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["job_key"], r["algorithm"]))

    differences: list[str] = []
    for factor in SPLIT_FACTORS:
        values = {j.job_key: _factor(j, factor) for j in jobs}
        if len(set(values.values())) > 1:
            shown = ", ".join(f"{k}: {v}" for k, v in values.items())
            differences.append(f"{factor} differs ({shown})")
    split_note = None
    if differences:
        split_note = (
            "These jobs did not use identical data and splits, so their cross-validation scores "
            "are not directly comparable: " + "; ".join(differences)
        )
    library_differences = sorted(
        name
        for name in {n for j in jobs for n in j.libraries}
        if len({j.libraries.get(name) for j in jobs}) > 1
    )
    parameter_differences: dict[str, list[str]] = {}
    for name in sorted({a for j in jobs for a in j.algorithms}):
        present = [j.algorithms[name].params for j in jobs if name in j.algorithms]
        different = sorted(
            {p for params in present for p in params if len({q.get(p) for q in present}) > 1}
        )
        if different:
            parameter_differences[name] = [p[6:] for p in different]
    return {
        "task": task,
        "target": target,
        "metric": metric,
        "direction": direction,
        "basis": "cross-validation on each job's training part only (no test scores)",
        "jobs": [
            {
                "job_key": j.job_key,
                "kind": j.kind,
                "dataset_ref": j.tags.get("platform.dataset_ref"),
                "dataset_sha256": j.tags.get("platform.dataset_sha256"),
                "seed": j.tags.get("platform.seed"),
                "folds_used": j.params.get("folds_used"),
                "test_fraction": j.params.get("test_fraction"),
                "selected_algorithm": j.tags.get("platform.selected_algorithm"),
            }
            for j in jobs
        ],
        "rows": rows,
        "split_differences": differences,
        "split_note": split_note,
        "parameter_differences": parameter_differences,
        "library_differences": library_differences,
    }
