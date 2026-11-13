"""Model selection and the frozen selection (BUILD_PLAN Step 11).

`select_model` ranks the algorithms of one `TrainingResult` using **cross-validation results and
out-of-fold predictions only**. It receives no data frame and no row positions of the test part, so
no test information can reach the ranking, the tie-breaking or the frozen selection.
`freeze_selection` turns the result into an immutable `FrozenSelection` carrying a SHA-256
that `finalize.py` verifies before it touches the test part.

**Primary metric.** Macro F1 (classification) or RMSE (regression) unless another metric from the
closed list for the task is chosen. The ranking value is the mean of the metric over the folds.
RMSE and MAE are lower-is-better; every other metric is higher-is-better.

**Eligibility.** A candidate is ranked only if it finished, produced out-of-fold predictions, and
the primary metric is defined in every fold and on the pooled out-of-fold rows. Everything else is
listed as excluded, with the reason.

**Ties.** Values within `TIE_TOLERANCE` (1e-9) of the best are tied. Ties are broken, in order, by
the pooled out-of-fold value of the primary metric, a secondary metric (accuracy, or MAE; RMSE when
MAE is the primary), the smaller standard deviation over folds, and finally the fixed catalogue
order. Fit time is never used because it is not reproducible.

**Positive class.** For two classes it is `classes[1]` (the second of the sorted labels). It is
recorded in the report and in the frozen selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from automl_ops.algorithms import algorithms_for, get_algorithm
from automl_ops.evaluate import CLASSIFICATION_METRICS, REGRESSION_METRICS
from automl_ops.hashing import canonical_json, sha256_text
from automl_ops.task import TaskType
from automl_ops.train import AlgorithmResult, TrainingResult

SELECTION_FORMAT = 1
TIE_TOLERANCE = 1e-9
DEFAULT_METRIC: dict[str, str] = {"classification": "f1_macro", "regression": "rmse"}
LOWER_IS_BETTER = frozenset({"rmse", "mae"})
ALLOWED_METRICS: dict[str, tuple[str, ...]] = {
    "classification": CLASSIFICATION_METRICS,
    "regression": REGRESSION_METRICS,
}


class SelectionError(ValueError):
    """Nothing can be selected, or a frozen selection no longer matches its hash."""


def metric_direction(metric: str) -> str:
    return "lower" if metric in LOWER_IS_BETTER else "higher"


def _secondary_metric(task: TaskType, primary: str) -> str:
    if task == "classification":
        return "f1_macro" if primary == "accuracy" else "accuracy"
    return "rmse" if primary == "mae" else "mae"


def _signed(metric: str, value: float | None) -> float:
    """A value where larger is always better; an undefined value is worse than any number."""
    if value is None:
        return float("-inf")
    return -value if metric in LOWER_IS_BETTER else value


def _within(best: float, value: float) -> bool:
    return value == best or best - value <= TIE_TOLERANCE


@dataclass(frozen=True)
class _Candidate:
    key: str
    label: str
    params: dict[str, Any]
    order: int
    mean: float
    std: float | None
    oof: float
    secondary: float | None


def _mean_std(result: AlgorithmResult, metric: str) -> tuple[float | None, float | None, bool]:
    entry = result.summary.get(metric)
    if not entry:
        return None, None, False
    complete = entry["folds_defined"] == entry["folds"] and entry["folds"] > 0
    return entry["mean"], entry["std"], complete


def _exclusion(result: AlgorithmResult, metric: str) -> str | None:
    if result.status != "ok":
        error = result.error or {}
        return f"the algorithm failed ({error.get('type', 'error')}: {error.get('message', '')})"
    if result.oof is None or result.oof_predicted is None:
        return "no out-of-fold predictions were produced"
    mean, _, complete = _mean_std(result, metric)
    if mean is None or not complete:
        entry = result.summary.get(metric, {})
        return (
            f"{metric} is undefined in {entry.get('folds', 0) - entry.get('folds_defined', 0)} "
            f"of {entry.get('folds', 0)} folds"
        )
    if result.oof["metrics"].get(metric) is None:
        why = result.oof.get("undefined", {}).get(metric, "undefined")
        return f"{metric} is undefined on the pooled out-of-fold rows ({why})"
    return None


def _pick(
    metric: str, secondary: str, group: list[_Candidate]
) -> tuple[_Candidate, str, list[str]]:
    steps: list[tuple[str, Any]] = [
        ("fold_mean", lambda c: _signed(metric, c.mean)),
        ("pooled_oof", lambda c: _signed(metric, c.oof)),
        ("secondary_metric", lambda c: _signed(secondary, c.secondary)),
        ("fold_std", lambda c: float("-inf") if c.std is None else -c.std),
        ("catalogue_order", lambda c: -float(c.order)),
    ]
    tied: list[str] = []
    for index, (name, value) in enumerate(steps):
        values = {c.key: value(c) for c in group}
        best = max(values.values())
        group = [c for c in group if _within(best, values[c.key])]
        if index == 0 and len(group) > 1:
            tied = sorted(c.key for c in group)
        if len(group) == 1:
            decided = "fold_mean" if index == 0 and not tied else f"tie-break: {name}"
            return group[0], decided, tied
    raise AssertionError("the catalogue order always separates candidates")  # pragma: no cover


@dataclass(frozen=True)
class SelectionReport:
    """The ranking and the reasons, as plain JSON-ready data (`data`)."""

    data: dict[str, Any]

    @property
    def selected_key(self) -> str:
        return str(self.data["selected"]["algorithm"])

    def to_dict(self) -> dict[str, Any]:
        return self.data


def select_model(training: TrainingResult, metric: str | None = None) -> SelectionReport:
    """Rank the algorithms of `training` and say why the winner won. See the module docstring."""
    task = training.task
    metric = metric or DEFAULT_METRIC[task]
    if metric not in ALLOWED_METRICS[task]:
        raise SelectionError(
            f"{metric!r} is not a {task} selection metric "
            f"(use one of {list(ALLOWED_METRICS[task])})"
        )
    secondary = _secondary_metric(task, metric)
    order = {a.key: i for i, a in enumerate(algorithms_for(task))}
    eligible: list[_Candidate] = []
    excluded: list[dict[str, str]] = []
    for result in training.results:
        reason = _exclusion(result, metric)
        if reason is not None:
            excluded.append({"algorithm": result.key, "reason": reason})
            continue
        mean, std, _ = _mean_std(result, metric)
        assert mean is not None and result.oof is not None
        sec_mean, _, _ = _mean_std(result, secondary)
        eligible.append(
            _Candidate(
                result.key,
                result.label,
                dict(result.params),
                order[result.key],
                mean,
                std,
                result.oof["metrics"][metric],
                sec_mean,
            )
        )
    if not eligible:
        why = "; ".join(f"{e['algorithm']}: {e['reason']}" for e in excluded)
        raise SelectionError(f"no algorithm can be selected ({why or 'none were trained'})")

    ranking: list[dict[str, Any]] = []
    remaining = list(eligible)
    while remaining:
        winner, decided, tied = _pick(metric, secondary, remaining)
        remaining.remove(winner)
        ranking.append(
            {
                "rank": len(ranking) + 1,
                "algorithm": winner.key,
                "label": winner.label,
                "params": winner.params,
                "metric_mean": winner.mean,
                "metric_std": winner.std,
                "pooled_oof": winner.oof,
                "secondary": {"metric": secondary, "mean": winner.secondary},
                "decided_by": decided,
                "tied_with": [k for k in tied if k != winner.key],
            }
        )

    best, runner = ranking[0], (ranking[1] if len(ranking) > 1 else None)
    warnings: list[str] = []
    margin: dict[str, Any] = {"runner_up": None, "difference": None, "within_fold_std": None}
    if runner is not None:
        difference = abs(best["metric_mean"] - runner["metric_mean"])
        spread = best["metric_std"]
        close = spread is not None and difference <= spread
        margin = {
            "runner_up": runner["algorithm"],
            "difference": difference,
            "within_fold_std": close,
        }
        if close:
            warnings.append(
                f"not a clear winner: the difference to {runner['label']} ({difference:.4g}) is "
                f"within the fold-to-fold standard deviation of the selected model ({spread:.4g})"
            )
    else:
        warnings.append(
            "selected by default: it is the only algorithm that completed"
            if excluded
            else "only one algorithm was trained, so there was nothing to compare"
        )
    best_oof = max(_signed(metric, c.oof) for c in eligible)
    rankings_agree = _within(best_oof, _signed(metric, best["pooled_oof"]))
    if not rankings_agree:
        warnings.append(
            "the pooled out-of-fold values would have ranked another algorithm first; the "
            "fold-mean ranking was used as documented"
        )

    direction = metric_direction(metric)
    explanation = (
        f"Selected {best['label']}: best mean {metric} over {training.folds_used} "
        f"cross-validation folds ({best['metric_mean']:.4f}, {direction} is better)"
    )
    if runner is not None:
        explanation += f", ahead of {runner['label']} ({runner['metric_mean']:.4f})"
    if best["decided_by"] != "fold_mean":
        explanation += (
            f". The top scores were tied within {TIE_TOLERANCE:g}; decided by "
            f"{best['decided_by'].removeprefix('tie-break: ').replace('_', ' ')}"
        )
    explanation += ". Only cross-validation results and out-of-fold predictions were used."

    positive = training.classes[1] if training.classes and len(training.classes) == 2 else None
    return SelectionReport(
        {
            "selection_format": SELECTION_FORMAT,
            "task": task,
            "metric": metric,
            "direction": direction,
            "secondary_metric": secondary,
            "tie_tolerance": TIE_TOLERANCE,
            "folds_used": training.folds_used,
            "classes": None if training.classes is None else list(training.classes),
            "positive_class": positive,
            "selected": {"algorithm": best["algorithm"], "label": best["label"]},
            "ranking": ranking,
            "excluded": excluded,
            "margin": margin,
            "rankings_agree": rankings_agree,
            "warnings": warnings,
            "explanation": explanation,
        }
    )


@dataclass(frozen=True)
class FrozenSelection:
    """An immutable record of the chosen configuration. `payload` is canonical JSON text and
    `sha256` its hash; `verify()` fails if either was altered."""

    payload: str
    sha256: str

    @property
    def data(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.payload)
        return loaded

    def verify(self) -> None:
        if sha256_text(self.payload) != self.sha256:
            raise SelectionError("the frozen selection does not match its SHA-256")


def freeze_selection(training: TrainingResult, report: SelectionReport) -> FrozenSelection:
    """Freeze the selected configuration: algorithm, validated parameters, seed, settings, column
    plan, class order, split fingerprints and the full ranking."""
    spec = get_algorithm(training.task, report.selected_key)
    chosen = next(r for r in training.results if r.key == report.selected_key)
    payload = canonical_json(
        {
            "selection_format": SELECTION_FORMAT,
            "task": training.task,
            "target": training.target,
            "algorithm": spec.key,
            "label": spec.label,
            "params": chosen.params,
            "scale_numeric": spec.scale_numeric,
            "seed": training.settings.seed,
            "settings": training.settings.to_dict(),
            "folds_used": training.folds_used,
            "plan": training.plan.to_dict(),
            "classes": report.data["classes"],
            "positive_class": report.data["positive_class"],
            "train_class_counts": training.train_class_counts,
            "split": training.split.fingerprints(),
            "selection": report.data,
        }
    )
    return FrozenSelection(payload, sha256_text(payload))
