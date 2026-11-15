"""Drift statistics and alert rules (P4-9, BUILD_PLAN Step 24): the one module that decides whether
logged traffic differs from a model's reference sample. Pure functions over plain values: no
database, no files, no model.

**Why not Evidently here.** The plan names Evidently with SciPy as the fallback, and says to
measure Evidently before installing it. Installing a package needs the owner's approval, which has
not been given, so this module implements the same kinds of tests with numpy only (no dependency
change). It is the single place to swap in Evidently later; callers only see `build_summary` and
`evaluate_rules`.

**Tests used** (all thresholds are provisional defaults in `Thresholds`, stored with every report):

- numeric feature, and regression predictions: the two-sample Kolmogorov-Smirnov test; drift when
  the asymptotic p-value is below `alpha` (0.05). With large samples a tiny shift becomes
  significant, so the statistic is shown beside the p-value.
- categorical feature, and predicted classes: the Jensen-Shannon distance (base 2, 0 to 1) between
  the category frequencies; drift when it is at least `js_distance` (0.1).
- a *dataset drift* share: the share of evaluated features that drifted; the
  `drifted_feature_share` rule fires at `feature_share` (0.5).
- `performance_drop`: the selection metric measured on labelled live predictions got worse than the
  stored offline value by more than `performance_drop` (10 % relative). It is evaluated only with
  at least `min_labelled` labels and is a comparison of two different measurements, so it is an
  alert to look at, not proof the model is wrong.

Drift means "inputs (or predictions) differ from the training sample", not "the model is wrong".
A window or reference smaller than `min_rows` gives status `insufficient_data` and no drift result.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

RULES = ("drifted_feature_share", "prediction_shift", "performance_drop")
SEVERITY = {
    "drifted_feature_share": "warning",
    "prediction_shift": "warning",
    "performance_drop": "critical",
}
LOWER_IS_BETTER = frozenset({"rmse", "mae"})
# Provisional fixed constants (owner-approved), not thresholds: the bootstrap that guards the
# performance rule against small-sample noise.
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_CONFIDENCE = 0.90
MIN_VALID_RESAMPLES = 100


class MonitorError(ValueError):
    """Invalid thresholds or inputs."""


@dataclass(frozen=True)
class Thresholds:
    alpha: float = 0.05
    js_distance: float = 0.1
    feature_share: float = 0.5
    min_rows: int = 30
    performance_drop: float = 0.10
    min_labelled: int = 30
    window: int = 1000

    def __post_init__(self) -> None:
        for name in ("alpha", "js_distance", "feature_share", "performance_drop"):
            value = getattr(self, name)
            if isinstance(value, bool) or not 0 < value <= 1:
                raise MonitorError(f"{name} must be greater than 0 and at most 1")
        for name, low, high in (
            ("min_rows", 5, 100_000),
            ("min_labelled", 1, 100_000),
            ("window", 5, 5000),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise MonitorError(f"{name} must be a whole number from {low} to {high}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------------
# The statistics
# --------------------------------------------------------------------------------------------


def ks_two_sample(reference: Array, current: Array) -> tuple[float, float]:
    """The two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value (Stephens'
    small-sample correction). Needs at least one value in each sample."""
    a, b = np.sort(reference), np.sort(current)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise MonitorError("the KS test needs at least one value in each sample")
    grid = np.concatenate([a, b])
    d = float(
        np.max(
            np.abs(
                np.searchsorted(a, grid, side="right") / n
                - np.searchsorted(b, grid, side="right") / m
            )
        )
    )
    effective = n * m / (n + m)
    lam = (math.sqrt(effective) + 0.12 + 0.11 / math.sqrt(effective)) * d
    if lam < 0.2:
        return d, 1.0
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def js_distance(reference: list[str], current: list[str]) -> float:
    """Jensen-Shannon distance (base 2, in [0, 1]) between two category frequency tables."""
    ref, cur = Counter(reference), Counter(current)
    categories = sorted(set(ref) | set(cur))
    p = np.array([ref[c] for c in categories], dtype=float)
    q = np.array([cur[c] for c in categories], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        raise MonitorError("the Jensen-Shannon distance needs values in both samples")
    p, q = p / p.sum(), q / q.sum()
    mid = (p + q) / 2

    def kl(x: Array, y: Array) -> float:
        keep = x > 0
        return float(np.sum(x[keep] * np.log2(x[keep] / y[keep])))

    return float(math.sqrt(max(0.0, 0.5 * kl(p, mid) + 0.5 * kl(q, mid))))


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


def _missing_share(values: list[Any]) -> float:
    return round(sum(v is None for v in values) / len(values), 6) if values else 0.0


def _feature(
    name: str, kind: str, reference: list[Any], current: list[Any], th: Thresholds
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "reference_missing_share": _missing_share(reference),
        "current_missing_share": _missing_share(current),
    }
    if kind == "numeric":
        ref = np.array([v for v in map(_number, reference) if v is not None])
        cur = np.array([v for v in map(_number, current) if v is not None])
        if len(ref) < 2 or len(cur) < 2:
            return {
                **base,
                "method": "ks",
                "status": "no_data",
                "drifted": False,
                "statistic": None,
                "p_value": None,
                "threshold": th.alpha,
            }
        statistic, p_value = ks_two_sample(ref, cur)
        return {
            **base,
            "method": "ks",
            "status": "ok",
            "statistic": round(statistic, 6),
            "p_value": round(p_value, 6),
            "threshold": th.alpha,
            "drifted": p_value < th.alpha,
        }
    ref_text = [v for v in map(_text, reference) if v is not None]
    cur_text = [v for v in map(_text, current) if v is not None]
    if not ref_text or not cur_text:
        return {
            **base,
            "method": "js",
            "status": "no_data",
            "drifted": False,
            "statistic": None,
            "p_value": None,
            "threshold": th.js_distance,
        }
    distance = js_distance(ref_text, cur_text)
    return {
        **base,
        "method": "js",
        "status": "ok",
        "statistic": round(distance, 6),
        "p_value": None,
        "threshold": th.js_distance,
        "drifted": distance >= th.js_distance,
    }


def _prediction(
    task: str, reference: list[Any], current: list[Any], th: Thresholds
) -> dict[str, Any]:
    if task == "classification":
        ref, cur = [str(v) for v in reference], [str(v) for v in current]
        distance = js_distance(ref, cur)

        def share(values: list[str]) -> dict[str, float]:
            return {k: round(c / len(values), 6) for k, c in sorted(Counter(values).items())}

        return {
            "method": "js",
            "status": "ok",
            "statistic": round(distance, 6),
            "p_value": None,
            "threshold": th.js_distance,
            "drifted": distance >= th.js_distance,
            "reference": share(ref),
            "current": share(cur),
        }
    ref_n = np.array([float(v) for v in reference])
    cur_n = np.array([float(v) for v in current])
    statistic, p_value = ks_two_sample(ref_n, cur_n)

    def describe(a: Array) -> dict[str, float]:
        return {
            "mean": round(float(a.mean()), 6),
            "std": round(float(a.std()), 6),
            "min": round(float(a.min()), 6),
            "max": round(float(a.max()), 6),
        }

    return {
        "method": "ks",
        "status": "ok",
        "statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "threshold": th.alpha,
        "drifted": p_value < th.alpha,
        "reference": describe(ref_n),
        "current": describe(cur_n),
    }


def build_summary(
    columns: list[dict[str, str]],
    reference_rows: list[list[Any]],
    reference_predictions: list[Any],
    current_inputs: list[dict[str, Any]],
    current_predictions: list[Any],
    task: str,
    th: Thresholds,
) -> dict[str, Any]:
    """Feature drift, the dataset-drift share and prediction drift for a current window against the
    reference. Raises `MonitorError` if either side is below `min_rows`."""
    if len(reference_rows) < th.min_rows or len(current_inputs) < th.min_rows:
        raise MonitorError(
            f"need at least {th.min_rows} rows on each side (reference {len(reference_rows)}, "
            f"current {len(current_inputs)})"
        )
    features = [
        _feature(
            c["name"],
            c["kind"],
            [row[i] for row in reference_rows],
            [row.get(c["name"]) for row in current_inputs],
            th,
        )
        for i, c in enumerate(columns)
    ]
    evaluated = [f for f in features if f["status"] == "ok"]
    drifted = [f for f in evaluated if f["drifted"]]
    share = round(len(drifted) / len(evaluated), 6) if evaluated else None
    return {
        "features": features,
        "dataset_drift": {
            "evaluated_features": len(evaluated),
            "drifted_features": len(drifted),
            "share": share,
            "threshold": th.feature_share,
            "drifted": share is not None and share >= th.feature_share,
        },
        "prediction": _prediction(task, reference_predictions, current_predictions, th),
    }


# --------------------------------------------------------------------------------------------
# Alert rules
# --------------------------------------------------------------------------------------------


def performance_comparison(
    performance: dict[str, Any],
    offline_metrics: dict[str, Any] | None,
    metric: str | None,
    th: Thresholds,
) -> dict[str, Any]:
    """The live metric against the stored offline value, or why it cannot be compared."""
    out: dict[str, Any] = {
        "metric": metric,
        "live_status": performance.get("status"),
        "threshold": th.performance_drop,
        "evaluated": False,
        "triggered": False,
    }
    live = performance.get("live")
    if metric is None or not offline_metrics or live is None:
        out["reason"] = "live metrics are not available (too few labels, or no stored metric)"
        return out
    offline_value, live_value = offline_metrics.get(metric), live["metrics"].get(metric)
    if offline_value is None or live_value is None or offline_value == 0:
        out["reason"] = f"{metric} is undefined or zero on one side"
        return out
    worse = (
        (live_value - offline_value) if metric in LOWER_IS_BETTER else (offline_value - live_value)
    )
    relative = worse / abs(offline_value)
    out.update(
        evaluated=True,
        offline=offline_value,
        live=live_value,
        labelled=live.get("rows"),
        relative_worsening=round(relative, 6),
        triggered=relative > th.performance_drop,
    )
    return out


def bootstrap_seed(prediction_ids: list[int]) -> int:
    """A seed that depends only on which predictions are scored, so a rerun over the same window
    gives the same interval."""
    digest = hashlib.sha256(",".join(str(i) for i in sorted(prediction_ids)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def performance_verdict(
    comparison: dict[str, Any],
    sampler: Callable[[Array], float | None] | None,
    rows: int,
    seed: int,
) -> dict[str, Any]:
    """Add a verdict to an evaluated comparison: `triggered`, `clear` or `inconclusive`.

    The point estimate decides `clear` (not worse than the threshold). When it is worse than the
    threshold, a seeded bootstrap of the labelled rows (`BOOTSTRAP_RESAMPLES` resamples) gives a
    one-sided `BOOTSTRAP_CONFIDENCE` lower bound on the relative worsening: `triggered` only if that
    bound is above zero (the worsening is not explained by sampling noise), else `inconclusive`,
    which is no verdict. Direction follows `LOWER_IS_BETTER`, so F1 and RMSE are handled alike.
    """
    if not comparison.get("evaluated") or sampler is None:
        return comparison
    if not comparison["triggered"]:
        comparison["verdict"] = "clear"
        return comparison
    offline = comparison["offline"]
    metric = comparison["metric"]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        live = sampler(rng.integers(0, rows, size=rows))
        if live is None:
            continue
        worse = (live - offline) if metric in LOWER_IS_BETTER else (offline - live)
        values.append(worse / abs(offline))
    interval: dict[str, Any] = {
        "method": "bootstrap",
        "resamples": BOOTSTRAP_RESAMPLES,
        "valid_resamples": len(values),
        "confidence": BOOTSTRAP_CONFIDENCE,
        "one_sided": "lower bound of the relative worsening",
        "seed": seed,
        "lower": None,
    }
    comparison["interval"] = interval
    lower = (
        float(np.quantile(values, 1.0 - BOOTSTRAP_CONFIDENCE))
        if len(values) >= MIN_VALID_RESAMPLES
        else None
    )
    if lower is None:
        interval["reason"] = "too few resamples had a defined metric"
    else:
        interval["lower"] = round(lower, 6)
    if lower is not None and lower > 0:
        comparison["verdict"] = "triggered"
    else:
        comparison["verdict"], comparison["triggered"], comparison["evaluated"] = (
            "inconclusive",
            False,
            False,
        )
        comparison["reason"] = (
            "worse than the threshold, but within sampling noise at this number of labels"
        )
    return comparison


def evaluate_rules(summary: dict[str, Any], comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """One result per rule: `evaluated` (False = no verdict, leave any alert as it is) and
    `triggered`, with the numbers that decided it."""
    drift = summary["dataset_drift"]
    prediction = summary["prediction"]
    return [
        {
            "rule": "drifted_feature_share",
            "severity": SEVERITY["drifted_feature_share"],
            "evaluated": drift["share"] is not None,
            "triggered": bool(drift["drifted"]),
            "details": {
                "share": drift["share"],
                "threshold": drift["threshold"],
                "drifted_features": [f["name"] for f in summary["features"] if f["drifted"]],
            },
        },
        {
            "rule": "prediction_shift",
            "severity": SEVERITY["prediction_shift"],
            "evaluated": prediction["status"] == "ok",
            "triggered": bool(prediction["drifted"]),
            "details": {k: prediction[k] for k in ("method", "statistic", "p_value", "threshold")},
        },
        {
            "rule": "performance_drop",
            "severity": SEVERITY["performance_drop"],
            "evaluated": bool(comparison["evaluated"]),
            "triggered": bool(comparison["triggered"]),
            "details": comparison,
        },
    ]
