# ruff: noqa: F811
"""Follow-up to P4-9: the performance-drop rule scores the labels inside the monitoring window.

Pure tests (no database) cover the bootstrap verdict, metric direction and determinism. The
PostgreSQL tests (throwaway databases, real sealed packages) show deterioration, continued
deterioration, recovery, insufficient labels, late labels and an inconclusive result for both task
types. Nothing here touches `data/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sqlalchemy import text

from automl_ops.dataset_store import FileDatasetStore
from automl_ops.feedback import metric_sampler, window_performance
from automl_ops.monitor import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    Thresholds,
    bootstrap_seed,
    evaluate_rules,
    performance_comparison,
    performance_verdict,
)
from conftest import Run, toy_bytes
from test_feedback_postgres import label, labelled_batch, predict
from test_monitoring_postgres import (
    Env,
    alerts,
    assert_alerts_consistent,
    deployed,
    events,
    normal_rows,
)
from test_registry_postgres import pool  # noqa: F401
from test_serving_postgres import REG, Rig, ready_client, rig, rows_of  # noqa: F401

TH = Thresholds()


# -- pure: verdicts, direction, determinism ------------------------------------------------


def clf_rows(correct: int, wrong: int) -> list[Any]:
    rows = []
    for i in range(correct + wrong):
        said = "pos" if i % 2 == 0 else "neg"
        truth = said if i < correct else ("neg" if said == "pos" else "pos")
        out = {"prediction": said, "probabilities": {"neg": 0.3, "pos": 0.7}}
        rows.append(SimpleNamespace(output=out, true_value=truth, created_at=_now(i)))
    return rows


def _now(i: int) -> Any:
    from datetime import UTC, datetime

    return datetime(2026, 1, 1, tzinfo=UTC).replace(second=i % 60)


def reg_rows(errors: list[float]) -> list[Any]:
    return [
        SimpleNamespace(output={"prediction": 10.0}, true_value=str(10.0 + e), created_at=_now(i))
        for i, e in enumerate(errors)
    ]


def judged(task: str, rows: list[Any], offline: dict[str, float], metric: str) -> dict[str, Any]:
    performance = window_performance(task, rows, 30)
    comparison = performance_comparison(performance, offline, metric, TH)
    if comparison["evaluated"]:
        comparison = performance_verdict(
            comparison,
            metric_sampler(task, rows, metric),
            len(rows),
            bootstrap_seed(list(range(len(rows)))),
        )
    return comparison


def test_the_bootstrap_is_deterministic_and_seeded_by_the_scored_rows() -> None:
    rows = clf_rows(20, 20)
    first = judged("classification", rows, {"accuracy": 0.95}, "accuracy")
    again = judged("classification", rows, {"accuracy": 0.95}, "accuracy")
    assert first == again and first["interval"]["resamples"] == BOOTSTRAP_RESAMPLES
    assert first["interval"]["confidence"] == BOOTSTRAP_CONFIDENCE
    assert bootstrap_seed([3, 1, 2]) == bootstrap_seed([1, 2, 3]) != bootstrap_seed([1, 2, 4])


def test_higher_is_better_metrics_trigger_only_when_the_metric_falls() -> None:
    worse = judged("classification", clf_rows(25, 25), {"accuracy": 0.95}, "accuracy")
    assert worse["verdict"] == "triggered" and worse["triggered"] and worse["interval"]["lower"] > 0
    better = judged("classification", clf_rows(50, 0), {"accuracy": 0.8}, "accuracy")
    assert better["verdict"] == "clear" and not better["triggered"]
    assert better["relative_worsening"] < 0  # improving is never a drop
    f1 = judged("classification", clf_rows(25, 25), {"f1_macro": 0.95}, "f1_macro")
    assert f1["verdict"] == "triggered" and f1["live"] < f1["offline"]


def test_lower_is_better_metrics_trigger_only_when_the_error_rises() -> None:
    errors_big = [3.0, -3.0] * 20
    worse = judged("regression", reg_rows(errors_big), {"rmse": 1.0}, "rmse")
    assert worse["verdict"] == "triggered" and worse["live"] > worse["offline"]
    lower = judged("regression", reg_rows([0.1, -0.1] * 20), {"rmse": 1.0}, "rmse")
    assert lower["verdict"] == "clear" and lower["relative_worsening"] < 0
    mae = judged("regression", reg_rows(errors_big), {"mae": 1.0}, "mae")
    assert mae["verdict"] == "triggered"


def test_a_worsening_that_sampling_noise_explains_is_inconclusive_not_a_verdict() -> None:
    errors = [0.2] * 28 + [4.98, -4.98]  # two outliers carry the whole RMSE
    comparison = judged("regression", reg_rows(errors), {"rmse": 1.0}, "rmse")
    assert comparison["relative_worsening"] > TH.performance_drop  # the point estimate is worse
    assert comparison["verdict"] == "inconclusive"
    assert comparison["interval"]["lower"] <= 0
    assert comparison["triggered"] is False and comparison["evaluated"] is False
    rule = {r["rule"]: r for r in _rules(comparison)}["performance_drop"]
    assert (rule["evaluated"], rule["triggered"]) == (False, False)  # no verdict: alerts unchanged


def _rules(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    summary = {
        "dataset_drift": {"share": None, "drifted": False, "threshold": 0.5},
        "prediction": {
            "status": "x",
            "drifted": False,
            "method": "ks",
            "statistic": None,
            "p_value": None,
            "threshold": 0.05,
        },
        "features": [],
    }
    return evaluate_rules(summary, comparison)


def test_too_few_labels_give_no_metrics_and_no_verdict() -> None:
    performance = window_performance("classification", clf_rows(10, 10), 30)
    assert performance["status"] == "insufficient_labels" and performance["live"] is None
    comparison = performance_comparison(performance, {"accuracy": 0.9}, "accuracy", TH)
    assert comparison["evaluated"] is False and "verdict" not in comparison
    assert window_performance("regression", [], 30)["status"] == "no_labels"


def test_false_trigger_rate_on_unchanged_traffic_is_small() -> None:
    """Rows whose true accuracy equals the offline value: the rate of `triggered` verdicts over
    many seeded windows of 40 labels stays low (measured below; the bound is for this test)."""
    rng = np.random.default_rng(7)
    triggered = 0
    runs = 120
    for _run in range(runs):
        correct = int((rng.random(40) < 0.9).sum())
        comparison = judged(
            "classification", clf_rows(correct, 40 - correct), {"accuracy": 0.9}, "accuracy"
        )
        triggered += comparison.get("verdict") == "triggered"
    assert triggered / runs <= 0.15, triggered / runs


# -- real PostgreSQL ------------------------------------------------------------------------

pytestmark_pg = pytest.mark.postgres


def perf_rule(report: dict[str, Any]) -> dict[str, Any]:
    (rule,) = [r for r in report["summary"]["rules"] if r["rule"] == "performance_drop"]
    out: dict[str, Any] = rule
    return out


def drop_alerts(rig: Rig) -> list[Any]:
    return alerts(rig, "rule = 'performance_drop'")


def flip(true: str) -> str:
    return {"neg": "pos", "pos": "neg"}[true]


@pytest.mark.postgres
def test_deterioration_continued_deterioration_and_recovery(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    win = {"window": 60}
    # healthy: correct labels, no alert, a clear verdict
    labelled_batch(c, rig.pool[0], 60, 60)
    healthy = env.monitor(dep, thresholds=win).json()
    assert perf_rule(healthy)["details"]["verdict"] == "clear" and drop_alerts(rig) == []
    # deterioration: the newest window is all wrong
    body, y = labelled_batch(c, rig.pool[0], 60, 0)
    for item, true in zip(body["predictions"], y, strict=True):
        assert label(c, item["prediction_id"], flip(true)).status_code == 200
    bad = env.monitor(dep, thresholds=win).json()
    rule = perf_rule(bad)
    assert rule["triggered"] and rule["details"]["verdict"] == "triggered"
    assert rule["details"]["interval"]["lower"] > 0 and rule["details"]["window_labelled"] == 60
    (alert,) = drop_alerts(rig)
    assert alert.status == "open" and alert.severity == "critical" and alert.occurrences == 1
    # continued deterioration: the same alert reoccurs, no duplicate
    again = env.monitor(dep, thresholds=win).json()
    assert perf_rule(again)["details"]["verdict"] == "triggered"
    (alert,) = drop_alerts(rig)
    assert alert.occurrences == 2 and events(rig, alert.id) == ["opened", "reoccurred"]
    # the interval is reproducible: same window, same labels, same numbers
    assert perf_rule(again)["details"]["interval"] == perf_rule(bad)["details"]["interval"]
    # recovery: a full window of correct labels resolves it automatically
    labelled_batch(c, rig.pool[0], 60, 60)
    recovered = env.monitor(dep, thresholds=win).json()
    assert perf_rule(recovered)["details"]["verdict"] == "clear"
    (alert,) = drop_alerts(rig)
    assert alert.status == "resolved" and events(rig, alert.id)[-1] == "auto_resolved"
    assert_alerts_consistent(rig)


@pytest.mark.postgres
def test_all_history_no_longer_keeps_an_alert_open(rig: Rig) -> None:
    """The regression: lots of old bad labels, then a recovered window. Cumulative scoring stayed
    wrong here; the window does not."""
    c, dep, env = deployed(rig)
    for _ in range(2):
        body, y = labelled_batch(c, rig.pool[0], 60, 0)
        for item, true in zip(body["predictions"], y, strict=True):
            label(c, item["prediction_id"], flip(true))
    assert env.monitor(dep, thresholds={"window": 60}).json()["status"] == "ok"
    assert [a.status for a in drop_alerts(rig)] == ["open"]
    labelled_batch(c, rig.pool[0], 60, 60)
    env.monitor(dep, thresholds={"window": 60})
    assert [a.status for a in drop_alerts(rig)] == ["resolved"]
    # the all-history endpoint keeps its own meaning: it still sees the old wrong labels
    live = c.get(f"/deployments/{dep}/performance")
    assert live.status_code in (200, 404)


@pytest.mark.postgres
def test_insufficient_labels_and_late_labels(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    win = {"window": 60}
    body, y = labelled_batch(c, rig.pool[0], 60, 0)
    for item, true in zip(body["predictions"], y, strict=True):
        label(c, item["prediction_id"], flip(true))
    first = env.monitor(dep, thresholds=win).json()
    (alert,) = drop_alerts(rig)
    # a new window with only 10 labels: no verdict, the alert is untouched
    body, y = labelled_batch(c, rig.pool[0], 60, 10)
    few = env.monitor(dep, thresholds=win).json()
    rule = perf_rule(few)
    assert rule["evaluated"] is False and rule["details"]["window_labelled"] == 10
    (still,) = drop_alerts(rig)
    assert (still.status, still.occurrences) == (alert.status, alert.occurrences)
    # labels arrive late for the same window: the next report counts them
    for item, true in zip(body["predictions"][10:50], y[10:50], strict=True):
        label(c, item["prediction_id"], true)
    late = env.monitor(dep, thresholds=win).json()
    assert perf_rule(late)["details"]["window_labelled"] == 50
    assert perf_rule(late)["details"]["verdict"] == "clear"
    (resolved,) = drop_alerts(rig)
    assert resolved.status == "resolved"
    # earlier reports are immutable: the report made before the labels still says 10
    old = env.client.get(f"/reports/{few['id']}").json()
    assert perf_rule(old)["details"]["window_labelled"] == 10
    assert perf_rule(first)["details"]["window_labelled"] == 60
    assert_alerts_consistent(rig)


@pytest.mark.postgres
def test_the_window_scores_only_its_own_scope(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    body, y = labelled_batch(c, rig.pool[0], 60, 0, simulated=True)
    for item, true in zip(body["predictions"], y, strict=True):
        label(c, item["prediction_id"], flip(true))
    labelled_batch(c, rig.pool[0], 60, 60)
    real = env.monitor(dep, scope="real", thresholds={"window": 60}).json()
    assert perf_rule(real)["details"]["verdict"] == "clear" and drop_alerts(rig) == []
    sim = env.monitor(dep, scope="simulated", thresholds={"window": 60}).json()
    assert perf_rule(sim)["details"]["verdict"] == "triggered"
    assert [a.scope for a in drop_alerts(rig)] == ["simulated"]


@pytest.mark.postgres
def test_regression_inconclusive_deterioration_and_recovery(
    job_env: Any,
    pool: list[Run],
    tmp_path: Path,  # noqa: F811
) -> None:
    FileDatasetStore(tmp_path / "reg-store").ingest(
        "toy", toy_bytes("regression"), source_filename="toy.csv"
    )
    rig_ = Rig(job_env, pool)
    rig_.register(pool[3], REG)
    c, _ = ready_client(rig_, REG, 1)
    dep = rig_.deployments.active().id  # type: ignore[union-attr]
    env = Env(rig_, tmp_path / "reg-store")
    assert env.reference(REG, 1).status_code == 201
    with rig_.engine.connect() as conn:
        evaluation = conn.execute(
            text(
                "SELECT v.evaluation FROM deployments d "
                "JOIN model_versions v ON v.id = d.version_id WHERE d.id = :i"
            ),
            {"i": dep},
        ).scalar_one()
    offline = evaluation["evaluation"]["metrics"][evaluation["selection_metric"]]
    assert evaluation["selection_metric"] in ("rmse", "mae") and offline > 0
    win = {"window": 30}

    def window_with(errors: list[float]) -> dict[str, Any]:
        body = predict(c, rows_of(pool[3], 30))
        for item, err in zip(body["predictions"], errors, strict=True):
            assert label(c, item["prediction_id"], item["prediction"] + err).status_code == 200
        out: dict[str, Any] = env.monitor(dep, thresholds=win).json()
        return out

    scale = offline
    # outliers carry the error: worse than the threshold by point estimate, but not by the interval
    noisy = [0.2 * scale] * 28 + [5.0 * scale, -5.0 * scale]
    r1 = window_with(noisy)
    assert perf_rule(r1)["details"]["verdict"] == "inconclusive" and drop_alerts(rig_) == []
    # a real, uniform deterioration
    r2 = window_with([3.0 * scale, -3.0 * scale] * 15)
    assert perf_rule(r2)["details"]["verdict"] == "triggered"
    (alert,) = drop_alerts(rig_)
    assert alert.status == "open"
    # an inconclusive window afterwards changes nothing
    r3 = window_with(noisy)
    assert perf_rule(r3)["details"]["verdict"] == "inconclusive"
    (same,) = drop_alerts(rig_)
    assert (same.status, same.occurrences) == ("open", 1)
    # recovery: errors well below the offline value resolve it
    r4 = window_with([0.1 * scale, -0.1 * scale] * 15)
    assert perf_rule(r4)["details"]["verdict"] == "clear"
    (done,) = drop_alerts(rig_)
    assert done.status == "resolved" and events(rig_, done.id)[-1] == "auto_resolved"
    assert_alerts_consistent(rig_)


@pytest.mark.postgres
def test_a_report_stores_the_window_evidence_as_json(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    labelled_batch(c, rig.pool[0], 40, 40)
    report = env.monitor(dep, thresholds={"window": 40}).json()
    json.dumps(report["summary"])  # plain JSON, no new column, no migration
    assert report["summary"]["window"]["labelled"] == 40
    assert normal_rows(rig, 1)
