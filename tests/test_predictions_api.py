# ruff: noqa: F811
"""Phase 5: the two read-only endpoints the UI needs, `GET /deployments/{id}/predictions` (history)
and `GET /deployments/{id}/activity` (counts over time), on real PostgreSQL with a real model.
Throwaway databases only; nothing here touches `data/`."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from api_support import assert_envelope
from automl_ops.dataset_store import FileDatasetStore
from conftest import Run, toy_bytes
from test_feedback_postgres import label, predict
from test_monitoring_postgres import Env, deployed, normal_rows
from test_registry_postgres import pool  # noqa: F401
from test_serving_postgres import REG, Rig, ready_client, rig, rows_of  # noqa: F401

pytestmark = pytest.mark.postgres


def history(env: Any, dep: int, **params: Any) -> Any:
    return env.client.get(f"/deployments/{dep}/predictions", params=params)


def activity(env: Any, dep: int, **params: Any) -> Any:
    return env.client.get(f"/deployments/{dep}/activity", params=params)


def counts(rig: Rig) -> dict[str, int]:
    with rig.engine.connect() as c:
        return {
            t: c.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()
            for t in ("predictions", "feedback", "monitoring_reports", "alerts")
        }


def test_an_empty_deployment_has_no_history_and_a_zero_activity_chart(rig: Rig) -> None:
    _, dep, env = deployed(rig)
    body = history(env, dep).json()
    assert body["predictions"] == [] and body["next_before"] is None
    report = activity(env, dep).json()
    assert report["totals"]["real"] == {"predictions": 0, "labelled": 0}
    assert report["latency_ms"]["predictions"] == 0 and report["latency_ms"]["mean"] is None
    assert report["buckets"] and all(b["real"]["predictions"] == 0 for b in report["buckets"])


def test_history_is_newest_first_separates_traffic_and_hides_inputs(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    real = predict(c, normal_rows(rig, 5))
    simulated = predict(c, normal_rows(rig, 3), simulated=True)
    label(c, real["predictions"][0]["prediction_id"], "pos")
    before = counts(rig)
    only_real = history(env, dep).json()
    assert [p["is_simulated"] for p in only_real["predictions"]] == [False] * 5
    assert {p["id"] for p in only_real["predictions"]} == {
        p["prediction_id"] for p in real["predictions"]
    }
    sim = history(env, dep, scope="simulated").json()
    assert {p["id"] for p in sim["predictions"]} == {
        p["prediction_id"] for p in simulated["predictions"]
    }
    both = history(env, dep, scope="all").json()
    assert len(both["predictions"]) == 8 and both["scope"] == "all"
    stamps = [p["created_at"] for p in both["predictions"]]
    assert stamps == sorted(stamps, reverse=True)
    labelled = [p for p in both["predictions"] if p["true_value"] is not None]
    assert [p["true_value"] for p in labelled] == ["pos"]
    for p in both["predictions"]:
        assert p["latency_ms"] >= 0 and 0 <= p["confidence"] <= 1
        assert set(p) == {
            "id", "created_at", "model_version", "is_simulated", "prediction", "confidence",
            "latency_ms", "true_value",
        }  # fmt: skip
    assert "inputs" not in json.dumps(both) and '"color"' not in json.dumps(both)
    assert counts(rig) == before  # reading changes nothing


def test_paging_walks_every_row_once_even_when_timestamps_tie(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, normal_rows(rig, 8))  # one request: rows share a timestamp, so the id breaks ties
    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        params: dict[str, Any] = {"limit": 3, "scope": "all"}
        if cursor:
            params["before"] = cursor
        body = history(env, dep, **params).json()
        seen += [p["id"] for p in body["predictions"]]
        pages += 1
        cursor = body["next_before"]
        if cursor is None:
            break
    assert len(seen) == 8 and len(set(seen)) == 8 and pages == 3


def test_the_time_range_filters_history(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, normal_rows(rig, 4))
    first = history(env, dep).json()["predictions"][-1]["created_at"]
    now = datetime.now(UTC)
    assert len(history(env, dep, since=first).json()["predictions"]) == 4  # inclusive
    assert (
        history(env, dep, since=(now + timedelta(hours=1)).isoformat()).json()["predictions"] == []
    )
    assert history(env, dep, until=first).json()["predictions"] == []  # exclusive


def test_bad_input_and_unknown_deployments_use_the_error_envelope(rig: Rig) -> None:
    _, dep, env = deployed(rig)
    for params in ({"before": "nonsense"}, {"limit": 0}, {"limit": 101}, {"scope": "everything"}):
        response = history(env, dep, **params)
        assert response.status_code == 422, params
        assert response.json()["error"]["code"] in ("invalid_input", "validation_error")
    assert_envelope(history(env, 99999), 404, "deployment_not_found")
    assert_envelope(activity(env, 99999), 404, "deployment_not_found")
    assert activity(env, dep, bucket="week").status_code == 422
    assert activity(env, dep, scope="x").status_code == 422


def test_activity_counts_match_and_empty_buckets_are_filled(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    real = predict(c, normal_rows(rig, 6))
    predict(c, normal_rows(rig, 4), simulated=True)
    for item in real["predictions"][:2]:
        label(c, item["prediction_id"], "neg")
    since = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    body = activity(env, dep, since=since, bucket="hour").json()
    assert body["bucket"] == "hour" and 3 <= len(body["buckets"]) <= 5  # zero hours are included
    assert body["totals"]["real"] == {"predictions": 6, "labelled": 2}
    assert body["totals"]["simulated"] == {"predictions": 4, "labelled": 0}
    assert sum(b["real"]["predictions"] for b in body["buckets"]) == 6
    assert sum(1 for b in body["buckets"] if b["real"]["predictions"] == 0) >= 2
    assert body["latency_ms"]["predictions"] == 10 and body["latency_ms"]["p95"] >= 0
    only = activity(env, dep, since=since, bucket="hour", scope="real").json()
    assert only["totals"]["simulated"] == {"predictions": 0, "labelled": 0}
    auto = activity(env, dep).json()  # default range: first prediction to now, chosen bucket
    assert auto["bucket"] == "minute" and len(auto["buckets"]) <= 200


def test_activity_is_bounded(rig: Rig) -> None:
    _, dep, env = deployed(rig)
    far = (datetime.now(UTC) - timedelta(days=700)).isoformat()
    too_big = activity(env, dep, since=far, bucket="minute")
    assert too_big.status_code == 422 and "more than" in too_big.json()["error"]["message"]
    ok = activity(env, dep, since=far)  # automatic bucket: days, 700 of them is over 500
    assert ok.status_code == 422
    near = (datetime.now(UTC) - timedelta(days=300)).isoformat()
    assert len(activity(env, dep, since=near).json()["buckets"]) <= 500
    now = datetime.now(UTC)
    backwards = activity(
        env, dep, since=now.isoformat(), until=(now - timedelta(hours=1)).isoformat()
    )
    assert backwards.status_code == 422


def test_regression_history_has_a_number_and_no_confidence(
    job_env: Any, pool: list[Run], tmp_path: Path
) -> None:
    FileDatasetStore(tmp_path / "reg-store").ingest(
        "toy", toy_bytes("regression"), source_filename="toy.csv"
    )
    rig_ = Rig(job_env, pool)
    rig_.register(pool[3], REG)
    c, _ = ready_client(rig_, REG, 1)
    dep = rig_.deployments.active().id  # type: ignore[union-attr]
    env = Env(rig_, tmp_path / "reg-store")
    predict(c, rows_of(pool[3], 5))
    rows = history(env, dep).json()["predictions"]
    assert len(rows) == 5 and all(isinstance(r["prediction"], float) for r in rows)
    assert all(r["confidence"] is None for r in rows)
