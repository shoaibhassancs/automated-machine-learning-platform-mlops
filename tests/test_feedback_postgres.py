# ruff: noqa: F811
"""P4-8: feedback labels and live performance on real PostgreSQL with real sealed packages.

Throwaway databases only; NOT RUN when the server is unreachable. Nothing here touches `data/`.
"""

from __future__ import annotations

import builtins
import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from api_support import BASE, assert_envelope
from automl_ops import feedback as feedback_module
from automl_ops.api_control.app import create_app as create_control
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.feedback import FeedbackRepository
from conftest import Run
from test_registry_postgres import pid, pool, reg, sealed_copy, unlock  # noqa: F401
from test_serving_postgres import CLF, REG, Rig, ready_client, rig, rows_logged, rows_of  # noqa: F401

pytestmark = pytest.mark.postgres


def control(rig: Rig, *, wired: bool = True) -> TestClient:
    settings = ApiSettings(store_root=rig.env.store_root, models_root=rig.root)
    app = create_control(
        settings,
        CatalogedDatasetStore(FileDatasetStore(rig.env.store_root, settings.limits), rig.engine),
        lambda: migrate.status(rig.engine),
        deployments=rig.deployments,
        feedback=FeedbackRepository(rig.engine) if wired else None,
    )
    return TestClient(app, base_url=BASE, raise_server_exceptions=False)


def predict(c: TestClient, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    response = c.post("/predict", json={"rows": rows, **extra})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def label(c: TestClient, prediction_id: str, value: Any, **extra: Any) -> Any:
    return c.post("/feedback", json={"prediction_id": prediction_id, "true_value": value, **extra})


def truth(run: Run, count: int) -> list[str]:
    return [str(v) for v in run.frame.head(count)["y"]]


def labelled_batch(
    c: TestClient, run: Run, count: int, labelled: int, **extra: Any
) -> tuple[dict[str, Any], list[str]]:
    """Predict `count` rows and label the first `labelled` of them with the real outcome."""
    body = predict(c, rows_of(run, count), **extra)
    y = truth(run, count)
    for item, true in zip(body["predictions"][:labelled], y[:labelled], strict=True):
        assert label(c, item["prediction_id"], true).status_code == 200
    return body, y


def label_rows(rig: Rig) -> list[Any]:
    with rig.engine.connect() as c:
        return list(c.execute(text("SELECT * FROM feedback ORDER BY id")).all())


# -- valid feedback -----------------------------------------------------------------------


def test_a_classification_label_is_recorded_against_its_prediction_deployment_and_version(
    rig: Rig,
) -> None:
    c, runtime = ready_client(rig)
    body = predict(c, rows_of(rig.pool[0], 2))
    first = body["predictions"][0]
    response = label(c, first["prediction_id"], "pos")
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["created"] is True
    got = out["label"]
    assert got["prediction_id"] == first["prediction_id"] and got["true_value"] == "pos"
    assert got["deployment_id"] == body["deployment"]["deployment_id"] and got["model_version"] == 1
    assert got["prediction"] == first["prediction"] and got["is_simulated"] is False
    assert len(label_rows(rig)) == 1 and runtime.feedback is not None


def test_a_regression_label_is_stored_as_a_canonical_number(rig: Rig) -> None:
    c, _ = ready_client(rig, REG, 1)
    ids = [p["prediction_id"] for p in predict(c, rows_of(rig.pool[3], 3))["predictions"]]
    assert label(c, ids[0], 1.5).json()["label"]["true_value"] == "1.5"
    assert label(c, ids[1], "2").json()["label"]["true_value"] == "2.0"
    assert label(c, ids[2], 3).json()["label"]["true_value"] == "3.0"
    assert label(c, ids[0], "1.50").json()["created"] is False  # the same number


def test_labelling_needs_only_the_database_not_a_loaded_model(rig: Rig) -> None:
    c, runtime = ready_client(rig)
    pid_ = predict(c, rows_of(rig.pool[0], 1))["predictions"][0]["prediction_id"]
    active = rig.deployments.active()
    assert active is not None
    rig.deployments.undeploy(active.id, "maintenance")
    runtime.refresh()
    assert c.get("/ready").status_code == 503
    assert label(c, pid_, "neg").status_code == 200  # late feedback still lands


# -- invalid, duplicate, conflicting ------------------------------------------------------


def test_invalid_references_and_values_are_refused_and_store_nothing(rig: Rig) -> None:
    c, _ = ready_client(rig)
    pid_ = predict(c, rows_of(rig.pool[0], 1))["predictions"][0]["prediction_id"]
    ghost = "00000000-0000-4000-8000-000000000000"
    assert_envelope(label(c, ghost, "pos"), 404, "prediction_not_found")
    for bad_id in ("nope", "", pid_.upper(), pid_ + "0", "../../etc/passwd"):
        assert_envelope(label(c, bad_id, "pos"), 422, "validation_error")
    wrong = assert_envelope(label(c, pid_, "maybe"), 422, "invalid_true_value")
    assert wrong["details"]["classes"] == ["neg", "pos"]
    for value in (1, 1.5, True, None, ["pos"], {"a": 1}):
        assert label(c, pid_, value).status_code == 422
    assert_envelope(c.post("/feedback", json={"prediction_id": pid_}), 422, "validation_error")
    assert_envelope(
        c.post("/feedback", json={"prediction_id": pid_, "true_value": "pos", "extra": 1}),
        422,
        "validation_error",
    )
    assert label_rows(rig) == []


def test_regression_values_must_be_finite_numbers(rig: Rig) -> None:
    c, _ = ready_client(rig, REG, 1)
    pid_ = predict(c, rows_of(rig.pool[3], 1))["predictions"][0]["prediction_id"]
    for value in ("abc", "", "1e999", "inf", True, None, [1]):
        assert label(c, pid_, value).status_code == 422, value
    for raw in (
        b'{"prediction_id": "%s", "true_value": NaN}',
        b'{"prediction_id": "%s", "true_value": Infinity}',
    ):
        r = c.post(
            "/feedback",
            content=raw % pid_.encode(),
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 422
    assert label_rows(rig) == []


def test_the_same_label_again_is_a_no_op_and_a_different_one_never_overwrites(rig: Rig) -> None:
    c, _ = ready_client(rig)
    pid_ = predict(c, rows_of(rig.pool[0], 1))["predictions"][0]["prediction_id"]
    first = label(c, pid_, "pos").json()["label"]
    again = label(c, pid_, "pos")
    assert again.status_code == 200 and again.json()["created"] is False
    assert again.json()["label"]["id"] == first["id"]
    conflict = assert_envelope(label(c, pid_, "neg"), 409, "feedback_conflict")
    assert conflict["details"]["existing_true_value"] == "pos"
    rows = label_rows(rig)
    assert len(rows) == 1 and rows[0].true_value == "pos"


def test_a_label_cannot_be_attached_to_another_deployment_or_model_version(rig: Rig) -> None:
    c, _ = ready_client(rig)
    body = predict(c, rows_of(rig.pool[0], 1))
    pid_ = body["predictions"][0]["prediction_id"]
    real = body["deployment"]["deployment_id"]
    wrong_deployment = assert_envelope(
        label(c, pid_, "pos", deployment_id=real + 1), 409, "prediction_mismatch"
    )
    assert wrong_deployment["details"]["prediction_deployment_id"] == real
    assert_envelope(label(c, pid_, "pos", model_version=2), 409, "prediction_mismatch")
    assert label_rows(rig) == []
    ok = label(c, pid_, "pos", deployment_id=real, model_version=1)
    assert ok.status_code == 200 and ok.json()["created"] is True


def test_a_late_label_stays_with_the_deployment_that_made_the_prediction(rig: Rig) -> None:
    c, runtime = ready_client(rig)
    old = predict(c, rows_of(rig.pool[0], 2))
    first = rig.deployments.active()
    assert first is not None
    rig.deployments.switch(first.id, CLF, 2, "forest")
    runtime.refresh()
    late = label(c, old["predictions"][0]["prediction_id"], "pos").json()["label"]
    assert late["deployment_id"] == first.id and late["model_version"] == 1
    cp = control(rig)
    assert cp.get(f"/deployments/{first.id}/performance", params={"min_labelled": 1}).json()[
        "counts"
    ] == {"predictions": 2, "labelled": 1, "unlabelled": 1}
    new_id = rig.deployments.active().id  # type: ignore[union-attr]
    assert cp.get(f"/deployments/{new_id}/performance").json()["status"] == "no_predictions"


# -- the database's own rules -------------------------------------------------------------


def refused(rig: Rig, sql: str, **params: Any) -> str:
    with pytest.raises((IntegrityError, DBAPIError)) as raised, rig.engine.begin() as c:
        c.execute(text(sql), params)
    return str(raised.value)


def test_the_database_enforces_matching_references_and_keeps_labels_immutable(rig: Rig) -> None:
    c, _ = ready_client(rig)
    body = predict(c, rows_of(rig.pool[0], 2))
    a, b = (p["prediction_id"] for p in body["predictions"])
    dep = body["deployment"]["deployment_id"]
    insert = (
        "INSERT INTO feedback (prediction_id, deployment_id, model_version, true_value) "
        "VALUES (:p, :d, :v, :t)"
    )
    assert "does not match" in refused(rig, insert, p=a, d=dep, v=2, t="pos")
    refused(rig, insert, p="no-such-prediction", d=dep, v=1, t="pos")
    refused(rig, insert, p=a, d=dep, v=1, t="")
    refused(rig, insert, p=a, d=dep, v=1, t="x" * 201)
    with rig.engine.begin() as conn:
        conn.execute(text(insert), {"p": a, "d": dep, "v": 1, "t": "pos"})
    refused(rig, insert, p=a, d=dep, v=1, t="neg")  # unique
    for sql in (
        "UPDATE feedback SET true_value = 'neg'",
        "DELETE FROM feedback",
        "TRUNCATE feedback",
        "DELETE FROM predictions WHERE id = :p",
    ):
        refused(rig, sql, p=a)
    assert [r.true_value for r in label_rows(rig)] == ["pos"] and b


def test_a_failed_write_leaves_existing_predictions_and_labels_intact(rig: Rig) -> None:
    c, _ = ready_client(rig)
    body = predict(c, rows_of(rig.pool[0], 3))
    a, b, _ = (p["prediction_id"] for p in body["predictions"])
    assert label(c, a, "pos").status_code == 200
    with rig.engine.begin() as conn:
        conn.execute(text("ALTER TABLE feedback ADD CONSTRAINT never CHECK (false) NOT VALID"))
    broken = label(c, b, "neg")
    assert_envelope(broken, 503, "feedback_unavailable")
    with rig.engine.begin() as conn:
        conn.execute(text("ALTER TABLE feedback DROP CONSTRAINT never"))
    assert [r.prediction_id for r in label_rows(rig)] == [a]
    assert rows_logged(rig) == 3
    assert label(c, b, "neg").status_code == 200  # and it works again afterwards


def test_an_unreachable_database_is_a_clear_503_not_a_silent_loss(rig: Rig) -> None:
    c, runtime = ready_client(rig)
    pid_ = predict(c, rows_of(rig.pool[0], 1))["predictions"][0]["prediction_id"]
    good = runtime.feedback.engine
    runtime.feedback.engine = create_engine(
        "postgresql+psycopg://nobody:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    assert_envelope(label(c, pid_, "pos"), 503, "feedback_unavailable")
    runtime.feedback.engine = good
    assert label(c, pid_, "pos").status_code == 200


# -- concurrency --------------------------------------------------------------------------


def race(calls: list[Callable[[], Any]]) -> list[Any]:
    barrier = threading.Barrier(len(calls))

    def go(call: Callable[[], Any]) -> Any:
        barrier.wait()
        return call()

    with ThreadPoolExecutor(len(calls)) as pool_:
        return list(pool_.map(go, calls))


def test_simultaneous_identical_labels_write_exactly_one_row(rig: Rig) -> None:
    c, _ = ready_client(rig)
    pid_ = predict(c, rows_of(rig.pool[0], 1))["predictions"][0]["prediction_id"]
    results = race([lambda: label(c, pid_, "pos") for _ in range(8)])
    assert all(r.status_code == 200 for r in results)
    assert sum(r.json()["created"] for r in results) == 1
    assert len(label_rows(rig)) == 1


def test_simultaneous_conflicting_labels_have_one_winner_and_the_rest_are_refused(
    rig: Rig,
) -> None:
    c, _ = ready_client(rig)
    pid_ = predict(c, rows_of(rig.pool[0], 1))["predictions"][0]["prediction_id"]
    values = ["pos", "neg"] * 4
    calls: list[Callable[[], Any]] = [partial(label, c, pid_, v) for v in values]
    results = race(calls)
    stored = label_rows(rig)
    assert len(stored) == 1
    for response, value in zip(results, values, strict=True):
        if value == stored[0].true_value:
            assert response.status_code == 200
        else:
            assert_envelope(response, 409, "feedback_conflict")
    assert sum(r.status_code == 200 and r.json()["created"] for r in results) == 1


def test_labels_arriving_while_performance_is_read_never_break_the_numbers(rig: Rig) -> None:
    c, _ = ready_client(rig)
    cp = control(rig)
    body = predict(c, rows_of(rig.pool[0], 40))
    y = truth(rig.pool[0], 40)
    dep = body["deployment"]["deployment_id"]
    stop = threading.Event()
    seen: list[dict[str, Any]] = []

    def reader() -> None:
        while not stop.is_set():
            r = cp.get(f"/deployments/{dep}/performance", params={"min_labelled": 5})
            assert r.status_code == 200, r.text
            seen.append(r.json())

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        labels: list[Callable[[], Any]] = [
            partial(label, c, p["prediction_id"], t)
            for p, t in zip(body["predictions"], y, strict=True)
        ]
        race(labels)
    finally:
        stop.set()
        thread.join()
    assert seen
    for snapshot in seen:
        counts = snapshot["counts"]
        assert counts["labelled"] + counts["unlabelled"] == 40
        if snapshot["status"] == "ok":
            assert snapshot["live"]["rows"] == snapshot["rows_scored"] == counts["labelled"]


# -- live performance ---------------------------------------------------------------------


def test_live_classification_metrics_equal_an_independent_calculation_on_labelled_rows_only(
    rig: Rig,
) -> None:
    c, _ = ready_client(rig)
    cp = control(rig)
    body, y = labelled_batch(c, rig.pool[0], 60, 45)
    dep = body["deployment"]["deployment_id"]
    got = cp.get(f"/deployments/{dep}/performance")
    assert got.status_code == 200, got.text
    perf = got.json()
    assert perf["status"] == "ok" and perf["scope"] == "real"
    assert perf["counts"] == {"predictions": 60, "labelled": 45, "unlabelled": 15}
    assert perf["rows_scored"] == 45 and perf["truncated"] is False
    y_true = np.array(y[:45])
    y_pred = np.array([p["prediction"] for p in body["predictions"][:45]])
    proba_pos = np.array([p["probabilities"]["pos"] for p in body["predictions"][:45]])
    live = perf["live"]
    assert live["source"] == "live_feedback" and live["rows"] == 45
    assert live["classes"] == ["neg", "pos"]
    assert live["metrics"]["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert live["metrics"]["f1_macro"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    assert live["metrics"]["roc_auc"] == pytest.approx(roc_auc_score(y_true == "pos", proba_pos))
    matrix = live["confusion_matrix"]
    assert sum(map(sum, matrix)) == 45
    assert matrix[1][1] == int(((y_true == "pos") & (y_pred == "pos")).sum())
    # the 15 unlabelled predictions are in no number: label them and the numbers change
    for item, true in zip(body["predictions"][45:], y[45:], strict=True):
        label(c, item["prediction_id"], true)
    after = cp.get(f"/deployments/{dep}/performance").json()
    assert after["counts"]["unlabelled"] == 0 and after["live"]["rows"] == 60


def test_live_regression_metrics_equal_an_independent_calculation(rig: Rig) -> None:
    c, _ = ready_client(rig, REG, 1)
    cp = control(rig)
    rows = rows_of(rig.pool[3], 50)
    body = predict(c, rows)
    y = np.array([float(v) for v in rig.pool[3].frame.head(50)["y"]])
    for item, true in zip(body["predictions"][:40], y[:40], strict=True):
        assert label(c, item["prediction_id"], str(true)).status_code == 200
    dep = body["deployment"]["deployment_id"]
    perf = cp.get(f"/deployments/{dep}/performance").json()
    pred = np.array([p["prediction"] for p in body["predictions"][:40]])
    assert perf["status"] == "ok" and perf["task"] == "regression"
    assert perf["counts"] == {"predictions": 50, "labelled": 40, "unlabelled": 10}
    metrics = perf["live"]["metrics"]
    assert metrics["rmse"] == pytest.approx(math.sqrt(float(np.mean((y[:40] - pred) ** 2))))
    assert metrics["mae"] == pytest.approx(float(np.mean(np.abs(y[:40] - pred))))
    ss_res = float(np.sum((y[:40] - pred) ** 2))
    assert metrics["r2"] == pytest.approx(1 - ss_res / float(np.sum((y[:40] - y[:40].mean()) ** 2)))
    assert "classes" not in perf["live"] or perf["live"]["classes"] is None


def test_too_few_labels_give_a_status_and_counts_never_metrics(rig: Rig) -> None:
    c, _ = ready_client(rig)
    cp = control(rig)
    dep = predict(c, rows_of(rig.pool[0], 1))["deployment"]["deployment_id"]
    assert cp.get(f"/deployments/{dep}/performance").json()["status"] == "no_labels"
    body, _ = labelled_batch(c, rig.pool[0], 10, 5)
    perf = cp.get(f"/deployments/{dep}/performance").json()
    assert perf["status"] == "insufficient_labels" and perf["live"] is None
    assert perf["counts"]["labelled"] == 5 and perf["min_labelled"] == 30
    assert any("5 labelled" in n for n in perf["notes"])
    lowered = cp.get(f"/deployments/{dep}/performance", params={"min_labelled": 5}).json()
    assert lowered["status"] == "ok" and lowered["live"]["rows"] == 5
    assert body["deployment"]["deployment_id"] == dep


def test_a_deployment_with_no_predictions_says_so(rig: Rig) -> None:
    d = rig.deployments.deploy(CLF, 1, "quiet")
    perf = control(rig).get(f"/deployments/{d.id}/performance").json()
    assert perf["status"] == "no_predictions" and perf["live"] is None
    assert perf["counts"] == {"predictions": 0, "labelled": 0, "unlabelled": 0}


def test_labels_of_one_class_make_roc_auc_undefined_with_the_reason(rig: Rig) -> None:
    c, _ = ready_client(rig)
    body = predict(c, rows_of(rig.pool[0], 40))
    for item in body["predictions"]:
        label(c, item["prediction_id"], "pos")
    dep = body["deployment"]["deployment_id"]
    live = control(rig).get(f"/deployments/{dep}/performance").json()["live"]
    assert live["metrics"]["roc_auc"] is None and "does not occur" in live["undefined"]["roc_auc"]


def test_simulated_traffic_is_kept_apart_from_real_traffic(rig: Rig) -> None:
    c, _ = ready_client(rig)
    cp = control(rig)
    real, _ = labelled_batch(c, rig.pool[0], 35, 35)
    sim, _ = labelled_batch(c, rig.pool[0], 31, 31, simulated=True)
    dep = real["deployment"]["deployment_id"]
    url = f"/deployments/{dep}/performance"
    default = cp.get(url).json()
    assert default["scope"] == "real" and default["counts"]["labelled"] == 35
    assert default["by_traffic"] == {
        "real": {"predictions": 35, "labelled": 35},
        "simulated": {"predictions": 31, "labelled": 31},
    }
    only_sim = cp.get(url, params={"scope": "simulated"}).json()
    assert only_sim["counts"]["labelled"] == 31 and only_sim["live"]["rows"] == 31
    both = cp.get(url, params={"scope": "all"}).json()
    assert both["counts"]["labelled"] == 66 and any("simulated" in n for n in both["notes"])
    assert sim["deployment"]["deployment_id"] == dep


def test_offline_numbers_are_shown_separately_and_never_mixed_into_live_ones(rig: Rig) -> None:
    c, _ = ready_client(rig)
    body, _ = labelled_batch(c, rig.pool[0], 40, 40)
    dep = body["deployment"]["deployment_id"]
    perf = control(rig).get(f"/deployments/{dep}/performance").json()
    offline = perf["offline_reference"]
    stored = rig.registry.get_version(CLF, 1).evaluation
    assert stored is not None
    assert offline["source"] == "offline_final_test" and perf["live"]["source"] == "live_feedback"
    assert offline["metrics"] == stored["evaluation"]["metrics"]
    assert offline["limitation"] == stored["limitation"] and "not comparable" in offline["note"]
    assert perf["live"]["metrics"] != offline["metrics"] or perf["live"]["rows"] != offline["rows"]
    assert set(perf) >= {"live", "offline_reference"} and "delta" not in str(perf).lower()


def test_a_very_large_labelled_set_is_scored_in_part_and_says_so(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(feedback_module, "MAX_SCORED", 12)
    c, _ = ready_client(rig)
    body, _ = labelled_batch(c, rig.pool[0], 40, 40)
    perf = (
        control(rig).get(f"/deployments/{body['deployment']['deployment_id']}/performance").json()
    )
    assert (
        perf["rows_scored"] == 12 and perf["truncated"] is True and perf["counts"]["labelled"] == 40
    )
    assert any("12 most recent" in n for n in perf["notes"])


def test_the_label_history_pages_newest_first(rig: Rig) -> None:
    c, _ = ready_client(rig)
    cp = control(rig)
    body, _ = labelled_batch(c, rig.pool[0], 5, 5)
    dep = body["deployment"]["deployment_id"]
    page = cp.get(f"/deployments/{dep}/feedback", params={"limit": 3}).json()
    assert [x["prediction_id"] for x in page["labels"]] == [
        p["prediction_id"] for p in reversed(body["predictions"][2:])
    ]
    rest = cp.get(
        f"/deployments/{dep}/feedback", params={"limit": 3, "before_id": page["next_before_id"]}
    ).json()
    assert len(rest["labels"]) == 2 and rest["next_before_id"] is None


# -- API behaviour ------------------------------------------------------------------------


def test_performance_endpoint_errors_and_local_rules(rig: Rig) -> None:
    c, _ = ready_client(rig)
    cp = control(rig)
    assert_envelope(cp.get("/deployments/99999/performance"), 404, "deployment_not_found")
    assert_envelope(cp.get("/deployments/99999/feedback"), 404, "deployment_not_found")
    dep = rig.deployments.active().id  # type: ignore[union-attr]
    for params in ({"scope": "bogus"}, {"min_labelled": 0}, {"min_labelled": "x"}):
        assert_envelope(cp.get(f"/deployments/{dep}/performance", params=params), 422)
    assert_envelope(cp.get("/deployments/abc/performance"), 422, "validation_error")
    assert_envelope(
        cp.get(f"/deployments/{dep}/performance", headers={"host": "evil.example"}), 400
    )
    assert_envelope(cp.post(f"/deployments/{dep}/performance"), 405)
    off = control(rig, wired=False)
    assert_envelope(off.get(f"/deployments/{dep}/performance"), 503, "feedback_unavailable")
    pid_ = predict(c, rows_of(rig.pool[0], 1))["predictions"][0]["prediction_id"]
    good = {"prediction_id": pid_, "true_value": "pos"}
    assert_envelope(c.post("/feedback", json=good, headers={"origin": "http://evil.example"}), 403)
    assert_envelope(
        c.post(
            "/feedback",
            json={"prediction_id": pid_, "true_value": "pos"},
            headers={"host": "evil.example"},
        ),
        400,
    )
    assert_envelope(c.post("/feedback", content=b"x", headers={"content-type": "text/plain"}), 415)
    assert label_rows(rig) == []


def test_labels_and_performance_survive_a_restart_of_both_apps(rig: Rig) -> None:
    c, _ = ready_client(rig)
    body, _ = labelled_batch(c, rig.pool[0], 40, 35)
    dep = body["deployment"]["deployment_id"]
    before = control(rig).get(f"/deployments/{dep}/performance").json()
    # brand-new serving runtime and control app on the same database
    runtime = rig.runtime()
    runtime.refresh()
    fresh = rig.client(runtime)
    again = label(fresh, body["predictions"][0]["prediction_id"], truth(rig.pool[0], 1)[0])
    assert again.status_code == 200 and again.json()["created"] is False
    conflict = "neg" if truth(rig.pool[0], 1)[0] == "pos" else "pos"
    assert_envelope(label(fresh, body["predictions"][0]["prediction_id"], conflict), 409)
    assert control(rig).get(f"/deployments/{dep}/performance").json() == before


# -- held-out data, migration, existing behaviour -----------------------------------------


def test_feedback_and_performance_never_read_any_file(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, _ = ready_client(rig)
    cp = control(rig)
    body = predict(c, rows_of(rig.pool[0], 40))
    reads: list[str] = []
    real_open = builtins.open

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        path = str(file)
        if str(rig.root) in path or str(rig.env.store_root) in path:
            reads.append(path)
        return real_open(file, *args, **kwargs)

    real_path_open = Path.open

    def spy_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if str(rig.root) in str(self) or str(rig.env.store_root) in str(self):
            reads.append(str(self))
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "open", spy_path_open)
    for item, true in zip(body["predictions"], truth(rig.pool[0], 40), strict=True):
        assert label(c, item["prediction_id"], true).status_code == 200
    dep = body["deployment"]["deployment_id"]
    assert cp.get(f"/deployments/{dep}/performance").json()["status"] == "ok"
    assert cp.get(f"/deployments/{dep}/feedback").status_code == 200
    assert reads == []  # no package, test result or dataset file was opened


def test_the_feedback_module_uses_no_model_dataset_or_package_loading_code() -> None:
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(feedback_module))
    imported = {
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("automl_ops")
    }
    assert imported == {"automl_ops.deployments", "automl_ops.evaluate", "automl_ops.profile"}


def test_migration_0006_keeps_existing_predictions_and_goes_down_and_up(rig: Rig) -> None:
    c, _ = ready_client(rig)
    body, _ = labelled_batch(c, rig.pool[0], 6, 3)
    assert migrate.downgrade(rig.engine, "0005").current == "0005"
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM predictions")).scalar_one() == 6
        left = conn.execute(
            text("SELECT proname FROM pg_proc WHERE proname LIKE 'feedback%'")
        ).all()
        tables = conn.execute(
            text("SELECT 1 FROM information_schema.tables WHERE table_name = 'feedback'")
        ).all()
    assert left == [] and tables == []
    assert migrate.upgrade(rig.engine).current == "0006"
    assert migrate.check(rig.engine).ok
    assert label_rows(rig) == []  # labels were dropped with the table; predictions were kept
    assert label(c, body["predictions"][0]["prediction_id"], "pos").status_code == 200
