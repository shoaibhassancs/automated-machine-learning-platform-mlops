# ruff: noqa: F811
"""P4-9: reference samples, drift monitoring, reports and alerts on real PostgreSQL.

Real sealed packages and a real dataset store; throwaway databases only; NOT RUN when the server is
unreachable. Nothing here touches `data/`.
"""

from __future__ import annotations

import builtins
import hashlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from api_support import BASE, assert_envelope
from automl_ops import monitoring as monitoring_module
from automl_ops.api_control.app import create_app as create_control
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.feedback import FeedbackRepository
from automl_ops.monitor import MonitorError, Thresholds, build_summary, js_distance, ks_two_sample
from automl_ops.monitoring import MonitoringRepository
from automl_ops.reference import ReferenceService
from automl_ops.train import TrainSettings, make_split, target_values
from conftest import CLF, Run, toy_bytes
from conftest import REG as REG_CONFIG
from test_feedback_postgres import label, labelled_batch, predict
from test_registry_postgres import pid, pool, reg, sealed_copy, unlock  # noqa: F401
from test_serving_postgres import CLF as MODEL
from test_serving_postgres import REG, Rig, ready_client, rig, rows_of  # noqa: F401

pytestmark = pytest.mark.postgres

HIGH = Thresholds()  # defaults


class Env:
    """A control-plane app over the rig's database, with reference and monitoring wired."""

    def __init__(self, rig: Rig, store_root: Path | None = None) -> None:
        self.rig = rig
        store_root = store_root or rig.env.store_root
        self.references = ReferenceService(rig.engine, store_root, rig.root)
        self.feedback = FeedbackRepository(rig.engine)
        self.monitoring = MonitoringRepository(rig.engine, self.references, self.feedback)
        settings = ApiSettings(store_root=store_root, models_root=rig.root)
        app = create_control(
            settings,
            CatalogedDatasetStore(FileDatasetStore(store_root, settings.limits), rig.engine),
            lambda: migrate.status(rig.engine),
            deployments=rig.deployments,
            feedback=self.feedback,
            monitoring=self.monitoring,
            references=self.references,
            registry=rig.registry,
        )
        self.client = TestClient(app, base_url=BASE, raise_server_exceptions=False)

    def reference(self, name: str = MODEL, version: int = 1) -> Any:
        return self.client.post(f"/models/{name}/versions/{version}/reference")

    def monitor(self, deployment_id: int, **body: Any) -> Any:
        return self.client.post(f"/deployments/{deployment_id}/monitor", json=body)


def normal_rows(rig: Rig, n: int) -> list[dict[str, Any]]:
    return rows_of(rig.pool[0], n)


def shifted_rows(rig: Rig, n: int) -> list[dict[str, Any]]:
    rows = rows_of(rig.pool[0], n)
    for row in rows:
        row["x"] = str(float(row["x"]) + 3.0)
        row["color"] = "red"
    return rows


def deployed(rig: Rig) -> tuple[TestClient, int, Env]:
    c, _ = ready_client(rig)
    active = rig.deployments.active()
    assert active is not None
    env = Env(rig)
    assert env.reference().status_code == 201
    return c, active.id, env


def alerts(rig: Rig, where: str = "TRUE") -> list[Any]:
    with rig.engine.connect() as conn:
        return list(conn.execute(text(f"SELECT * FROM alerts WHERE {where} ORDER BY id")).all())


def events(rig: Rig, alert_id: int) -> list[str]:
    with rig.engine.connect() as conn:
        return [
            r.action
            for r in conn.execute(
                text("SELECT action FROM alert_events WHERE alert_id = :i ORDER BY id"),
                {"i": alert_id},
            )
        ]


def assert_alerts_consistent(rig: Rig) -> None:
    """Replay every alert's events: the stored status and occurrence count follow from them, and at
    most one alert per (deployment, rule, scope) is unresolved."""
    with rig.engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM alerts")).all()
        for a in rows:
            acts = events(rig, a.id)
            assert acts[0] == "opened"
            assert a.occurrences == acts.count("opened") + acts.count("reoccurred")
            last = acts[-1]
            expected = {
                "acknowledged": "acknowledged",
                "resolved": "resolved",
                "auto_resolved": "resolved",
            }.get(last)
            if expected is not None:
                assert a.status == expected, (a.id, acts, a.status)
            else:
                assert a.status in ("open", "acknowledged")
        dupes = conn.execute(
            text(
                "SELECT deployment_id, rule, scope, count(*) FROM alerts "
                "WHERE status <> 'resolved' "
                "GROUP BY 1, 2, 3 HAVING count(*) > 1"
            )
        ).all()
        assert dupes == []


# -- the statistics -----------------------------------------------------------------------


def test_ks_and_js_behave_like_the_textbook_tests() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=400), rng.normal(size=400)
    d, p = ks_two_sample(a, b)
    assert d < 0.15 and p > 0.05
    d2, p2 = ks_two_sample(a, b + 1.0)
    assert d2 > 0.3 and p2 < 1e-6
    assert ks_two_sample(a, a) == (0.0, 1.0)
    with pytest.raises(MonitorError):
        ks_two_sample(a, np.array([]))
    assert js_distance(["a", "b"] * 50, ["a", "b"] * 50) == 0.0
    assert js_distance(["a"] * 10, ["b"] * 10) == pytest.approx(1.0)
    assert 0.0 < js_distance(["a"] * 9 + ["b"], ["a"] * 5 + ["b"] * 5) < 1.0


def test_thresholds_are_validated() -> None:
    for bad in (
        {"alpha": 0},
        {"alpha": 1.5},
        {"min_rows": 1},
        {"window": 99999},
        {"min_rows": True},
        {"js_distance": -1},
    ):
        with pytest.raises(MonitorError):
            Thresholds(**bad)
    assert Thresholds().to_dict()["window"] == 1000


# -- reference samples --------------------------------------------------------------------


def test_a_reference_sample_is_drawn_from_training_rows_only_and_records_its_source(
    rig: Rig,
) -> None:
    env = Env(rig)
    created = env.reference()
    assert created.status_code == 201, created.text
    body = created.json()
    run = rig.pool[0]
    manifest = run.result.manifest if hasattr(run.result, "manifest") else None
    train = TrainSettings(**{k: v for k, v in run.training.settings.to_dict().items()})
    split = make_split(target_values(run.frame, CLF), "classification", train)
    assert body["created"] is True and body["model"] == MODEL and body["version"] == 1
    assert body["package_id"] == pid(run) and body["dataset_ref"] == "toy@v1"
    assert body["dataset_sha256"] == run.dataset.sha256
    assert body["train_sha256"] == split.fingerprints()["train_sha256"]
    assert body["train_rows"] == len(split.train_rows) == body["sample_rows"] == 120
    assert body["seed"] == 42 and manifest is None or manifest is not None
    data = env.references.data(body["id"])
    assert [c["name"] for c in data.columns] == ["x", "z", "color"]
    stored = {tuple(r) for r in data.rows}
    frame = run.frame[["x", "z", "color"]]
    test_rows = {
        tuple(None if v is None or v != v else str(v) for v in row)
        for row in frame.iloc[split.test_rows].itertuples(index=False)
    }
    train_rows = {
        tuple(None if v is None or v != v else str(v) for v in row)
        for row in frame.iloc[split.train_rows].itertuples(index=False)
    }
    assert stored <= train_rows and not (stored & test_rows)  # no held-out test row, ever
    labels = rig.pool[0]
    assert len(data.predictions) == 120 and set(data.predictions) <= {"neg", "pos"} and labels


def test_a_reference_sample_is_deterministic_persistent_and_never_replaced(rig: Rig) -> None:
    env = Env(rig)
    first = env.reference().json()
    again = env.reference()
    assert again.status_code == 200 and again.json()["created"] is False
    assert (
        again.json()["id"] == first["id"]
        and again.json()["sample_sha256"] == first["sample_sha256"]
    )
    rebuilt = env.references._build(pid(rig.pool[0]))
    assert rebuilt["h"] == first["sample_sha256"]  # the same package always gives the same sample
    fetched = env.client.get(f"/models/{MODEL}/versions/1/reference").json()
    assert fetched["id"] == first["id"] and fetched["created"] is None
    second = env.reference(MODEL, 2).json()
    assert second["id"] != first["id"] and second["package_id"] == pid(rig.pool[1])
    for sql in (
        "UPDATE reference_samples SET rows = '[]'",
        "DELETE FROM reference_samples",
        "TRUNCATE reference_samples CASCADE",
        "INSERT INTO reference_samples (version_id, package_id, dataset_ref, dataset_sha256, "
        "train_sha256, seed, train_rows, sample_rows, columns, rows, predictions, sample_sha256) "
        "SELECT version_id, package_id, 'x', dataset_sha256, train_sha256, 1, 2, 1, '[]', '[]', "
        "'[]', sample_sha256 FROM reference_samples LIMIT 1",
    ):
        with pytest.raises((IntegrityError, DBAPIError)), rig.engine.begin() as conn:
            conn.execute(text(sql))


def test_making_a_reference_changes_no_package_file(rig: Rig) -> None:
    def fingerprint() -> str:
        h = hashlib.sha256()
        for path in sorted(rig.root.rglob("*")):
            if path.is_file():
                h.update(f"{path.relative_to(rig.root)}|{path.stat().st_size}|".encode())
                h.update(path.read_bytes())
        return h.hexdigest()

    before = fingerprint()
    assert Env(rig).reference().status_code == 201
    assert fingerprint() == before


def test_reference_errors_name_the_problem_and_store_nothing(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Env(rig)
    assert_envelope(env.client.post("/models/ghost/versions/1/reference"), 404, "model_not_found")
    assert_envelope(
        env.client.post(f"/models/{MODEL}/versions/99/reference"), 404, "version_not_found"
    )
    assert_envelope(
        env.client.get(f"/models/{MODEL}/versions/1/reference"), 404, "reference_not_found"
    )
    empty = Env(rig, tmp_path / "empty-store")
    assert_envelope(empty.reference(), 409, "reference_source_unavailable")
    other = FileDatasetStore(tmp_path / "other-store")
    other.ingest("toy", toy_bytes("regression"), source_filename="x.csv")  # same ref, other bytes
    changed = assert_envelope(
        Env(rig, tmp_path / "other-store").reference(), 409, "reference_source_unavailable"
    )
    assert "no longer matches" in changed["message"]
    real_split = make_split

    def other_split(labels: Any, task: Any, settings: Any) -> Any:
        return real_split(labels, task, TrainSettings(settings.test_fraction, 7, settings.folds))

    monkeypatch.setattr("automl_ops.reference.make_split", other_split)
    assert_envelope(env.reference(), 409, "reference_split_mismatch")
    monkeypatch.undo()
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM reference_samples")).scalar_one() == 0


def test_a_package_that_no_longer_verifies_gets_no_reference(
    job_env: Any, pool: list[Run], tmp_path: Path
) -> None:
    root = sealed_copy(pool[0], tmp_path / "copy")
    rig = Rig(job_env, pool, root)
    rig.register(pool[0], MODEL)
    unlock(root / pid(pool[0]) / "model.joblib").write_bytes(b"corrupt")
    response = Env(rig).reference()
    error = assert_envelope(response, 409, "package_unverified")
    assert str(root) not in response.text and error["message"]


def test_a_regression_reference_holds_numeric_predictions(
    job_env: Any, pool: list[Run], tmp_path: Path
) -> None:
    store = FileDatasetStore(tmp_path / "reg-store")
    store.ingest("toy", toy_bytes("regression"), source_filename="toy.csv")
    rig = Rig(job_env, pool)
    rig.register(pool[3], REG)
    env = Env(rig, tmp_path / "reg-store")
    created = env.reference(REG, 1)
    assert created.status_code == 201, created.text
    data = env.references.data(created.json()["id"])
    assert all(isinstance(p, float) for p in data.predictions) and len(data.predictions) == 120
    assert REG_CONFIG.task == "regression"


# -- monitoring: verdicts -----------------------------------------------------------------


def test_monitoring_needs_a_reference_and_says_so(rig: Rig) -> None:
    c, _ = ready_client(rig)
    dep = rig.deployments.active().id  # type: ignore[union-attr]
    predict(c, normal_rows(rig, 40))
    env = Env(rig)
    assert_envelope(env.monitor(dep), 409, "no_reference")
    assert_envelope(env.monitor(99999), 404, "deployment_not_found")
    status = env.client.get(f"/deployments/{dep}/monitoring").json()
    assert status["reference_id"] is None and status["latest_real"] is None


def test_too_little_data_is_a_status_not_a_drift_result(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, normal_rows(rig, 10))
    report = env.monitor(dep)
    assert report.status_code == 201, report.text
    body = report.json()
    assert body["status"] == "insufficient_data" and body["window_rows"] == 10
    assert "at least 30 rows" in body["summary"]["reason"] and body["summary"]["rules"] == []
    assert "features" not in body["summary"] and alerts(rig) == []
    empty = env.monitor(dep, scope="simulated").json()  # no simulated traffic at all
    assert empty["status"] == "insufficient_data" and empty["window_rows"] == 0


def test_unshifted_traffic_raises_no_alert_and_shifted_traffic_does(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, normal_rows(rig, 60))
    calm = env.monitor(dep).json()
    assert calm["status"] == "ok", calm
    summary = calm["summary"]
    assert [f["name"] for f in summary["features"]] == ["x", "z", "color"]
    assert summary["reference"]["sample_rows"] == 120 and summary["traffic"]["scope"] == "real"
    assert not summary["dataset_drift"]["drifted"] and alerts(rig) == []
    predict(c, shifted_rows(rig, 60))
    hot = env.monitor(dep, thresholds={"window": 60}).json()
    drift = hot["summary"]["dataset_drift"]
    by_name = {f["name"]: f for f in hot["summary"]["features"]}
    assert by_name["x"]["drifted"] and by_name["color"]["drifted"] and not by_name["z"]["drifted"]
    assert (
        drift["drifted"]
        and drift["drifted_features"] == 2
        and drift["share"] == pytest.approx(2 / 3, abs=1e-5)
    )
    rows = alerts(rig)
    assert [
        (a.rule, a.scope, a.status, a.severity) for a in rows if a.rule == "drifted_feature_share"
    ] == [("drifted_feature_share", "real", "open", "warning")]
    assert hot["thresholds"]["window"] == 60 and calm["thresholds"]["window"] == 1000
    assert_alerts_consistent(rig)


def test_a_repeated_condition_updates_one_alert_and_recovery_resolves_it(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    for _ in range(3):
        assert env.monitor(dep).json()["status"] == "ok"
    (alert,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    assert alert.occurrences == 3 and alert.status == "open"
    assert events(rig, alert.id) == ["opened", "reoccurred", "reoccurred"]
    predict(c, normal_rows(rig, 60))
    cleared = env.monitor(dep, thresholds={"window": 60}).json()
    assert cleared["status"] == "ok"
    (alert,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    assert alert.status == "resolved" and alert.resolved_at is not None
    assert events(rig, alert.id)[-1] == "auto_resolved"
    # it comes back: a NEW alert opens, the resolved one stays exactly as it was
    predict(c, shifted_rows(rig, 60))
    env.monitor(dep, thresholds={"window": 60})
    drifted = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    assert [a.status for a in drifted] == ["resolved", "open"] and drifted[0].id != drifted[1].id
    assert drifted[0].occurrences == alert.occurrences
    assert_alerts_consistent(rig)


def test_people_acknowledge_and_resolve_alerts_with_an_audit_trail(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    env.monitor(dep)
    (alert,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    url = f"/alerts/{alert.id}"
    acked = env.client.post(f"{url}/acknowledge", json={"reason": "looking into it"})
    assert acked.status_code == 200 and acked.json()["status"] == "acknowledged"
    assert [e["action"] for e in acked.json()["events"]] == ["opened", "acknowledged"]
    assert_envelope(
        env.client.post(f"{url}/acknowledge", json={"reason": "again"}),
        409,
        "invalid_alert_transition",
    )
    env.monitor(dep)  # an acknowledged alert is still the one unresolved alert: it just recurs
    assert env.client.get(url).json()["occurrences"] == 2
    assert env.client.get(url).json()["status"] == "acknowledged"
    for bad in ({"reason": ""}, {"reason": "x" * 501}, {}):
        assert_envelope(env.client.post(f"{url}/resolve", json=bad), 422, "validation_error")
    done = env.client.post(f"{url}/resolve", json={"reason": "fixed upstream"})
    assert done.status_code == 200 and done.json()["status"] == "resolved"
    assert done.json()["events"][-1]["reason"] == "fixed upstream"
    assert_envelope(
        env.client.post(f"{url}/resolve", json={"reason": "twice"}), 409, "invalid_alert_transition"
    )
    assert_envelope(
        env.client.post(f"{url}/acknowledge", json={"reason": "late"}),
        409,
        "invalid_alert_transition",
    )
    assert_envelope(
        env.client.post("/alerts/99999/resolve", json={"reason": "r"}), 404, "alert_not_found"
    )
    assert_alerts_consistent(rig)


def test_real_and_simulated_traffic_never_mix(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, normal_rows(rig, 60))
    predict(c, shifted_rows(rig, 60), simulated=True)
    real = env.monitor(dep, scope="real").json()
    assert real["status"] == "ok" and real["window_rows"] == 60
    assert not real["summary"]["dataset_drift"]["drifted"] and alerts(rig) == []
    sim = env.monitor(dep, scope="simulated").json()
    assert sim["window_rows"] == 60 and sim["summary"]["traffic"]["simulated"] is True
    assert "simulated" in sim["summary"]["traffic"]["note"]
    rows = alerts(rig)
    assert rows and {a.scope for a in rows} == {"simulated"}
    # real traffic turning bad later is a separate alert; the two scopes dedupe independently
    predict(c, shifted_rows(rig, 60))
    env.monitor(dep, scope="real", thresholds={"window": 60})
    scopes = sorted({(a.rule, a.scope) for a in alerts(rig) if a.rule == "drifted_feature_share"})
    assert scopes == [("drifted_feature_share", "real"), ("drifted_feature_share", "simulated")]
    listed = env.client.get("/alerts", params={"scope": "simulated"}).json()["alerts"]
    assert listed and all(a["scope"] == "simulated" for a in listed)
    status = env.client.get(f"/deployments/{dep}/monitoring").json()
    assert (
        status["latest_real"]["scope"] == "real"
        and status["latest_simulated"]["scope"] == "simulated"
    )
    assert_alerts_consistent(rig)


def test_prediction_shift_is_detected_in_classification(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    rows = shifted_rows(rig, 80)
    for row in rows:
        row["x"] = "9.5"  # pushes every prediction to the same class
    predict(c, rows)
    body = env.monitor(dep).json()
    prediction = body["summary"]["prediction"]
    assert prediction["method"] == "js" and set(prediction["reference"]) == {"neg", "pos"}
    assert prediction["drifted"] and any(
        a.rule == "prediction_shift" and a.status == "open" for a in alerts(rig)
    )


def test_a_performance_drop_alert_needs_enough_labels_and_a_real_drop(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    body, truth_labels = labelled_batch(c, rig.pool[0], 60, 0)
    few = predict(c, normal_rows(rig, 10))
    wrong = {"neg": "pos", "pos": "neg"}
    for item, true in zip(few["predictions"], truth_labels, strict=False):
        label(c, item["prediction_id"], wrong[true])
    first = env.monitor(dep, thresholds={"min_labelled": 30}).json()
    assert first["summary"]["performance"]["evaluated"] is False  # 10 labels: no verdict
    assert not [a for a in alerts(rig) if a.rule == "performance_drop"]
    for item, true in zip(body["predictions"][:45], truth_labels[:45], strict=True):
        label(c, item["prediction_id"], wrong[true])
    second = env.monitor(dep).json()
    perf = second["summary"]["performance"]
    assert perf["evaluated"] and perf["triggered"] and perf["metric"] == "f1_macro"
    assert perf["relative_worsening"] > 0.10 and perf["live"] < perf["offline"]
    (alert,) = [a for a in alerts(rig) if a.rule == "performance_drop"]
    assert alert.severity == "critical" and alert.status == "open"
    assert_alerts_consistent(rig)


# -- failures and interruptions -----------------------------------------------------------


def test_a_failed_run_is_recorded_as_failed_and_changes_no_alert(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    env.monitor(dep)
    (before,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("statistics library exploded")

    monkeypatch.setattr("automl_ops.monitoring.build_summary", boom)
    failed = env.monitor(dep)
    assert failed.status_code == 201
    body = failed.json()
    assert body["status"] == "failed" and body["summary"] is None
    assert "RuntimeError" in body["error"] and "exploded" in body["error"]
    (after,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    assert (after.status, after.occurrences) == (before.status, before.occurrences)
    monkeypatch.undo()
    assert env.monitor(dep).json()["status"] == "ok"  # and it recovers on the next run
    statuses = [
        r["status"] for r in env.client.get(f"/deployments/{dep}/reports").json()["reports"]
    ]
    assert statuses == ["ok", "failed", "ok"]


def test_an_interrupted_run_leaves_no_partial_report_or_alert(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    real_apply = MonitoringRepository._apply_rule
    calls = {"n": 0}

    def crash_second(conn: Any, *a: Any, **k: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt  # not an Exception: nothing may catch and "complete" the run
        real_apply(conn, *a, **k)

    monkeypatch.setattr(MonitoringRepository, "_apply_rule", staticmethod(crash_second))
    with pytest.raises(KeyboardInterrupt):
        env.monitoring.run(dep)
    monkeypatch.undo()
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM monitoring_reports")).scalar_one() == 0
    assert alerts(rig) == []
    assert env.monitor(dep).json()["status"] == "ok" and alerts(rig)  # and the lock was released
    assert_alerts_consistent(rig)


def test_a_database_outage_is_a_clear_503_and_stores_nothing(rig: Rig) -> None:
    from sqlalchemy import create_engine

    c, dep, env = deployed(rig)
    predict(c, normal_rows(rig, 40))
    dead = create_engine(
        "postgresql+psycopg://nobody:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    broken = MonitoringRepository(dead, env.references, env.feedback)
    with pytest.raises(monitoring_module.MonitoringUnavailable):
        broken.run(dep)
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM monitoring_reports")).scalar_one() == 0


# -- concurrency --------------------------------------------------------------------------


def test_a_second_run_for_the_same_deployment_is_refused_while_one_is_running(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    started, release = threading.Event(), threading.Event()
    real_build = build_summary

    def slow(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(20)
        return real_build(*args, **kwargs)

    monkeypatch.setattr("automl_ops.monitoring.build_summary", slow)
    first: list[Any] = []
    thread = threading.Thread(target=lambda: first.append(env.monitor(dep)))
    thread.start()
    assert started.wait(20)
    assert_envelope(env.monitor(dep), 409, "monitoring_in_progress")
    release.set()
    thread.join(30)
    assert first[0].status_code == 201 and first[0].json()["status"] == "ok"
    with rig.engine.connect() as conn:  # the lock is released afterwards
        assert conn.execute(text("SELECT count(*) FROM monitoring_reports")).scalar_one() == 1
    assert env.monitor(dep).status_code == 201


def test_racing_runs_never_duplicate_alerts_or_corrupt_state(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    barrier = threading.Barrier(6)

    def go() -> Any:
        barrier.wait()
        return env.monitor(dep)

    with ThreadPoolExecutor(6) as pool_:
        results = list(pool_.map(lambda _: go(), range(6)))
    assert {r.status_code for r in results} <= {201, 409}
    assert any(r.status_code == 201 for r in results)
    for r in results:
        if r.status_code == 409:
            assert r.json()["error"]["code"] == "monitoring_in_progress"
    ok = sum(r.status_code == 201 for r in results)
    (alert,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    assert alert.occurrences == ok
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM monitoring_reports")).scalar_one() == ok
    assert_alerts_consistent(rig)


def test_simultaneous_acknowledge_and_resolve_have_a_consistent_outcome(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    env.monitor(dep)
    (alert,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    calls: list[Callable[[], Any]] = [
        partial(env.client.post, f"/alerts/{alert.id}/resolve", json={"reason": "a"}),
        partial(env.client.post, f"/alerts/{alert.id}/resolve", json={"reason": "b"}),
        partial(env.client.post, f"/alerts/{alert.id}/acknowledge", json={"reason": "c"}),
    ]
    barrier = threading.Barrier(3)

    def go(call: Callable[[], Any]) -> Any:
        barrier.wait()
        return call()

    with ThreadPoolExecutor(3) as pool_:
        out = list(pool_.map(go, calls))
    assert sum(r.status_code == 200 and r.json()["status"] == "resolved" for r in out) == 1
    assert_alerts_consistent(rig)


# -- serving is never affected ------------------------------------------------------------


def test_monitoring_failures_and_load_never_interrupt_serving(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, dep, env = deployed(rig)
    predict(c, normal_rows(rig, 40))
    stop = threading.Event()
    bad: list[str] = []
    served = [0]

    def hammer() -> None:
        while not stop.is_set():
            r = c.post("/predict", json={"rows": normal_rows(rig, 2)})
            if r.status_code != 200 or not r.json()["logged"]:
                bad.append(r.text)
            served[0] += 1

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("monitor broke")

    threads = [threading.Thread(target=hammer) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        env.monitor(dep)
        monkeypatch.setattr("automl_ops.monitoring.build_summary", boom)
        assert env.monitor(dep).json()["status"] == "failed"
        env.client.get(f"/deployments/{dep}/monitoring")
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert bad == [] and served[0] > 5


def test_a_monitoring_run_opens_no_model_dataset_or_package_file(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    reads: list[str] = []
    real_open, real_path_open = builtins.open, Path.open

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(rig.root) in str(file) or str(rig.env.store_root) in str(file):
            reads.append(str(file))
        return real_open(file, *args, **kwargs)

    def spy_path(self: Path, *args: Any, **kwargs: Any) -> Any:
        if str(rig.root) in str(self) or str(rig.env.store_root) in str(self):
            reads.append(str(self))
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "open", spy_path)
    assert env.monitor(dep).json()["status"] == "ok"
    assert env.client.get(f"/deployments/{dep}/monitoring").status_code == 200
    assert reads == []


def test_monitoring_code_never_trains_loads_models_or_reads_datasets() -> None:
    import ast
    import inspect

    for module in (monitoring_module, __import__("automl_ops.monitor", fromlist=["x"])):
        tree = ast.parse(inspect.getsource(module))
        imported = {
            n.module
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("automl_ops")
        }
        forbidden = {
            "automl_ops.train",
            "automl_ops.package",
            "automl_ops.finalize",
            "automl_ops.dataset_store",
            "automl_ops.selection",
            "automl_ops.runner",
        }
        assert not (imported & forbidden), imported & forbidden


# -- reports, history, persistence --------------------------------------------------------


def test_reports_are_immutable_and_history_stays_traceable(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    first = env.monitor(dep).json()
    stored = env.client.get(f"/reports/{first['id']}").json()
    assert stored == first and stored["reference_id"] == env.references.get(MODEL, 1).id
    predict(c, normal_rows(rig, 60))
    env.monitor(dep, thresholds={"window": 60})
    assert env.client.get(f"/reports/{first['id']}").json() == first  # history never rewritten
    for sql in (
        "UPDATE monitoring_reports SET status = 'ok'",
        "DELETE FROM monitoring_reports",
        "TRUNCATE monitoring_reports CASCADE",
        "DELETE FROM reference_samples",
    ):
        with pytest.raises((IntegrityError, DBAPIError)), rig.engine.begin() as conn:
            conn.execute(text(sql))
    page = env.client.get(f"/deployments/{dep}/reports", params={"limit": 1}).json()
    assert len(page["reports"]) == 1 and page["reports"][0]["summary"] is None
    assert page["next_before_id"] == page["reports"][0]["id"]
    rest = env.client.get(
        f"/deployments/{dep}/reports", params={"limit": 5, "before_id": page["next_before_id"]}
    ).json()
    assert [r["id"] for r in rest["reports"]] == [first["id"]]
    assert_envelope(env.client.get("/reports/99999"), 404, "report_not_found")
    assert_envelope(env.client.get("/deployments/99999/reports"), 404, "deployment_not_found")


def test_the_database_enforces_report_and_alert_rules(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    report = env.monitor(dep).json()
    (alert,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    ref = report["reference_id"]
    now = "now()"
    bad = (
        # a failed report must carry an error; a successful one a summary and no error
        f"INSERT INTO monitoring_reports (deployment_id, reference_id, scope, status, window_rows, "
        f"thresholds, started_at) VALUES ({dep}, {ref}, 'real', 'failed', 0, '{{}}', {now})",
        f"INSERT INTO monitoring_reports (deployment_id, reference_id, scope, status, window_rows, "
        f"thresholds, started_at, error) VALUES ({dep}, {ref}, 'real', 'ok', 0, '{{}}', "
        f"{now}, 'x')",
        # a second unresolved alert for the same condition
        f"INSERT INTO alerts (deployment_id, rule, scope, severity, details, first_report_id, "
        f"last_report_id) VALUES ({dep}, 'drifted_feature_share', 'real', 'warning', '{{}}', "
        f"{report['id']}, {report['id']})",
        f"UPDATE alerts SET rule = 'prediction_shift' WHERE id = {alert.id}",
        f"UPDATE alerts SET status = 'resolved' WHERE id = {alert.id}",  # needs resolved_at
        f"UPDATE alerts SET occurrences = 0 WHERE id = {alert.id}",
        f"DELETE FROM alerts WHERE id = {alert.id}",
        "TRUNCATE alerts CASCADE",
        "UPDATE alert_events SET action = 'resolved'",
        "DELETE FROM alert_events",
        "TRUNCATE alert_events",
    )
    for sql in bad:
        with pytest.raises((IntegrityError, DBAPIError)), rig.engine.begin() as conn:
            conn.execute(text(sql))
    with rig.engine.begin() as conn:  # a legal move works, an illegal one afterwards does not
        conn.execute(text(f"UPDATE alerts SET status = 'acknowledged' WHERE id = {alert.id}"))
        conn.execute(
            text(
                f"UPDATE alerts SET status = 'resolved', resolved_at = now() WHERE id = {alert.id}"
            )
        )
    for sql in (
        f"UPDATE alerts SET status = 'open', resolved_at = NULL WHERE id = {alert.id}",
        f"UPDATE alerts SET occurrences = 9 WHERE id = {alert.id}",
    ):
        with pytest.raises((IntegrityError, DBAPIError)), rig.engine.begin() as conn:
            conn.execute(text(sql))


def test_reports_alerts_and_references_survive_a_restart(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    predict(c, shifted_rows(rig, 60))
    report = env.monitor(dep).json()
    fresh = Env(rig)  # brand-new repositories and app on the same database
    assert fresh.client.get(f"/reports/{report['id']}").json() == report
    assert (
        fresh.references.get(MODEL, 1).sample_sha256 == env.references.get(MODEL, 1).sample_sha256
    )
    assert fresh.client.get("/alerts", params={"status": "open"}).json()["alerts"]
    assert fresh.reference().json()["created"] is False
    again = fresh.monitor(dep).json()
    (alert,) = [a for a in alerts(rig) if a.rule == "drifted_feature_share"]
    assert alert.occurrences == 2 and again["id"] > report["id"]


# -- API behaviour ------------------------------------------------------------------------


def test_api_validation_security_and_availability(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    cl = env.client
    for body in (
        {"scope": "bogus"},
        {"thresholds": {"alpha": 0}},
        {"thresholds": {"window": 4}},
        {"thresholds": {"min_rows": 1.5}},
        {"thresholds": {"nope": 1}},
        {"extra": 1},
    ):
        assert_envelope(cl.post(f"/deployments/{dep}/monitor", json=body), 422, "validation_error")
    assert_envelope(cl.post("/deployments/abc/monitor", json={}), 422, "validation_error")
    assert_envelope(cl.get("/alerts", params={"status": "bogus"}), 422, "validation_error")
    assert_envelope(cl.get("/alerts", params={"rule": "bogus"}), 422, "validation_error")
    assert_envelope(cl.get("/alerts/abc"), 422, "validation_error")
    assert_envelope(cl.get("/alerts/99999"), 404, "alert_not_found")
    assert_envelope(
        cl.post(f"/deployments/{dep}/monitor", json={}, headers={"origin": "http://evil.example"}),
        403,
    )
    assert_envelope(
        cl.post(f"/deployments/{dep}/monitor", json={}, headers={"host": "evil.example"}), 400
    )
    assert_envelope(
        cl.post(
            f"/deployments/{dep}/monitor", content=b"x", headers={"content-type": "text/plain"}
        ),
        415,
    )
    assert_envelope(cl.delete("/alerts/1"), 405)
    off = create_control(
        ApiSettings(store_root=rig.env.store_root, models_root=rig.root),
        CatalogedDatasetStore(FileDatasetStore(rig.env.store_root), rig.engine),
        lambda: migrate.status(rig.engine),
    )
    disabled = TestClient(off, base_url=BASE, raise_server_exceptions=False)
    for response in (
        disabled.get(f"/deployments/{dep}/monitoring"),
        disabled.get("/alerts"),
        disabled.post(f"/deployments/{dep}/monitor", json={}),
        disabled.post(f"/models/{MODEL}/versions/1/reference"),
    ):
        assert_envelope(response, 503, "monitoring_unavailable")
    assert alerts(rig) == []


def test_migration_0007_keeps_predictions_and_feedback_and_goes_down_and_up(rig: Rig) -> None:
    c, dep, env = deployed(rig)
    body, _ = labelled_batch(c, rig.pool[0], 6, 3)
    assert migrate.downgrade(rig.engine, "0006").current == "0006"
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM predictions")).scalar_one() == 6
        assert conn.execute(text("SELECT count(*) FROM feedback")).scalar_one() == 3
        left = conn.execute(
            text(
                "SELECT proname FROM pg_proc "
                "WHERE proname IN ('alerts_guard', 'immutable_rows_guard')"
            )
        ).all()
        tables = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_name IN "
                "('reference_samples', 'monitoring_reports', 'alerts', 'alert_events')"
            )
        ).all()
    assert left == [] and tables == []
    assert migrate.upgrade(rig.engine).current == "0007" and migrate.check(rig.engine).ok
    assert body["deployment"]["deployment_id"] == dep


def test_regression_monitoring_compares_prediction_distributions(
    job_env: Any, pool: list[Run], tmp_path: Path
) -> None:
    FileDatasetStore(tmp_path / "reg-store").ingest(
        "toy", toy_bytes("regression"), source_filename="toy.csv"
    )
    rig = Rig(job_env, pool)
    rig.register(pool[3], REG)
    c, _ = ready_client(rig, REG, 1)
    dep = rig.deployments.active().id  # type: ignore[union-attr]
    env = Env(rig, tmp_path / "reg-store")
    assert env.reference(REG, 1).status_code == 201
    rows = rows_of(pool[3], 60)
    predict(c, rows)
    calm = env.monitor(dep).json()
    assert calm["status"] == "ok" and calm["summary"]["prediction"]["method"] == "ks"
    assert not calm["summary"]["prediction"]["drifted"] and alerts(rig) == []
    shifted = [{**row, "x": str(float(row["x"]) + 6.0)} for row in rows]
    predict(c, shifted)
    hot = env.monitor(dep, thresholds={"window": 60}).json()
    prediction = hot["summary"]["prediction"]
    assert prediction["drifted"] and prediction["current"]["mean"] > prediction["reference"]["mean"]
    assert {a.rule for a in alerts(rig)} >= {"prediction_shift"}
    assert hot["summary"]["performance"]["evaluated"] is False  # no labels yet: no verdict
    assert_alerts_consistent(rig)
