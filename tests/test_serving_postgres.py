# ruff: noqa: F811
"""P4-7: the serving app on real PostgreSQL with real sealed packages (throwaway databases).

NOT RUN when the server is unreachable. Nothing here touches `data/` or the real database.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text

from api_support import assert_envelope
from automl_ops.api_serving.app import create_app
from automl_ops.api_serving.runtime import LoadFailure, ServingRuntime
from automl_ops.api_serving.settings import ServingSettings
from automl_ops.db import migrate
from automl_ops.deployments import DeploymentRepository
from automl_ops.package import load_package
from automl_ops.registry import RegistryRepository
from conftest import JobEnv, Run
from test_registry_postgres import pid, pool, reg, sealed_copy, unlock  # noqa: F401

pytestmark = pytest.mark.postgres

PORT = 8200
BASE = f"http://127.0.0.1:{PORT}"
CLF = "churn"
REG = "price"


class Rig:
    def __init__(self, env: JobEnv, pool: list[Run], root: Path | None = None) -> None:
        self.env = env
        self.pool = pool
        self.root = root or pool[0].root
        self.engine = env.engine
        self.registry = RegistryRepository(env.engine, self.root)
        self.deployments = DeploymentRepository(env.engine, self.root)

    def register(self, run: Run, name: str) -> int:
        version = reg(self.registry, run, name).version
        self.registry.promote(name, version, "staging", "ready")
        return int(version)

    def runtime(self, **kw: Any) -> ServingRuntime:
        return ServingRuntime(self.engine, self.root, retry_seconds=kw.pop("retry", 1.0))

    def client(self, runtime: ServingRuntime, *, poll: float = 60.0, **kw: Any) -> TestClient:
        app = create_app(
            ServingSettings(port=PORT, models_root=self.root, poll_seconds=poll, **kw), runtime
        )
        return TestClient(app, base_url=BASE, raise_server_exceptions=False)


@pytest.fixture
def rig(job_env: JobEnv, pool: list[Run]) -> Rig:
    r = Rig(job_env, pool)
    assert r.register(pool[0], CLF) == 1
    assert r.register(pool[1], CLF) == 2
    r.register(pool[3], REG)
    return r


def rows_of(run: Run, count: int = 8) -> list[dict[str, Any]]:
    frame = run.frame.head(count)[["x", "z", "color"]]
    names = list(frame.columns)
    return [
        {name: (None if pd.isna(v) else str(v)) for name, v in zip(names, row, strict=True)}
        for row in frame.itertuples(index=False)
    ]


def expected(root: Path, run: Run, rows: list[dict[str, Any]]) -> Any:
    package = load_package(root, pid(run))
    frame = pd.DataFrame(rows, columns=package.feature_columns, dtype="string")
    labels = package.predict(frame)
    proba = package.predict_proba(frame) if package.classes is not None else None
    return labels, proba


def ready_client(rig: Rig, name: str = CLF, version: int = 1) -> tuple[TestClient, ServingRuntime]:
    rig.deployments.deploy(name, version, "go live")
    runtime = rig.runtime()
    assert runtime.refresh().ready
    return rig.client(runtime), runtime


# -- nothing deployed ---------------------------------------------------------------------------


def test_with_nothing_deployed_the_app_is_alive_but_not_ready(rig: Rig) -> None:
    runtime = rig.runtime()
    runtime.refresh()
    c = rig.client(runtime)
    health = c.get("/health")
    assert health.status_code == 200
    assert health.json()["ready"] is False and health.json()["model_state"] == "no_deployment"
    assert health.json()["database"] == "ok"
    for method, path in (("get", "/ready"), ("get", "/model")):
        assert_envelope(getattr(c, method)(path), 503)
    body = assert_envelope(c.post("/predict", json={"rows": [{}]}), 503, "no_deployment")
    assert "deployed" in body["message"]


def test_before_the_first_refresh_the_app_says_it_is_starting_not_ready(rig: Rig) -> None:
    c = rig.client(rig.runtime())
    assert c.get("/health").json()["model_state"] == "starting"
    assert_envelope(c.get("/ready"), 503)
    assert_envelope(c.post("/predict", json={"rows": [{}]}), 503, "model_not_ready")


# -- parity and the response format ---------------------------------------------------------------


def test_served_classifications_equal_the_saved_pipelines_predictions(rig: Rig) -> None:
    c, _ = ready_client(rig)
    rows = rows_of(rig.pool[0])
    response = c.post("/predict", json={"rows": rows})
    assert response.status_code == 200, response.text
    body = response.json()
    labels, proba = expected(rig.root, rig.pool[0], rows)
    assert [p["prediction"] for p in body["predictions"]] == [str(v) for v in labels]
    classes = load_package(rig.root, pid(rig.pool[0])).classes
    assert classes is not None
    for item, row_proba in zip(body["predictions"], proba, strict=True):
        assert list(item["probabilities"]) == classes
        np.testing.assert_allclose(list(item["probabilities"].values()), row_proba, atol=1e-12)
        assert abs(sum(item["probabilities"].values()) - 1) < 1e-9
    assert body["deployment"]["model"] == CLF and body["deployment"]["version"] == 1
    assert body["deployment"]["task"] == "classification" and body["logged"] is True
    assert len({p["prediction_id"] for p in body["predictions"]}) == len(rows)
    assert body["latency_ms"] >= 0


def test_served_regressions_equal_the_saved_pipelines_predictions(rig: Rig) -> None:
    c, _ = ready_client(rig, REG, 1)
    rows = rows_of(rig.pool[3])
    body = c.post("/predict", json={"rows": rows}).json()
    labels, _ = expected(rig.root, rig.pool[3], rows)
    np.testing.assert_allclose([p["prediction"] for p in body["predictions"]], labels, atol=1e-12)
    assert all(p["probabilities"] is None for p in body["predictions"])
    assert body["deployment"]["task"] == "regression"


def test_missing_values_and_unseen_categories_still_predict(rig: Rig) -> None:
    c, _ = ready_client(rig)
    rows = [
        {"x": None, "z": None, "color": None},
        {"x": "", "z": 0.5, "color": "never-seen-before"},
    ]
    response = c.post("/predict", json={"rows": rows})
    assert response.status_code == 200 and len(response.json()["predictions"]) == 2


def test_model_info_lists_the_input_schema(rig: Rig) -> None:
    c, _ = ready_client(rig)
    info = c.get("/model").json()
    assert info["input_columns"] == [
        {"name": "x", "kind": "numeric"},
        {"name": "z", "kind": "numeric"},
        {"name": "color", "kind": "categorical"},
    ]
    assert info["classes"] == load_package(rig.root, pid(rig.pool[0])).classes
    assert (info["deployment"]["version"], info["registry_status"], info["max_rows"]) == (
        1,
        "staging",
        100,
    )
    assert info["target"] == "y" and info["deployment"]["package_id"] == pid(rig.pool[0])
    assert c.get("/ready").status_code == 200


# -- input validation ------------------------------------------------------------------------------


def test_wrong_columns_are_a_clear_422_that_names_them(rig: Rig) -> None:
    c, _ = ready_client(rig)
    body = assert_envelope(
        c.post(
            "/predict", json={"rows": [{"x": 1, "z": 2}, {"x": 1, "z": 2, "color": "a", "q": 1}]}
        ),
        422,
        "invalid_input",
    )
    problems = body["details"]["problems"]
    assert {"row": 0, "column": None, "message": "missing column(s): ['color']"} in problems
    assert {"row": 1, "column": None, "message": "unknown column(s): ['q']"} in problems
    assert rows_logged(rig) == 0  # a refused request logs nothing


def test_bad_values_and_bad_request_shapes_are_refused(rig: Rig) -> None:
    c, _ = ready_client(rig)
    bad_number = c.post("/predict", json={"rows": [{"x": "abc", "z": 1, "color": "a"}]})
    problems = assert_envelope(bad_number, 422, "invalid_input")["details"]["problems"]
    assert problems[0]["column"] == "x" and "clean number" in problems[0]["message"]
    payloads: list[Any] = [
        {"rows": []},
        {"rows": [{"x": 1, "z": 1, "color": "a"}] * 101},
        {"rows": "nope"},
        {"rows": [[1, 2, 3]]},
        {"rows": [{"x": 1, "z": 1, "color": "a"}], "extra": 1},
        {},
    ]
    for payload in payloads:
        assert_envelope(c.post("/predict", json=payload), 422, "validation_error")
    assert_envelope(
        c.post("/predict", content=b"{not json", headers={"content-type": "application/json"}), 422
    )
    assert rows_logged(rig) == 0


def test_the_request_rules_of_the_control_plane_apply_here_too(rig: Rig) -> None:
    c, _ = ready_client(rig)
    row = {"rows": [{"x": 1, "z": 1, "color": "a"}]}
    assert_envelope(c.post("/predict", json=row, headers={"host": "evil.example"}), 400)
    assert_envelope(c.post("/predict", json=row, headers={"origin": "http://evil.example"}), 403)
    assert_envelope(c.post("/reload", headers={"origin": "http://evil.example"}), 403)
    assert_envelope(c.post("/predict", content=b"x=1", headers={"content-type": "text/plain"}), 415)
    assert_envelope(
        c.post(
            "/predict",
            content=b" " * (1024 * 1024 + 10),
            headers={"content-type": "application/json"},
        ),
        413,
    )
    assert c.post("/predict", json=row).status_code == 200


# -- the prediction log ----------------------------------------------------------------------------


def rows_logged(rig: Rig) -> int:
    with rig.engine.connect() as connection:
        return int(connection.execute(text("SELECT count(*) FROM predictions")).scalar_one())


def test_each_row_is_logged_with_input_output_version_latency_and_the_simulated_label(
    rig: Rig,
) -> None:
    c, _ = ready_client(rig)
    rows = rows_of(rig.pool[0], 3)
    plain = c.post("/predict", json={"rows": rows}).json()
    simulated = c.post("/predict", json={"rows": rows[:1], "simulated": True}).json()
    with rig.engine.connect() as connection:
        logged = connection.execute(
            text(
                "SELECT id, deployment_id, model_version, inputs, output, latency_ms, is_simulated "
                "FROM predictions ORDER BY created_at, id"
            )
        ).all()
    assert len(logged) == 4
    by_id = {r.id: r for r in logged}
    for row, item in zip(rows, plain["predictions"], strict=True):
        record = by_id[item["prediction_id"]]
        assert record.inputs == row and record.is_simulated is False
        assert record.output["prediction"] == item["prediction"]
        assert record.output["probabilities"] == item["probabilities"]
        assert (
            record.model_version == 1
            and record.deployment_id == plain["deployment"]["deployment_id"]
        )
        assert record.latency_ms == pytest.approx(plain["latency_ms"])
    assert by_id[simulated["predictions"][0]["prediction_id"]].is_simulated is True


def test_a_logging_failure_never_fails_the_prediction(rig: Rig) -> None:
    c, runtime = ready_client(rig)
    healthy_engine = runtime.engine
    runtime.engine = create_engine(
        "postgresql+psycopg://nobody:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    rows = rows_of(rig.pool[0], 2)
    response = c.post("/predict", json={"rows": rows})
    assert response.status_code == 200
    body = response.json()
    labels, _ = expected(rig.root, rig.pool[0], rows)
    assert body["logged"] is False
    assert [p["prediction"] for p in body["predictions"]] == [str(v) for v in labels]
    # the database being unreachable is reported, and the verified model keeps serving
    health = c.post("/reload").json()
    assert health["database"] == "unavailable" and health["ready"] is True
    assert c.post("/predict", json={"rows": rows}).status_code == 200
    runtime.engine = healthy_engine
    assert c.post("/reload").json()["database"] == "ok"
    assert c.post("/predict", json={"rows": rows}).json()["logged"] is True


def test_a_failed_insert_for_one_reason_or_another_is_still_a_valid_prediction(rig: Rig) -> None:
    c, _ = ready_client(rig)
    with rig.engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE predictions ADD CONSTRAINT never CHECK (false) NOT VALID")
        )
    response = c.post("/predict", json={"rows": rows_of(rig.pool[0], 1)})
    assert response.status_code == 200 and response.json()["logged"] is False
    assert rows_logged(rig) == 0


# -- switching, rollback, undeploy, polling --------------------------------------------------------


def test_switching_changes_the_model_that_answers_and_is_visible_in_model(rig: Rig) -> None:
    c, runtime = ready_client(rig)
    rows = rows_of(rig.pool[0])
    first = c.post("/predict", json={"rows": rows}).json()
    assert first["deployment"]["version"] == 1
    active = rig.deployments.active()
    assert active is not None
    rig.deployments.switch(active.id, CLF, 2, "try the forest")
    # without a reload the app still serves the previous deployment, until it polls or is told
    assert c.get("/model").json()["deployment"]["version"] == 1
    health = c.post("/reload").json()
    assert health["ready"] and health["loaded_deployment_id"] == health["desired_deployment_id"]
    assert c.get("/model").json()["deployment"]["version"] == 2
    second = c.post("/predict", json={"rows": rows}).json()
    assert second["deployment"]["version"] == 2
    labels, proba = expected(rig.root, rig.pool[1], rows)
    assert [p["prediction"] for p in second["predictions"]] == [str(v) for v in labels]
    np.testing.assert_allclose(
        [list(p["probabilities"].values()) for p in second["predictions"]], proba, atol=1e-12
    )
    assert runtime.health().loaded_deployment_id == rig.deployments.active().id  # type: ignore[union-attr]


def test_rollback_restores_the_previous_model_and_undeploy_stops_serving(rig: Rig) -> None:
    c, _ = ready_client(rig)
    first = rig.deployments.active()
    assert first is not None
    rig.deployments.switch(first.id, CLF, 2, "b")
    c.post("/reload")
    assert c.get("/model").json()["deployment"]["version"] == 2
    back = rig.deployments.rollback("b misbehaves")
    c.post("/reload")
    info = c.get("/model").json()
    assert info["deployment"]["version"] == 1 and info["deployment"]["deployment_id"] == back.id
    rig.deployments.undeploy(back.id, "maintenance")
    assert c.post("/reload").json()["model_state"] == "no_deployment"
    assert_envelope(
        c.post("/predict", json={"rows": rows_of(rig.pool[0], 1)}), 503, "no_deployment"
    )


def test_the_app_follows_deployment_changes_by_polling_without_being_told(rig: Rig) -> None:
    first = rig.deployments.deploy(CLF, 1, "go live")
    runtime = rig.runtime()
    with rig.client(runtime, poll=1.0) as c:  # runs the start-up refresh and the poll loop
        assert c.get("/model").json()["deployment"]["version"] == 1
        rig.deployments.switch(first.id, CLF, 2, "polled")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if c.get("/health").json()["loaded_deployment_id"] != first.id:
                break
            time.sleep(0.25)
        assert c.get("/model").json()["deployment"]["version"] == 2


def test_a_restarted_app_serves_the_active_deployment_from_postgres_alone(rig: Rig) -> None:
    rig.deployments.deploy(CLF, 2, "go live")
    fresh = rig.runtime()
    with rig.client(fresh) as c:  # nobody told this instance anything
        assert c.get("/ready").status_code == 200
        assert c.get("/model").json()["deployment"]["version"] == 2


def test_a_model_that_does_not_verify_is_not_served_and_no_stale_model_stands_in(
    job_env: JobEnv, pool: list[Run], tmp_path: Path
) -> None:
    root = sealed_copy(pool[0], tmp_path / "copy")
    sealed_copy(pool[1], root)
    rig = Rig(job_env, pool, root)
    rig.register(pool[0], CLF)
    rig.register(pool[1], CLF)
    first = rig.deployments.deploy(CLF, 1, "go live")
    runtime = rig.runtime()
    runtime.refresh()
    c = rig.client(runtime)
    assert c.get("/ready").status_code == 200
    rig.deployments.switch(first.id, CLF, 2, "forest")  # verified fine at switch time
    unlock(root / pid(pool[1]) / "model.joblib").write_bytes(b"corrupted after deployment")
    health = c.post("/reload").json()
    assert health["ready"] is False and health["model_state"] == "load_failed"
    assert "model.joblib" in health["error"] or "failed verification" in health["error"]
    assert str(root) not in str(health)
    body = assert_envelope(
        c.post("/predict", json={"rows": rows_of(pool[0], 1)}), 503, "model_not_ready"
    )
    assert str(root) not in str(body)
    assert_envelope(c.get("/ready"), 503)
    assert_envelope(c.get("/model"), 503)
    assert c.get("/health").status_code == 200  # alive, and says why it is not ready
    # roll back to version 1: the app recovers on its own
    rig.deployments.rollback("v2 is broken")
    assert c.post("/reload").json()["ready"] is True
    assert c.get("/model").json()["deployment"]["version"] == 1


def test_a_failing_load_is_not_retried_at_every_poll_but_an_explicit_reload_retries(
    job_env: JobEnv, pool: list[Run], tmp_path: Path
) -> None:
    root = sealed_copy(pool[0], tmp_path / "copy")
    rig = Rig(job_env, pool, root)
    rig.register(pool[0], CLF)
    rig.deployments.deploy(CLF, 1, "go live")
    unlock(root / pid(pool[0]) / "model.joblib").write_bytes(b"bad")
    now = [100.0]
    runtime = ServingRuntime(job_env.engine, root, retry_seconds=30.0, clock=lambda: now[0])
    loads: list[int] = []
    original = runtime._load

    def counting(row: Any) -> Any:
        loads.append(row.id)
        return original(row)

    runtime._load = counting  # type: ignore[method-assign]
    assert runtime.refresh().model_state == "load_failed"
    runtime.refresh()
    runtime.refresh()
    assert len(loads) == 1  # inside the retry window
    now[0] += 31
    runtime.refresh()
    assert len(loads) == 2  # after it
    runtime.refresh(force=True)
    assert len(loads) == 3  # an explicit reload always tries


def test_the_loaded_model_must_match_the_registrys_record(rig: Rig) -> None:
    rig.deployments.deploy(CLF, 1, "go live")
    runtime = rig.runtime()
    active = rig.deployments.active()
    assert active is not None
    assert runtime._load(active).package.package_id == active.package_id
    for changed, text_ in (
        (dataclasses.replace(active, selection_sha256="0" * 64), "frozen selection"),
        (dataclasses.replace(active, task="regression"), "task"),
    ):
        with pytest.raises(LoadFailure, match=text_):
            runtime._load(changed)


# -- when the database is not what the app needs ----------------------------


def test_an_unreachable_database_at_start_means_not_ready_and_never_a_crash(rig: Rig) -> None:
    broken = create_engine(
        "postgresql+psycopg://nobody:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    runtime = ServingRuntime(broken, rig.root)
    health = runtime.refresh()
    assert (health.ready, health.database, health.model_state) == (False, "unavailable", "starting")
    c = rig.client(runtime)
    assert (
        c.get("/health").status_code == 200 and c.get("/health").json()["database"] == "unavailable"
    )
    assert_envelope(c.get("/ready"), 503)


def test_a_schema_that_is_not_current_is_not_served_from(rig: Rig) -> None:
    c, runtime = ready_client(rig)
    migrate.downgrade(rig.engine, "0004")
    try:
        health = runtime.refresh()
        assert health.model_state == "schema_not_current" and not health.ready
        assert "scripts/db.py upgrade" in (health.error or "")
        assert_envelope(c.get("/ready"), 503)
    finally:
        migrate.upgrade(rig.engine)


def test_a_database_outage_after_start_keeps_serving_the_verified_model(rig: Rig) -> None:
    c, runtime = ready_client(rig)
    good = runtime.engine
    runtime.engine = create_engine(
        "postgresql+psycopg://nobody:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    runtime.repository = DeploymentRepository(runtime.engine, rig.root)
    health = runtime.refresh()
    assert health.database == "unavailable" and health.ready and health.loaded_deployment_id
    assert c.post("/predict", json={"rows": rows_of(rig.pool[0], 2)}).status_code == 200
    runtime.engine = good
    runtime.repository = DeploymentRepository(good, rig.root)
    assert runtime.refresh().database == "ok"


# -- concurrency ---------------------------------------------------------------------------------


def test_predictions_during_a_switch_each_come_from_one_consistent_deployment(rig: Rig) -> None:
    c, _ = ready_client(rig)
    rows = rows_of(rig.pool[0], 4)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            r = c.post("/predict", json={"rows": rows})
            if r.status_code == 200:
                results.append(r.json())
            else:
                failures.append(r.text)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for target in (2, 1, 2):
            active = rig.deployments.active()
            assert active is not None
            rig.deployments.switch(active.id, CLF, target, f"flip to {target}")
            c.post("/reload")
            time.sleep(0.2)
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert failures == [] and len(results) > 8
    for body in results:
        version = body["deployment"]["version"]
        pool_index = version - 1
        labels, _ = expected(rig.root, rig.pool[pool_index], rows)
        assert [p["prediction"] for p in body["predictions"]] == [str(v) for v in labels]


def test_the_deep_package_check_is_the_registrys_not_a_second_one(rig: Rig) -> None:
    import inspect

    from automl_ops import deployments, package

    source = inspect.getsource(deployments)
    assert "verify_package" in source and "joblib" not in source and "pickle" not in source
    assert "loaded_parity_problems" in inspect.getsource(package)


def _engine_count(engine: Engine) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text("SELECT count(*) FROM deployments")).scalar_one())
