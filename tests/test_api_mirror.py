# ruff: noqa: F811
"""P4-6: the mirror endpoints and the restart-safe background retry, on real PostgreSQL + MLflow."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from api_support import BASE, PORT, UI_ORIGIN, assert_envelope
from automl_ops.api_control.app import create_app
from automl_ops.api_control.schemas import (
    MirrorCheck,
    MirrorOverview,
    MirrorSummary,
    ModelVersionDetail,
    ReconcileResult,
    RegisteredModel,
    RollbackResult,
    SyncResult,
)
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.registry import RegistryRepository
from automl_ops.registry_mirror import Mirror
from automl_ops.tracking import MirrorError, TrackingStore, mirror_snapshot
from conftest import JobEnv, Run
from test_registry_postgres import pid, pool  # noqa: F401
from test_tracking_postgres import mlflow_template, pg_clone, pg_store, server  # noqa: F401

pytestmark = pytest.mark.postgres

NAME = "churn"


def build(
    env: JobEnv,
    pool: list[Run],
    store: TrackingStore | None,
    *,
    enabled: bool = True,
    wired: bool = True,
    factory: Any = None,
) -> tuple[TestClient, Mirror | None, RegistryRepository]:
    models = pool[0].root
    settings = ApiSettings(
        port=PORT, store_root=env.store_root, models_root=models, mirror_retry_seconds=1.0
    )
    catalogued = CatalogedDatasetStore(
        FileDatasetStore(env.store_root, settings.limits), env.engine
    )
    repo = RegistryRepository(env.engine, models)
    mirror = (
        Mirror(
            env.engine,
            models,
            enabled=lambda: enabled,
            store_factory=factory or (lambda: store),  # type: ignore[arg-type]
        )
        if wired
        else None
    )
    app = create_app(
        settings, catalogued, lambda: migrate.status(env.engine), registry=repo, mirror=mirror
    )
    return TestClient(app, base_url=BASE, raise_server_exceptions=False), mirror, repo


def register(c: TestClient, run: Run, **extra: Any) -> Any:
    return c.post("/models", json={"name": NAME, "package_id": pid(run), **extra})


def promote(c: TestClient, n: int, to: str) -> Any:
    return c.post(f"/models/{NAME}/versions/{n}/promote", json={"to": to, "reason": "ok"})


# -- the happy path -------------------------------------------------------------------------------


def test_registry_changes_are_mirrored_inline_and_reported(
    job_env: JobEnv, pool: list[Run], pg_store: TrackingStore
) -> None:
    c, mirror, _ = build(job_env, pool, pg_store)
    first = register(c, pool[0])
    assert first.status_code == 201, first.text
    v1 = ModelVersionDetail(**first.json())
    assert v1.mirror is not None and v1.mirror.state == "in_sync" and v1.mirror.pending_events == 0
    assert register(c, pool[1]).json()["mirror"]["state"] == "in_sync"
    assert promote(c, 1, "staging").json()["mirror"]["state"] == "in_sync"
    promoted = ModelVersionDetail(**promote(c, 1, "production").json())
    assert promoted.mirror is not None and promoted.mirror.synced_seq == promoted.mirror.desired_seq
    snapshot = mirror_snapshot(pg_store, NAME)
    assert snapshot.owned and snapshot.aliases == {
        "production": next(v.mlflow_version for v in snapshot.versions if v.registry_version == "1")
    }
    promote(c, 2, "staging")
    promote(c, 2, "production")
    rolled = c.post(f"/models/{NAME}/rollback", json={"reason": "back"})
    out = RollbackResult(**rolled.json())
    assert out.mirror is not None and out.mirror.state == "in_sync"
    assert (out.restored.version, out.archived.version) == (1, 2)
    assert mirror is not None and mirror.status(NAME).state == "in_sync"

    model = RegisteredModel(**c.get(f"/models/{NAME}").json())
    assert model.mirror is not None and model.mirror.state == "in_sync"
    assert ModelVersionDetail(**c.get(f"/models/{NAME}/versions/1").json()).mirror is not None
    summary = MirrorSummary(**c.get(f"/models/{NAME}/sync").json())
    assert summary.state == "in_sync" and summary.attempts == 0 and summary.last_error is None
    overview = MirrorOverview(**c.get("/registry/sync").json())
    assert overview.enabled and overview.problem is None
    assert [(m.model, m.state) for m in overview.models] == [(NAME, "in_sync")]

    synced = SyncResult(**c.post(f"/models/{NAME}/sync").json())
    assert synced.result == "up_to_date" and synced.mirror.state == "in_sync"
    checked = MirrorCheck(**c.get(f"/models/{NAME}/mirror/check").json())
    assert checked.in_sync and checked.model == NAME
    assert not [f for f in checked.findings if f.category != "conflict"]


def test_an_mlflow_failure_never_fails_or_undoes_a_registry_operation(
    job_env: JobEnv, pool: list[Run]
) -> None:
    def refuse() -> TrackingStore:
        raise MirrorError("MLflow is unreachable (simulated)")

    c, _, repo = build(job_env, pool, None, factory=refuse)
    made = register(c, pool[0])
    assert made.status_code == 201
    mirror = ModelVersionDetail(**made.json()).mirror
    assert mirror is not None and mirror.state == "failed" and mirror.pending_events == 1
    assert "unreachable" in (mirror.last_error or "")
    assert promote(c, 1, "staging").status_code == 200  # the registry keeps working
    assert repo.get_version(NAME, 1).status == "staging"
    assert c.get(f"/models/{NAME}/versions/1").json()["status"] == "staging"
    result = SyncResult(**c.post(f"/models/{NAME}/sync").json())
    assert result.result == "failed" and "unreachable" in (result.error or "")


# -- drift and reconcile through the API ----------------------------------------------------------


def test_drift_is_reported_and_only_an_explicit_reconcile_repairs_it(
    job_env: JobEnv, pool: list[Run], pg_store: TrackingStore
) -> None:
    c, mirror, _ = build(job_env, pool, pg_store)
    register(c, pool[0])
    register(c, pool[1])
    promote(c, 1, "staging")
    promote(c, 1, "production")
    snapshot = mirror_snapshot(pg_store, NAME)
    two = next(v.mlflow_version for v in snapshot.versions if v.registry_version == "2")
    MlflowClient(tracking_uri=pg_store.tracking_uri).set_registered_model_alias(
        NAME, "production", two
    )
    check = MirrorCheck(**c.get(f"/models/{NAME}/mirror/check").json())
    assert not check.in_sync
    assert [(f.kind, f.category) for f in check.findings if f.category == "drift"] == [
        ("alias_wrong", "drift")
    ]
    promote(c, 2, "staging")  # the registry moves on, so a sync is due
    blocked = SyncResult(**c.post(f"/models/{NAME}/sync").json())
    assert blocked.result == "drift" and blocked.mirror.state == "drifted"
    assert_envelope(c.post(f"/models/{NAME}/reconcile", json={}), 422, "validation_error")
    assert_envelope(
        c.post(f"/models/{NAME}/reconcile", json={"reason": ""}), 422, "validation_error"
    )
    assert_envelope(c.post(f"/models/{NAME}/reconcile", json={"reason": "x", "force": True}), 422)
    done = ReconcileResult(
        **c.post(f"/models/{NAME}/reconcile", json={"reason": "alias moved by hand"}).json()
    )
    assert done.outcome == "ok" and done.mirror.state == "in_sync"
    assert any(f.kind == "alias_wrong" for f in done.detected) and done.actions
    assert any("production alias" in line for line in done.intended)
    assert mirror is not None
    assert not [f for f in mirror.check(NAME).findings if f.stops_sync or f.category == "pending"]


def test_a_foreign_model_is_reported_and_the_reconcile_answers_conflict(
    job_env: JobEnv, pool: list[Run], pg_store: TrackingStore
) -> None:
    client = MlflowClient(tracking_uri=pg_store.tracking_uri)
    client.create_registered_model(NAME)  # someone else's model with our name
    c, _, _ = build(job_env, pool, pg_store)
    assert register(c, pool[0]).status_code == 201
    assert SyncResult(**c.post(f"/models/{NAME}/sync").json()).result == "conflict"
    done = ReconcileResult(
        **c.post(f"/models/{NAME}/reconcile", json={"reason": "take it over?"}).json()
    )
    assert done.outcome == "conflict" and done.actions == []
    assert not dict(client.get_registered_model(NAME).tags)  # untouched
    assert c.get(f"/models/{NAME}/sync").json()["state"] == "conflict"


# -- disabled, unwired and error handling ---------------------------------------------------------


def test_a_disabled_or_missing_mirror_changes_no_registry_behaviour(
    job_env: JobEnv, pool: list[Run]
) -> None:
    c, _, _ = build(job_env, pool, None, enabled=False)
    made = register(c, pool[0])
    assert made.status_code == 201 and made.json()["mirror"] is None  # no sync was attempted
    assert c.get(f"/models/{NAME}").json()["mirror"]["state"] == "disabled"
    assert c.get(f"/models/{NAME}/sync").json()["state"] == "disabled"
    overview = MirrorOverview(**c.get("/registry/sync").json())
    assert not overview.enabled and overview.problem
    assert_envelope(c.post(f"/models/{NAME}/sync"), 409, "mirror_disabled")
    assert_envelope(c.get(f"/models/{NAME}/mirror/check"), 409, "mirror_disabled")
    assert_envelope(
        c.post(f"/models/{NAME}/reconcile", json={"reason": "x"}), 409, "mirror_disabled"
    )
    bare, _, _ = build(job_env, pool, None, wired=False)
    assert bare.get(f"/models/{NAME}").json()["mirror"] is None
    assert_envelope(bare.get(f"/models/{NAME}/sync"), 409, "mirror_disabled")
    assert MirrorOverview(**bare.get("/registry/sync").json()).enabled is False


def test_unknown_models_methods_and_the_local_security_rules(
    job_env: JobEnv, pool: list[Run], pg_store: TrackingStore
) -> None:
    c, _, _ = build(job_env, pool, pg_store)
    assert_envelope(c.get("/models/nope/sync"), 404, "model_not_found")
    assert_envelope(c.post("/models/nope/sync"), 404, "model_not_found")
    assert_envelope(c.get("/models/nope/mirror/check"), 404, "model_not_found")
    assert_envelope(c.post("/models/nope/reconcile", json={"reason": "x"}), 404, "model_not_found")
    assert_envelope(c.get("/models/Bad_Name/sync"), 422, "validation_error")
    assert c.put("/models/churn/sync").headers["allow"] == "GET, POST"
    assert c.post("/models/churn/mirror/check").headers["allow"] == "GET"
    assert c.get("/models/churn/reconcile").headers["allow"] == "POST"
    assert c.post("/registry/sync").headers["allow"] == "GET"
    register(c, pool[0])
    evil = {"Origin": "http://evil.example"}
    assert_envelope(c.post(f"/models/{NAME}/sync", headers=evil), 403)
    assert_envelope(c.post(f"/models/{NAME}/reconcile", headers=evil, json={"reason": "x"}), 403)
    assert_envelope(c.get("/registry/sync", headers={"Host": "evil.example"}), 400)
    assert c.post(f"/models/{NAME}/sync", headers={"Origin": UI_ORIGIN}).status_code == 200


# -- restart-safe retry ---------------------------------------------------------------------------


def _wait_for(c: TestClient, state: str, timeout: float = 40.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if c.get(f"/models/{NAME}/sync").json()["state"] == state:
            return True
        time.sleep(0.25)
    return False


def test_the_background_retry_finds_pending_work_at_start_up_and_after_every_restart(
    job_env: JobEnv, pool: list[Run], pg_store: TrackingStore
) -> None:
    models = pool[0].root
    repo = RegistryRepository(job_env.engine, models)
    repo.register(NAME, package_id=pid(pool[0]), job_id=None)  # work left by an earlier API
    first, mirror, _ = build(job_env, pool, pg_store)
    assert mirror is not None and mirror.status(NAME).pending_events == 1
    with first:  # the lifespan starts the retry task, which inspects the persisted work
        assert _wait_for(first, "in_sync")
    assert mirror_snapshot(pg_store, NAME).owned

    repo.register(NAME, package_id=pid(pool[1]), job_id=None)  # the API is stopped meanwhile
    repo.promote(NAME, 2, "staging", "ready")
    assert mirror.status(NAME).pending_events == 2
    second, _, _ = build(job_env, pool, pg_store)  # a new process
    with second:
        assert _wait_for(second, "in_sync")
    assert len(mirror_snapshot(pg_store, NAME).versions) == 2


def test_the_background_retry_backs_off_and_recovers_when_mlflow_returns(
    job_env: JobEnv, pool: list[Run], pg_store: TrackingStore
) -> None:
    models = pool[0].root
    RegistryRepository(job_env.engine, models).register(NAME, package_id=pid(pool[0]), job_id=None)
    state = {"up": False}

    def flaky() -> TrackingStore:
        if not state["up"]:
            raise MirrorError("MLflow is unreachable (simulated)")
        return pg_store

    c, mirror, _ = build(job_env, pool, None, factory=flaky)
    with c:
        assert _wait_for(c, "failed", 30.0)  # the first round fails: MLflow is "down"
        assert c.get(f"/models/{NAME}/sync").json()["attempts"] >= 1
        assert c.get(f"/models/{NAME}").json()["versions"] == 1  # the registry is unaffected
        state["up"] = True
        with job_env.engine.begin() as conn:
            from sqlalchemy import text

            conn.execute(text("UPDATE registry_sync SET next_attempt_at = now()"))
        assert _wait_for(c, "in_sync")
