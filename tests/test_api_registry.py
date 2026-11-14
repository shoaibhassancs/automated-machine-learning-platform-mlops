"""P4-5: the registry endpoints on a real PostgreSQL server (throwaway database, scratch stores)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from api_support import BASE, PORT, UI_ORIGIN, assert_envelope
from automl_ops.api_control.app import create_app
from automl_ops.api_control.schemas import (
    ModelVersionDetail,
    ModelVersionList,
    RegisteredModel,
    RegisteredModelList,
    RegistryEventList,
    RollbackResult,
)
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.registry import RegistryRepository
from conftest import JobEnv, Run
from test_registry_postgres import FAKE_PACKAGE, pid, pool  # noqa: F401
from test_worker_postgres import enqueue, make_spec, make_worker

pytestmark = pytest.mark.postgres


def build(env: JobEnv, models: Any, **over: Any) -> TestClient:
    settings = ApiSettings(port=PORT, store_root=env.store_root, models_root=models)
    store = CatalogedDatasetStore(FileDatasetStore(env.store_root, settings.limits), env.engine)
    kwargs: dict[str, Any] = {"registry": RegistryRepository(env.engine, models)}
    kwargs.update(over)
    app = create_app(settings, store, lambda: migrate.status(env.engine), **kwargs)
    return TestClient(app, base_url=BASE, raise_server_exceptions=False)


@pytest.fixture
def api(job_env: JobEnv, pool: list[Run]) -> TestClient:  # noqa: F811
    return build(job_env, pool[0].root)


def register(c: TestClient, run: Run, name: str = "churn", **extra: Any) -> Any:
    return c.post("/models", json={"name": name, "package_id": pid(run), **extra})


def promote(c: TestClient, n: int, to: str, name: str = "churn", reason: str = "ok") -> Any:
    return c.post(f"/models/{name}/versions/{n}/promote", json={"to": to, "reason": reason})


# -- the happy path -------------------------------------------------------------------------------


def test_register_read_promote_and_roll_back(api: TestClient, pool: list[Run]) -> None:  # noqa: F811
    first = register(api, pool[0], reason="first candidate")
    assert first.status_code == 201, first.text
    v1 = ModelVersionDetail(**first.json())
    assert (v1.model, v1.version, v1.status, v1.package_id) == (
        "churn",
        1,
        "development",
        pid(pool[0]),
    )
    assert v1.dataset == "toy@v1" and v1.job_id is None
    assert v1.evaluation["evaluation"]["metrics"] and v1.evaluation["limitation"]
    assert v1.evaluation["omitted"] == ["roc_curves", "pr_curves"]  # named, never cut silently
    assert register(api, pool[1]).json()["version"] == 2

    assert RegisteredModelList(**api.get("/models").json()).models[0].versions == 2
    one = RegisteredModel(**api.get("/models/churn").json())
    assert (one.production_version, one.latest_version) == (None, 2)
    listing = ModelVersionList(**api.get("/models/churn/versions").json())
    assert [v.version for v in listing.versions] == [2, 1]
    assert "evaluation" not in api.get("/models/churn/versions").json()["versions"][0]
    assert ModelVersionDetail(**api.get("/models/churn/versions/1").json()).evaluation

    assert promote(api, 1, "staging").json()["status"] == "staging"
    assert promote(api, 1, "production").json()["status"] == "production"
    assert promote(api, 2, "staging").status_code == 200
    assert promote(api, 2, "production").status_code == 200
    states = {
        v.version: v.status
        for v in ModelVersionList(**api.get("/models/churn/versions").json()).versions
    }
    assert states == {1: "archived", 2: "production"}
    rolled = api.post("/models/churn/rollback", json={"reason": "v2 misbehaves"})
    assert rolled.status_code == 200, rolled.text
    outcome = RollbackResult(**rolled.json())
    assert (outcome.restored.version, outcome.restored.status) == (1, "production")
    assert (outcome.archived.version, outcome.archived.status) == (2, "archived")
    assert RegisteredModel(**api.get("/models/churn").json()).production_version == 1

    events = RegistryEventList(**api.get("/models/churn/events").json())
    assert [e.seq for e in events.events] == list(range(1, len(events.events) + 1))
    assert events.events[-1].action == "rollback" and events.events[-1].to_status == "production"
    later = RegistryEventList(
        **api.get(f"/models/churn/events?after_seq={events.last_seq - 1}").json()
    )
    assert [e.seq for e in later.events] == [events.last_seq]
    assert (
        RegistryEventList(
            **api.get(f"/models/churn/events?after_seq={events.last_seq}").json()
        ).events
        == []
    )

    assert register(api, pool[2]).json()["version"] == 3
    archived = api.post("/models/churn/versions/3/archive", json={"reason": "retire"})
    assert archived.status_code == 200 and archived.json()["status"] == "archived"
    assert_envelope(
        api.post("/models/churn/versions/2/archive", json={"reason": "again"}),
        409,
        "invalid_transition",
    )


def test_registering_through_a_job(job_env: JobEnv) -> None:
    job_id = enqueue(job_env, make_spec("final"))
    make_worker(job_env).run(once=True)
    c = build(job_env, job_env.models)
    done = register_job = c.post("/models", json={"name": "from-job", "job_id": job_id})
    assert done.status_code == 201, done.text
    assert register_job.json()["job_id"] == job_id and register_job.json()["dataset"] == "toy@v1"
    package = register_job.json()["package_id"]
    assert_envelope(
        c.post("/models", json={"name": "again", "job_id": job_id, "package_id": package}),
        409,
        "package_already_registered",
    )
    other = FAKE_PACKAGE
    assert_envelope(
        c.post("/models", json={"name": "x", "job_id": job_id, "package_id": other}),
        422,
        "provenance_mismatch",
    )
    assert_envelope(c.post("/models", json={"name": "x", "job_id": 9999}), 404, "job_not_found")
    train = enqueue(job_env, make_spec("train"))
    assert_envelope(
        c.post("/models", json={"name": "x", "job_id": train}), 422, "job_not_registerable"
    )
    make_worker(job_env).run(once=True)
    assert_envelope(
        c.post("/models", json={"name": "x", "job_id": train}), 422, "job_not_registerable"
    )


# -- rejections -----------------------------------------------------------------------------------


def test_registration_rules_are_reported_with_stable_codes(
    api: TestClient,
    pool: list[Run],  # noqa: F811
) -> None:
    assert_envelope(api.post("/models", json={"name": "churn"}), 422, "package_required")
    assert_envelope(
        api.post("/models", json={"name": "churn", "package_id": FAKE_PACKAGE}),
        422,
        "package_not_registerable",
    )
    bad: dict[str, Any]
    for bad in (
        {"name": "Churn", "package_id": pid(pool[0])},
        {"name": "", "package_id": pid(pool[0])},
    ):
        assert_envelope(api.post("/models", json=bad), 422, "validation_error")
    for bad in (
        {"name": "churn", "package_id": "../../etc/passwd"},
        {"name": "churn", "package_id": pid(pool[0]), "extra": 1},
        {"name": "churn", "package_id": pid(pool[0]), "reason": ""},
        {"name": "churn", "package_id": pid(pool[0]), "reason": "x" * 501},
        {"name": "churn", "job_id": 0},
    ):
        assert_envelope(api.post("/models", json=bad), 422, "validation_error")
    assert api.get("/models").json() == {"models": []}
    assert register(api, pool[0]).status_code == 201
    assert_envelope(register(api, pool[0]), 409, "package_already_registered")
    assert_envelope(register(api, pool[0], "elsewhere"), 409, "package_already_registered")
    assert api.get("/models").json()["models"][0]["name"] == "churn"
    assert len(api.get("/models").json()["models"]) == 1


def test_move_rules_and_missing_things(api: TestClient, pool: list[Run]) -> None:  # noqa: F811
    register(api, pool[0])
    assert_envelope(promote(api, 1, "production"), 409, "invalid_transition")
    assert_envelope(promote(api, 7, "staging"), 404, "version_not_found")
    assert_envelope(promote(api, 1, "staging", "nope"), 404, "model_not_found")
    assert_envelope(promote(api, 1, "archived"), 422, "validation_error")
    assert_envelope(promote(api, 1, "staging", reason=""), 422, "validation_error")
    assert_envelope(
        api.post("/models/churn/versions/1/promote", json={"to": "staging"}),
        422,
        "validation_error",
    )
    assert_envelope(
        api.post("/models/churn/versions/0/promote", json={"to": "staging", "reason": "x"}), 422
    )
    promote(api, 1, "staging")
    promote(api, 1, "production")
    assert_envelope(
        api.post("/models/churn/versions/1/archive", json={"reason": "x"}),
        409,
        "invalid_transition",
    )
    assert_envelope(
        api.post("/models/churn/rollback", json={"reason": "x"}), 409, "no_previous_production"
    )
    assert_envelope(api.post("/models/nope/rollback", json={"reason": "x"}), 404, "model_not_found")
    assert_envelope(api.get("/models/nope"), 404, "model_not_found")
    assert_envelope(api.get("/models/nope/versions"), 404, "model_not_found")
    assert_envelope(api.get("/models/churn/versions/9"), 404, "version_not_found")
    assert_envelope(api.get("/models/nope/events"), 404, "model_not_found")
    assert_envelope(api.get("/models/Bad_Name"), 422, "validation_error")
    assert_envelope(api.get("/models/churn/events?after_seq=-1"), 422, "validation_error")
    assert api.get("/models/churn").json()["production_version"] == 1  # nothing above changed it


def test_a_package_that_stops_verifying_is_refused_at_promotion(
    job_env: JobEnv,
    pool: list[Run],  # noqa: F811
    tmp_path: Any,
) -> None:
    from test_registry_postgres import sealed_copy, unlock

    root = sealed_copy(pool[0], tmp_path / "w")
    c = build(job_env, root)
    assert register(c, pool[0]).status_code == 201
    model = unlock(root / pid(pool[0]) / "model.joblib")
    model.write_bytes(model.read_bytes() + b"x")
    refused = assert_envelope(promote(c, 1, "staging"), 409, "package_unverified")
    assert refused["details"]["problems"]
    assert c.get("/models/churn/versions/1").json()["status"] == "development"


def test_the_error_envelope_holds_for_malformed_bodies_and_methods(
    api: TestClient,
    pool: list[Run],  # noqa: F811
) -> None:
    assert_envelope(
        api.post("/models", content=b"{no", headers={"content-type": "application/json"}), 422
    )
    assert_envelope(api.post("/models", data={"name": "x"}), 415, "unsupported_media_type")
    assert_envelope(
        api.post(
            "/models",
            content=b"x" * (1024 * 1024 + 1),
            headers={"content-type": "application/json"},
        ),
        413,
    )
    assert_envelope(api.delete("/models"), 405, "method_not_allowed")
    assert api.delete("/models").headers["allow"] == "GET, POST"
    assert api.put("/models/churn").headers["allow"] == "GET"
    assert api.get("/models/churn/rollback").headers["allow"] == "POST"
    assert api.get("/models/churn/versions/1/promote").headers["allow"] == "POST"
    assert api.post("/models/churn/events").headers["allow"] == "GET"


def test_the_local_security_rules_apply_to_the_registry_too(
    api: TestClient,
    pool: list[Run],  # noqa: F811
) -> None:
    body = {"json": {"name": "churn", "package_id": pid(pool[0])}}
    assert_envelope(api.post("/models", headers={"Origin": "http://evil.example"}, **body), 403)
    assert_envelope(api.post("/models", headers={"Origin": "null"}, **body), 403)
    assert_envelope(
        api.post(
            "/models/churn/rollback",
            headers={"Origin": "http://evil.example"},
            json={"reason": "x"},
        ),
        403,
    )
    assert_envelope(api.get("/models", headers={"Host": "evil.example"}), 400)
    assert api.get("/models").json() == {"models": []}  # nothing was registered by those attempts
    assert api.post("/models", headers={"Origin": UI_ORIGIN}, **body).status_code == 201


def test_the_registry_answers_503_when_it_is_not_wired(job_env: JobEnv) -> None:
    offline = build(job_env, job_env.models, registry=None)
    assert_envelope(offline.get("/models"), 503, "registry_unavailable")
    assert_envelope(
        offline.post("/models", json={"name": "x", "job_id": 1}), 503, "registry_unavailable"
    )
    assert offline.get("/health").json()["status"] == "ok"
