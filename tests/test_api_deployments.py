# ruff: noqa: F811
"""P4-7: the deployment endpoints of the control plane, with a real serving app on a real port.

Real PostgreSQL (throwaway databases), real sealed packages. NOT RUN when the server is
unreachable. The serving app runs in a thread on 127.0.0.1 at a free port, so the control plane's
notification is the real HTTP call.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from api_support import BASE, UI_ORIGIN, assert_envelope
from automl_ops.api_control.app import create_app
from automl_ops.api_control.schemas import DeploymentDetail, DeploymentList, DeploymentModel
from automl_ops.api_control.settings import ApiSettings
from automl_ops.api_serving.app import create_app as create_serving_app
from automl_ops.api_serving.runtime import ServingRuntime
from automl_ops.api_serving.settings import ServingSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.deployments import DeploymentRepository
from automl_ops.registry import RegistryRepository
from automl_ops.serving_client import ServingClient
from conftest import JobEnv, Run
from test_registry_postgres import pid, pool, reg, sealed_copy, unlock  # noqa: F401
from test_serving_postgres import rows_of

pytestmark = pytest.mark.postgres

NAME = "churn"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LiveServing:
    """The real serving app on a real port, in a thread."""

    def __init__(self, env: JobEnv, root: Path, port: int, poll: float = 60.0) -> None:
        self.port = port
        self.runtime = ServingRuntime(env.engine, root, retry_seconds=1.0)
        app = create_serving_app(
            ServingSettings(port=port, models_root=root, poll_seconds=poll), self.runtime
        )
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            assert time.monotonic() < deadline, "serving app did not start"
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15)

    def get(self, path: str) -> httpx.Response:
        return httpx.get(f"http://127.0.0.1:{self.port}{path}", timeout=10)

    def predict(self, rows: list[dict[str, Any]]) -> httpx.Response:
        return httpx.post(f"http://127.0.0.1:{self.port}/predict", json={"rows": rows}, timeout=10)


class Rig:
    def __init__(
        self, env: JobEnv, pool: list[Run], root: Path | None = None, count: int = 3
    ) -> None:
        self.env = env
        self.pool = pool
        self.root = root or pool[0].root
        self.registry = RegistryRepository(env.engine, self.root)
        self.deployments = DeploymentRepository(env.engine, self.root)
        reg(self.registry, pool[0], NAME)
        reg(self.registry, pool[1], NAME)
        if count > 2:
            reg(self.registry, pool[2], NAME)  # stays in development
        for n in (1, 2):
            self.registry.promote(NAME, n, "staging", "ready")

    def control(self, serving_port: int | None, *, wired: bool = True) -> TestClient:
        settings = ApiSettings(
            store_root=self.env.store_root,
            models_root=self.root,
            serving_port=serving_port or free_port(),
        )
        app = create_app(
            settings,
            CatalogedDatasetStore(
                FileDatasetStore(self.env.store_root, settings.limits), self.env.engine
            ),
            lambda: migrate.status(self.env.engine),
            registry=self.registry,
            deployments=self.deployments if wired else None,
            serving=ServingClient(settings.serving_port, timeout=10.0),
        )
        return TestClient(app, base_url=BASE, raise_server_exceptions=False)


@pytest.fixture
def rig(job_env: JobEnv, pool: list[Run]) -> Rig:
    return Rig(job_env, pool)


@pytest.fixture
def live(job_env: JobEnv, pool: list[Run]) -> Iterator[LiveServing]:
    serving = LiveServing(job_env, pool[0].root, free_port())
    serving.start()
    try:
        yield serving
    finally:
        serving.stop()


def body(model: str = NAME, version: int = 1, reason: str = "go live") -> dict[str, Any]:
    return {"model": model, "version": version, "reason": reason}


# -- the whole lifecycle through the API, with the real serving app ----------------------------


def test_deploy_switch_rollback_undeploy_drive_the_serving_app(rig: Rig, live: LiveServing) -> None:
    c = rig.control(live.port)
    rows = rows_of(rig.pool[0], 3)
    assert live.get("/health").json()["model_state"] == "no_deployment"

    created = c.post("/deployments", json=body())
    assert created.status_code == 201, created.text
    first = DeploymentDetail(**created.json())
    assert (first.version, first.status, first.action, first.registry_status) == (
        1,
        "active",
        "deploy",
        "staging",
    )
    assert first.serving is not None and first.serving.state == "in_sync"
    assert first.serving.loaded_deployment_id == first.id
    assert live.predict(rows).json()["deployment"]["version"] == 1

    second = DeploymentDetail(
        **c.post(f"/deployments/{first.id}/switch", json=body(version=2, reason="forest")).json()
    )
    assert (second.version, second.action, second.replaced_id) == (2, "switch", first.id)
    assert second.serving is not None and second.serving.state == "in_sync"
    assert live.get("/model").json()["deployment"]["version"] == 2
    assert live.predict(rows).json()["deployment"]["version"] == 2

    back = DeploymentDetail(**c.post("/deployments/rollback", json={"reason": "undo"}).json())
    assert (back.version, back.action, back.replaced_id) == (1, "rollback", second.id)
    assert live.get("/model").json()["deployment"]["version"] == 1

    ended = DeploymentDetail(
        **c.post(f"/deployments/{back.id}/undeploy", json={"reason": "maintenance"}).json()
    )
    assert ended.status == "ended" and ended.end_reason == "undeployed: maintenance"
    assert ended.serving is not None and ended.serving.state == "in_sync"
    assert live.get("/health").json()["model_state"] == "no_deployment"
    assert live.predict(rows).status_code == 503

    history = DeploymentList(**c.get("/deployments").json())
    assert [(d.version, d.action, d.status) for d in history.deployments] == [
        (1, "rollback", "ended"),
        (2, "switch", "ended"),
        (1, "deploy", "ended"),
    ]
    assert history.next_before_id is None


def test_active_reports_the_deployment_and_whether_the_serving_app_agrees(
    rig: Rig, live: LiveServing
) -> None:
    c = rig.control(live.port)
    assert_envelope(c.get("/deployments/active"), 404, "no_active_deployment")
    deployed = rig.deployments.deploy(NAME, 1, "behind the API's back")
    active = DeploymentDetail(**c.get("/deployments/active").json())
    assert active.id == deployed.id
    assert active.serving is not None and active.serving.state == "out_of_sync"  # not reloaded yet
    live.runtime.refresh()
    assert DeploymentDetail(**c.get("/deployments/active").json()).serving.state == "in_sync"  # type: ignore[union-attr]


def test_history_pages_and_single_reads(rig: Rig) -> None:
    c = rig.control(None)
    first = rig.deployments.deploy(NAME, 1, "a")
    second = rig.deployments.switch(first.id, NAME, 2, "b")
    rig.deployments.rollback("c")
    page = DeploymentList(**c.get("/deployments", params={"limit": 2}).json())
    assert len(page.deployments) == 2 and page.next_before_id == second.id
    rest = DeploymentList(
        **c.get("/deployments", params={"limit": 2, "before_id": page.next_before_id}).json()
    )
    assert [d.id for d in rest.deployments] == [first.id] and rest.next_before_id is None
    only_active = DeploymentList(**c.get("/deployments", params={"status": "active"}).json())
    assert len(only_active.deployments) == 1 and only_active.deployments[0].action == "rollback"
    assert DeploymentModel(**c.get(f"/deployments/{first.id}").json()).status == "ended"
    assert_envelope(c.get("/deployments/99999"), 404, "deployment_not_found")
    assert_envelope(c.get("/deployments/abc"), 422, "validation_error")
    assert_envelope(c.get("/deployments", params={"status": "bogus"}), 422, "validation_error")


# -- the serving app being away or unable ------------------------------------------------------


def test_a_stopped_serving_app_never_fails_a_deployment_and_catches_up_when_it_starts(
    rig: Rig, job_env: JobEnv
) -> None:
    port = free_port()
    c = rig.control(port)  # nothing listens there
    created = c.post("/deployments", json=body())
    assert created.status_code == 201
    detail = DeploymentDetail(**created.json())
    assert detail.serving is not None and detail.serving.state == "unreachable"
    assert detail.serving.error is not None and "did not answer" in detail.serving.error
    assert rig.deployments.active() is not None  # PostgreSQL has the change
    serving = LiveServing(job_env, rig.root, port)
    serving.start()
    try:
        assert serving.get("/ready").status_code == 200
        assert serving.get("/model").json()["deployment"]["version"] == 1
        assert DeploymentDetail(**c.get("/deployments/active").json()).serving.state == "in_sync"  # type: ignore[union-attr]
    finally:
        serving.stop()


def test_the_control_plane_refuses_a_package_that_no_longer_verifies_without_showing_paths(
    job_env: JobEnv, pool: list[Run], tmp_path: Path
) -> None:
    root = sealed_copy(pool[0], tmp_path / "copy")
    sealed_copy(pool[1], root)
    rig = Rig(job_env, pool, root, count=2)
    serving = LiveServing(job_env, root, free_port())
    serving.start()
    try:
        c = rig.control(serving.port)
        first = DeploymentDetail(**c.post("/deployments", json=body()).json())
        assert first.serving is not None and first.serving.state == "in_sync"
        unlock(root / pid(pool[1]) / "model.joblib").write_bytes(b"x")
        # the control plane's own deep check refuses the broken package: nothing changes
        refused = c.post(f"/deployments/{first.id}/switch", json=body(version=2))
        err = assert_envelope(refused, 409, "package_unverified")
        assert str(root) not in refused.text and err["details"]["state"] == "sealed"
        assert rig.deployments.active().id == first.id  # type: ignore[union-attr]
        assert serving.get("/ready").status_code == 200
    finally:
        serving.stop()


# -- errors ------------------------------------------------------------------------------------


def test_every_refusal_has_its_status_and_code(rig: Rig) -> None:
    c = rig.control(None)
    assert_envelope(c.post("/deployments", json=body(version=3)), 409, "version_not_deployable")
    assert_envelope(c.post("/deployments", json=body(version=99)), 404, "version_not_found")
    assert_envelope(c.post("/deployments", json=body(model="ghost")), 404, "model_not_found")
    first = DeploymentDetail(**c.post("/deployments", json=body()).json())
    again = assert_envelope(c.post("/deployments", json=body(version=2)), 409, "deployment_active")
    assert again["details"] == {"active_deployment_id": first.id}
    assert_envelope(
        c.post(f"/deployments/{first.id}/switch", json=body(version=1)), 409, "already_deployed"
    )
    assert_envelope(
        c.post("/deployments/12345/switch", json=body(version=2)), 409, "deployment_not_active"
    )
    assert_envelope(
        c.post("/deployments/rollback", json={"reason": "r"}), 409, "no_previous_deployment"
    )
    assert_envelope(
        c.post("/deployments/12345/undeploy", json={"reason": "r"}), 409, "deployment_not_active"
    )
    for payload in (
        body(reason=""),
        body(reason="x" * 501),
        {"model": NAME, "version": 0, "reason": "r"},
        {"model": "Bad Name!", "version": 1, "reason": "r"},
        {"model": NAME, "version": 1},
        {**body(), "extra": 1},
    ):
        assert_envelope(c.post("/deployments", json=payload), 422, "validation_error")
    assert_envelope(c.post("/deployments/rollback", json={}), 422, "validation_error")
    assert_envelope(c.get("/deployments/active/extra"), 404)
    assert_envelope(c.delete("/deployments/1"), 405)
    assert len(DeploymentList(**c.get("/deployments").json()).deployments) == 1


def test_a_disabled_deployment_repository_answers_503(rig: Rig) -> None:
    c = rig.control(None, wired=False)
    for response in (
        c.get("/deployments"),
        c.get("/deployments/1"),
        c.get("/deployments/active"),
        c.post("/deployments", json=body()),
        c.post("/deployments/rollback", json={"reason": "r"}),
    ):
        assert_envelope(response, 503, "deployments_unavailable")


def test_the_control_planes_local_request_rules_apply(rig: Rig) -> None:
    c = rig.control(None)
    assert_envelope(
        c.post("/deployments", json=body(), headers={"origin": "http://evil.example"}), 403
    )
    assert_envelope(c.post("/deployments", json=body(), headers={"host": "evil.example"}), 400)
    assert_envelope(
        c.post("/deployments", content=b"x", headers={"content-type": "text/plain"}), 415
    )
    ok = c.post("/deployments", json=body(), headers={"origin": UI_ORIGIN})
    assert ok.status_code == 201 and ok.headers["access-control-allow-origin"] == UI_ORIGIN


def test_registry_changes_do_not_undeploy_and_the_status_shows_the_difference(rig: Rig) -> None:
    c = rig.control(None)
    rig.registry.promote(NAME, 1, "production", "live")
    first = DeploymentDetail(**c.post("/deployments", json=body()).json())
    assert first.registry_status == "production"
    rig.registry.promote(NAME, 2, "production", "newer")  # supersedes (archives) version 1
    now = DeploymentModel(**c.get(f"/deployments/{first.id}").json())
    assert (now.status, now.registry_status) == ("active", "archived")
    # the archived version cannot be switched to again, but the running deployment is unaffected
    assert_envelope(
        c.post(f"/deployments/{first.id}/switch", json=body(version=1)),
        409,
        "version_not_deployable",
    )


def test_a_failed_change_leaves_the_history_untouched(rig: Rig) -> None:
    c = rig.control(None)
    first = DeploymentDetail(**c.post("/deployments", json=body()).json())
    before = c.get("/deployments").json()
    assert c.post(f"/deployments/{first.id}/switch", json=body(version=3)).status_code == 409
    assert c.get("/deployments").json() == before
    assert DeploymentModel(**c.get(f"/deployments/{first.id}").json()).status == "active"


def test_a_serving_app_that_cannot_load_the_deployment_is_reported_out_of_sync_without_paths(
    rig: Rig, job_env: JobEnv, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "wrong-models-folder"
    elsewhere.mkdir()
    serving = LiveServing(job_env, elsewhere, free_port())  # misconfigured: other models folder
    serving.start()
    try:
        c = rig.control(serving.port)
        created = c.post("/deployments", json=body())
        assert created.status_code == 201  # the change is committed in PostgreSQL regardless
        detail = DeploymentDetail(**created.json())
        assert detail.serving is not None and detail.serving.state == "out_of_sync"
        assert detail.serving.model_state == "load_failed" and detail.serving.error
        assert str(elsewhere) not in created.text and str(rig.root) not in created.text
        assert serving.get("/ready").status_code == 503
        assert serving.predict(rows_of(rig.pool[0], 1)).status_code == 503
    finally:
        serving.stop()
