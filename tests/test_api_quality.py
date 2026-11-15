# ruff: noqa: F811
"""P4-10 (Step 26): API quality pass. Structure checks on both exported OpenAPI files need no
database; the runtime sweeps and the reliability tests use real PostgreSQL (throwaway databases) and
both apps in process, and are NOT RUN (skipped) when the server is unreachable."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api_support import assert_envelope
from automl_ops import simulator as sim
from test_monitoring_postgres import Env
from test_registry_postgres import pool  # noqa: F401
from test_serving_postgres import CLF, Rig, ready_client, rig  # noqa: F401
from test_simulator import AppHttp, World, world  # noqa: F401

DOCS = Path(__file__).resolve().parents[1] / "docs"
SPECS = {
    name: json.loads((DOCS / f"openapi-{name}.json").read_text()) for name in ("control", "serving")
}

# Lists that are NOT paged, with the reason. A new unpaged list fails the structure test until it is
# added here on purpose (the gap is listed, not hidden). docs/api.md repeats this list.
UNPAGED = {
    "/algorithms": "a fixed catalogue of eight algorithms",
    "/datasets": "reads the file store; one dataset has few versions (single user)",
    "/models": "registered model names (single user; one row per model, no per-version rows)",
    "/models/{name}/versions": "one model's versions (a handful)",
}
OFFSET_PAGED = {"/jobs/{job_id}": "after_seq", "/models/{name}/events": "after_seq"}
# Predictions have a UUID id (no numeric key), so this one list pages by a string cursor.
CURSOR_PAGED = {"/deployments/{deployment_id}/predictions": ("before", "next_before")}


def resolve(spec: dict[str, Any], node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        node = spec["components"]["schemas"][node["$ref"].rsplit("/", 1)[1]]
    return node


def operations(spec: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [(m, p, o) for p, item in spec["paths"].items() for m, o in item.items()]


@pytest.mark.parametrize("name", ["control", "serving"])
def test_every_operation_is_documented_the_same_way(name: str) -> None:
    spec = SPECS[name]
    ids = [o["operationId"] for _, _, o in operations(spec)]
    assert len(ids) == len(set(ids))
    for method, path, op in operations(spec):
        assert op.get("summary") and op.get("tags") is not None or name == "serving", (method, path)
        assert op["summary"].strip() and not op["summary"].endswith("."), (method, path)
        errors = {c: r for c, r in op["responses"].items() if not c.startswith("2")}
        assert errors, (method, path)
        shapes = {
            resolve(spec, r["content"]["application/json"]["schema"]).get("title")
            for r in errors.values()
            if c_has_json(r)
        }
        # one error shape everywhere; only /health may also answer 503 with its own health body
        assert shapes - {"ErrorEnvelope", "Health", "HealthResponse"} == set(), (method, path)
        assert "ErrorEnvelope" in shapes or path == "/health", (method, path, shapes)
    assert any(o["responses"].get("422") for _, _, o in operations(spec))


def c_has_json(response: dict[str, Any]) -> bool:
    return "application/json" in response.get("content", {})


def test_paged_lists_share_one_pagination_contract() -> None:
    spec = SPECS["control"]
    paged: set[str] = set()
    unpaged: set[str] = set()
    for method, path, op in operations(spec):
        if method != "get":
            continue
        params = {p["name"]: p for p in op.get("parameters", []) if p["in"] == "query"}
        schema = op["responses"]["200"]["content"]["application/json"]["schema"]
        if "limit" in params:
            paged.add(path)
            cursor, next_cursor = CURSOR_PAGED.get(path, ("before_id", "next_before_id"))
            assert cursor in params, path
            assert (
                params["limit"]["schema"]["maximum"] <= 100
                and params["limit"]["schema"]["default"] <= params["limit"]["schema"]["maximum"]
            ), path
            assert next_cursor in resolve(spec, schema)["properties"], path
        elif str(schema.get("$ref", "")).endswith("List"):  # DatasetList, ModelVersionList, ...
            unpaged.add(path)
    assert unpaged - set(OFFSET_PAGED) == set(UNPAGED)
    assert {"/alerts", "/jobs", "/deployments", "/deployments/{deployment_id}/reports"} <= paged
    assert {"/deployments/{deployment_id}/feedback", *CURSOR_PAGED} <= paged


def test_serving_spec_has_the_small_documented_surface() -> None:
    paths = {(m.upper(), p) for m, p, _ in operations(SPECS["serving"])}
    assert paths == {
        ("POST", "/predict"),
        ("POST", "/feedback"),
        ("GET", "/model"),
        ("GET", "/health"),
        ("GET", "/ready"),
        ("POST", "/reload"),
    }


# -- runtime sweep -----------------------------------------------------------------------------


def fill(path: str, value: str) -> str:
    return re.sub(r"\{[^}]+\}", value, path)


def sweep(client: TestClient, spec: dict[str, Any]) -> int:
    """Every documented operation, with unknown, malformed and empty input: never a 5xx other than
    the documented 503, and every non-2xx answer is a well-formed error envelope."""
    count = 0
    for method, path, _ in operations(spec):
        for value in ("1", "nope", "-1", "9" * 30):
            bodies: list[Any] = [None, {}, {"x": 1}, []]
            for body in bodies:
                if method == "get" and body is not None:
                    continue
                response = client.request(
                    method.upper(), fill(path, value), json=body if body is not None else None
                )
                count += 1
                assert response.status_code != 500, (method, path, value, body, response.text)
                if response.status_code >= 400:
                    assert_envelope(response)
    return count


@pytest.mark.postgres
def test_every_control_plane_route_answers_bad_input_with_an_envelope(rig: Rig) -> None:
    ready_client(rig)
    env = Env(rig)
    assert sweep(env.client, SPECS["control"]) > 300
    with rig.engine.connect() as c:  # the sweep changed nothing it should not have
        assert c.execute(text("SELECT count(*) FROM predictions")).scalar_one() == 0


@pytest.mark.postgres
def test_every_serving_route_answers_bad_input_with_an_envelope(rig: Rig) -> None:
    client, _ = ready_client(rig)
    assert sweep(client, SPECS["serving"]) > 30
    with rig.engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM predictions")).scalar_one() == 0


# -- reliability around the simulator ----------------------------------------------------------


class Switcher:
    """Passes everything through and switches the deployment after `after` predict calls."""

    def __init__(self, world: World, after: int) -> None:
        self.world, self.after, self.seen = world, after, 0
        self.inner = world.serving

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> sim.Reply:
        reply = self.inner.request(method, path, body)
        if path == "/predict":
            self.seen += 1
            if self.seen == self.after:
                active = self.world.rig.deployments.active()
                assert active is not None
                self.world.rig.deployments.switch(active.id, CLF, 2, "mid-run switch")
                assert self.world.runtime.refresh(force=True).ready
        return reply


@pytest.mark.postgres
def test_a_switch_during_a_run_is_reported_and_never_mislabels(world: World) -> None:
    first = world.rig.deployments.active()
    assert first is not None
    result = sim.run(
        world.normal,
        Switcher(world, after=2),
        world.control,
        rows=60,
        batch_size=10,
        seed=4,
        feedback_fraction=1.0,
    )
    assert result.rows_sent == 60 and not result.aborted
    # rows after the switch were predicted by the new deployment, so the old-deployment labels for
    # them are refused (409), counted, and the run is reported as not clean
    assert result.feedback_created == 20 and result.feedback_failed == 40 and not result.clean
    with world.rig.engine.connect() as c:
        wrong = c.execute(
            text(
                "SELECT count(*) FROM feedback f JOIN predictions p ON p.id = f.prediction_id "
                "WHERE f.deployment_id <> p.deployment_id"
            )
        ).scalar_one()
        by_deployment: dict[int, int] = {
            int(d): int(n)
            for d, n in c.execute(
                text("SELECT deployment_id, count(*) FROM predictions GROUP BY 1")
            )
        }
    assert wrong == 0 and sorted(by_deployment.values()) == [20, 40]


@pytest.mark.postgres
def test_state_survives_restarting_both_apps(world: World, tmp_path: Path) -> None:
    result = sim.run(world.shifted, world.serving, world.control, rows=100, feedback_fraction=0.5)
    assert result.clean
    deployment = result.deployment_id
    assert world.env.reference().status_code in (200, 201)
    report = world.control.request(
        "POST", f"/deployments/{deployment}/monitor", {"scope": "simulated"}
    )
    assert report.body is not None and report.body["status"] == "ok"

    def picture(control: AppHttp) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, path in (
            ("status", f"/deployments/{deployment}/monitoring"),
            ("alerts", "/alerts"),
            ("perf", f"/deployments/{deployment}/performance?scope=simulated&min_labelled=5"),
            ("reports", f"/deployments/{deployment}/reports"),
        ):
            reply = control.request("GET", path)
            assert reply.status == 200 and reply.body is not None
            out[key] = reply.body
        return out

    before = picture(world.control)
    assert before["alerts"]["alerts"], "the shifted traffic should have opened alerts"
    fresh = World(
        world.rig, tmp_path / "again", attach=True
    )  # new runtime and new apps over the same database
    assert picture(fresh.control) == before
    again = sim.run(fresh.normal, fresh.serving, fresh.control, rows=20, seed=9)
    assert again.clean and again.deployment_id == deployment
