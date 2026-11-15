# ruff: noqa: F811
"""P4-10 (Step 25): the traffic simulator. The pure rules need no database; the runs use real
sealed packages, real PostgreSQL (throwaway databases) and both apps in process. NOT RUN (skipped)
when the server is unreachable. Nothing here touches `data/`."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from automl_ops import simulator as sim
from automl_ops.simulator import Reply, SimulatorError
from test_monitoring_postgres import Env
from test_registry_postgres import pool  # noqa: F401
from test_serving_postgres import CLF, Rig, ready_client, rig  # noqa: F401

# -- a stand-in for the demo directory ---------------------------------------------------------


def stream_frame(n: int, seed: int, shifted: bool = False) -> pd.DataFrame:
    """The toy data's own recipe (conftest.toy_bytes) with another seed: same distribution as the
    training data, so unshifted traffic should look like the reference."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).round(4)
    z = rng.normal(size=n).round(4)
    color = rng.choice(["red", "green", "blue"], n)
    frame = pd.DataFrame({"x": x, "z": z, "color": color})
    frame.loc[::17, "z"] = np.nan
    frame["y"] = np.where(x + 0.5 * z.round(4) > 0, "pos", "neg")
    if shifted:
        frame["x"] = (frame["x"] + 3.0).round(4)
        frame["color"] = "red"
    return frame


def demo_dir(root: Path, n: int = 120, name: str = "toy") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind, shifted in (("stream", False), ("shifted", True)):
        data = stream_frame(n, 5, shifted).to_csv(index=False).encode()
        (root / f"{name}_monitor_{kind}.csv").write_bytes(data)
        files["monitoring_" + kind] = {"sha256": hashlib.sha256(data).hexdigest()}
    (root / "manifest.json").write_text(json.dumps({"datasets": {name: {"files": files}}}))
    return root


class AppHttp:
    """The simulator's transport, pointed at an in-process app."""

    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Reply:
        self.calls.append((method, path))
        r = self.client.request(method, path, json=body)
        value = r.json() if r.content else None
        return Reply(r.status_code, value if isinstance(value, dict) else None)


class Flaky:
    """Passes everything through except `/predict`, which fails after `ok` good calls."""

    def __init__(self, inner: AppHttp, ok: int = 0, status: int = 0) -> None:
        self.inner, self.ok, self.status, self.seen = inner, ok, status, 0

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Reply:
        if path == "/predict":
            self.seen += 1
            if self.seen > self.ok:
                return Reply(self.status, None)
        return self.inner.request(method, path, body)


# -- the file rules (no database) --------------------------------------------------------------


def test_only_the_known_demo_files_load(tmp_path: Path) -> None:
    root = demo_dir(tmp_path / "demo")
    normal = sim.load_stream(root / "toy_monitor_stream.csv")
    shifted = sim.load_stream(root / "toy_monitor_shifted.csv")
    assert (normal.kind, shifted.kind, normal.dataset) == ("normal", "shifted", "toy")
    assert len(normal.frame) == 120 and all(isinstance(v, str) for v in normal.frame["x"])
    (root / "other.csv").write_bytes(b"a\n1\n")
    for bad in ("other.csv", "missing_monitor_stream.csv"):
        with pytest.raises(SimulatorError) as caught:
            sim.load_stream(root / bad)
        assert caught.value.code == "unknown_file"


def test_a_changed_file_or_missing_manifest_is_refused(tmp_path: Path) -> None:
    root = demo_dir(tmp_path / "demo")
    with (root / "toy_monitor_stream.csv").open("ab") as handle:
        handle.write(b"1,1,red,pos\n")
    with pytest.raises(SimulatorError, match="hash"):
        sim.load_stream(root / "toy_monitor_stream.csv")
    (root / "manifest.json").unlink()
    with pytest.raises(SimulatorError, match="manifest"):
        sim.load_stream(root / "toy_monitor_shifted.csv")
    # a file named like a stream but absent from the manifest of another dataset
    other = demo_dir(tmp_path / "demo2", name="other")
    (other / "toy_monitor_stream.csv").write_bytes(b"x\n1\n")
    with pytest.raises(SimulatorError):
        sim.load_stream(other / "toy_monitor_stream.csv")


def test_row_choice_is_a_function_of_the_seed_only() -> None:
    a = sim.select_rows(500, 100, 0.3, seed=7)
    assert a == sim.select_rows(500, 100, 0.3, seed=7)
    assert a != sim.select_rows(500, 100, 0.3, seed=8)
    order, labelled = a
    assert len(order) == len(set(order)) == 100 and len(labelled) == 30 and labelled <= set(order)
    assert sim.select_rows(500, 100, 0.0, seed=7)[1] == set()
    for bad in ((500, 0, 0.1), (500, 501, 0.1), (500, 10, 1.5), (500, 10, -0.1)):
        with pytest.raises(SimulatorError):
            sim.select_rows(*bad, seed=1)


def test_the_transport_only_reaches_loopback_and_reports_no_answer() -> None:
    for port in (0, 70000, True):
        with pytest.raises(SimulatorError):
            sim.LocalHttp(port)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert sim.LocalHttp(free, timeout=1).request("GET", "/health") == Reply(0, None)
    assert sim.HOST == "127.0.0.1"


# -- runs against the real apps ----------------------------------------------------------------


class World:
    def __init__(self, rig: Rig, tmp: Path, rows: int = 120, *, attach: bool = False) -> None:
        self.rig = rig
        if attach:  # new runtime and app over the deployment that is already active (a restart)
            self.runtime = rig.runtime()
            assert self.runtime.refresh().ready
            self.serving_client = rig.client(self.runtime)
        else:
            self.serving_client, self.runtime = ready_client(rig)
        self.env = Env(rig)
        self.serving, self.control = AppHttp(self.serving_client), AppHttp(self.env.client)
        self.root = demo_dir(tmp / "demo", rows)
        self.normal = sim.load_stream(self.root / "toy_monitor_stream.csv")
        self.shifted = sim.load_stream(self.root / "toy_monitor_shifted.csv")

    def rows(self, where: str = "TRUE") -> list[Any]:
        with self.rig.engine.connect() as c:
            return list(c.execute(text(f"SELECT * FROM predictions WHERE {where} ORDER BY id")))


@pytest.fixture
def world(rig: Rig, tmp_path: Path) -> World:
    return World(rig, tmp_path)


@pytest.mark.postgres
def test_a_run_logs_only_simulated_predictions_and_labels(world: World) -> None:
    result = sim.run(
        world.normal, world.serving, world.control, rows=60, seed=3, feedback_fraction=0.5
    )
    assert result.clean and result.rows_sent == result.predictions == 60
    assert (result.feedback_planned, result.feedback_created) == (30, 30)
    assert result.model == CLF and result.deployment_id is not None
    rows = world.rows()
    assert len(rows) == 60 and all(r.is_simulated for r in rows)
    # requests were all simulated and the simulator never wrote anywhere but the two APIs
    assert {m for m, _ in world.serving.calls} <= {"GET", "POST"}
    deployment = result.deployment_id
    real = world.control.request("GET", f"/deployments/{deployment}/performance")
    sim_ = world.control.request(
        "GET", f"/deployments/{deployment}/performance?scope=simulated&min_labelled=5"
    )
    assert real.body is not None and sim_.body is not None
    assert real.body["scope"] == "real" and real.body["counts"]["labelled"] == 0
    assert sim_.body["scope"] == "simulated" and sim_.body["counts"]["labelled"] == 30


@pytest.mark.postgres
def test_the_same_seed_sends_the_same_traffic(world: World) -> None:
    def snapshot() -> list[str]:
        with world.rig.engine.connect() as c:
            found = c.execute(
                text(
                    "SELECT p.inputs::text, p.output->>'prediction', f.true_value "
                    "FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id"
                )
            )
            return sorted(repr(tuple(r)) for r in found)

    sim.run(world.normal, world.serving, world.control, rows=40, seed=11, feedback_fraction=0.25)
    first = snapshot()
    with world.rig.engine.begin() as c:  # a clean slate in this throwaway database only
        c.execute(text("ALTER TABLE feedback DISABLE TRIGGER USER"))
        c.execute(text("ALTER TABLE predictions DISABLE TRIGGER USER"))
        c.execute(text("TRUNCATE feedback, predictions"))
    sim.run(world.normal, world.serving, world.control, rows=40, seed=11, feedback_fraction=0.25)
    assert first == snapshot() and len(first) == 40
    assert sum("None" not in r.split(", ")[-1] for r in first) == 10  # the labelled rows


@pytest.mark.postgres
def test_training_data_is_refused_before_anything_is_sent(world: World) -> None:
    data = world.normal.path.read_bytes()
    up = world.env.client.post("/datasets", data={"name": "copy"}, files={"file": ("c.csv", data)})
    assert up.status_code in (200, 201)
    with pytest.raises(SimulatorError) as caught:
        sim.run(world.normal, world.serving, world.control, rows=10)
    assert caught.value.code == "training_data"
    assert world.rows() == [] and ("POST", "/predict") not in world.serving.calls
    # the shifted copy is a different file and is still allowed
    assert sim.run(world.shifted, world.serving, world.control, rows=10).clean


@pytest.mark.postgres
def test_control_plane_down_means_no_run(world: World) -> None:
    down = Flaky(world.control)
    down.request = lambda *a, **k: Reply(0, None)  # type: ignore[method-assign]
    with pytest.raises(SimulatorError) as caught:
        sim.run(world.normal, world.serving, down, rows=10)
    assert caught.value.code == "control_unavailable" and world.rows() == []


@pytest.mark.postgres
def test_a_dead_or_failing_serving_app_stops_the_run_honestly(world: World) -> None:
    dead = sim.run(
        world.normal, Flaky(world.serving), world.control, rows=100, batch_size=10, seed=1
    )
    assert dead.aborted and not dead.clean and dead.rows_sent == 0
    assert dead.batches_failed == sim.MAX_CONSECUTIVE_FAILURES and "failed batches" in str(
        dead.reason
    )
    partial = sim.run(
        world.normal,
        Flaky(world.serving, ok=2, status=503),
        world.control,
        rows=100,
        batch_size=10,
        seed=1,
        feedback_fraction=1.0,
    )
    assert partial.aborted and partial.rows_sent == 20 and partial.predictions == 20
    assert len(world.rows()) == 20 and partial.feedback_created == 20
    nothing = sim.run(
        world.normal, AppHttp(TestClient(__import__("fastapi").FastAPI())), world.control, rows=5
    )
    assert nothing.aborted and nothing.rows_sent == 0 and "no model" in str(nothing.reason)


@pytest.mark.postgres
def test_a_stream_that_does_not_fit_the_model_is_refused(world: World, tmp_path: Path) -> None:
    root = tmp_path / "wrong"
    root.mkdir()
    data = stream_frame(40, 1).drop(columns=["x"]).to_csv(index=False).encode()
    (root / "toy_monitor_stream.csv").write_bytes(data)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "toy": {
                        "files": {"monitoring_stream": {"sha256": hashlib.sha256(data).hexdigest()}}
                    }
                }
            }
        )
    )
    stream = sim.load_stream(root / "toy_monitor_stream.csv")
    with pytest.raises(SimulatorError) as caught:
        sim.run(stream, world.serving, world.control, rows=10)
    assert caught.value.code == "model_mismatch" and world.rows() == []


@pytest.mark.postgres
def test_the_rate_paces_batches(world: World) -> None:
    waits: list[float] = []
    sim.run(
        world.normal, world.serving, world.control, rows=30, batch_size=10, rate=20,
        sleep=waits.append,
    )  # fmt: skip
    assert waits == [0.5, 0.5]
    for bad in (0, -1, float("nan")):
        with pytest.raises(SimulatorError):
            sim.run(world.normal, world.serving, world.control, rows=5, rate=bad)


# -- the drift demonstration -------------------------------------------------------------------


@pytest.mark.postgres
def test_normal_shifted_and_recovery_with_the_default_thresholds(rig: Rig, tmp_path: Path) -> None:
    """The full alert story on the default thresholds (nothing tuned): normal traffic raises no
    alert, the shifted copy raises them, normal traffic again clears the drift alerts. Real traffic
    stays apart."""
    world = World(rig, tmp_path, rows=1100)
    real_rows = stream_frame(60, 9)[["x", "z", "color"]]
    real = [{k: (None if pd.isna(v) else str(v)) for k, v in r.items()}
            for r in real_rows.to_dict("records")]  # fmt: skip
    assert world.serving_client.post("/predict", json={"rows": real}).status_code == 200

    phases = sim.demo(
        (world.normal, world.shifted),
        world.serving,
        world.control,
        rows=1000,
        seed=2,
        feedback_fraction=0.1,
    )
    names = [p.name for p in phases]
    assert names == ["normal", "shifted", "recovery"]
    assert all(p.traffic.clean and p.traffic.predictions == 1000 for p in phases)
    normal, shifted, recovery = phases
    assert normal.monitoring["status"] == "ok" and normal.monitoring["window_rows"] == 1000
    assert set(normal.monitoring["drifted_features"]) <= {"z"} and normal.alerts == []
    assert {"x", "color"} <= set(shifted.monitoring["drifted_features"])  # z may flag by chance
    assert {a["rule"] for a in shifted.alerts if a["status"] != "resolved"} >= {
        "drifted_feature_share"
    }
    # All three alerts clear. `performance_drop` is scored over the labels inside the same
    # 1,000-row window as the drift verdict, so it recovers when traffic does (it stayed open
    # before, when it was computed over all labelled history).
    by_rule = {a["rule"]: a for a in recovery.alerts}
    assert by_rule["drifted_feature_share"]["status"] == "resolved"
    assert by_rule["prediction_shift"]["status"] == "resolved"
    assert by_rule["performance_drop"]["status"] == "resolved"
    assert by_rule["performance_drop"]["occurrences"] == 1
    assert recovery.monitoring["performance"]["triggered"] is False
    # (the 1,100-row stream differs a little from the training sample in `z` by chance: one feature
    # of three flags, below the share threshold of one half, so no alert)
    assert set(recovery.monitoring["drifted_features"]) <= {"z"}

    deployment = normal.traffic.deployment_id
    events = rig_events(rig)
    assert "opened" in events and "auto_resolved" in events
    # real traffic: untouched by the simulated shift (60 real rows, no alert on the real scope)
    real_report = world.control.request(
        "POST", f"/deployments/{deployment}/monitor", {"scope": "real"}
    )
    assert real_report.body is not None and real_report.body["scope"] == "real"
    assert real_report.body["window_rows"] == 60
    assert real_report.body["status"] == "ok"
    with rig.engine.connect() as c:
        scopes = {r[0] for r in c.execute(text("SELECT DISTINCT scope FROM alerts"))}
    assert scopes == {"simulated"}
    assert len(world.rows("NOT is_simulated")) == 60


def rig_events(rig: Rig) -> set[str]:
    with rig.engine.connect() as c:
        return {r[0] for r in c.execute(text("SELECT action FROM alert_events"))}
