"""Phase 6, Step 33: the container support code (offline; no Docker needed)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from automl_ops.bind import host_problem

ROOT = Path(__file__).resolve().parents[1]


def test_loopback_is_always_allowed(tmp_path: Path) -> None:
    for host in ("127.0.0.1", "::1", "localhost"):
        assert host_problem(host, {}, tmp_path / "missing") is None


def test_all_interfaces_refused_outside_a_container(tmp_path: Path) -> None:
    marker = tmp_path / "dockerenv"
    marker.write_text("")
    # the variable alone is not enough, and neither is the marker alone
    assert host_problem("0.0.0.0", {"AUTOML_IN_CONTAINER": "1"}, tmp_path / "missing")
    assert host_problem("0.0.0.0", {}, marker)
    assert host_problem("0.0.0.0", {"AUTOML_IN_CONTAINER": "0"}, marker)


def test_all_interfaces_allowed_only_with_both_conditions(tmp_path: Path) -> None:
    marker = tmp_path / "dockerenv"
    marker.write_text("")
    assert host_problem("0.0.0.0", {"AUTOML_IN_CONTAINER": "1"}, marker) is None


@pytest.mark.parametrize("host", ["192.168.1.5", "example.com", "::", ""])
def test_other_addresses_always_refused(host: str, tmp_path: Path) -> None:
    marker = tmp_path / "dockerenv"
    marker.write_text("")
    assert host_problem(host, {"AUTOML_IN_CONTAINER": "1"}, marker)


def test_requirement_files_are_current() -> None:
    """The checked-in pinned requirement files match uv.lock (regenerate with
    `uv run python scripts/export_requirements.py`)."""
    spec = importlib.util.spec_from_file_location(
        "export_requirements", ROOT / "scripts" / "export_requirements.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_requirements"] = module
    spec.loader.exec_module(module)
    for name, text in module.render().items():
        assert (ROOT / "docker" / name).read_text(encoding="utf-8") == text, name


def test_serving_requirements_exclude_mlflow_and_pandera() -> None:
    names = {
        line.split("==")[0].lower()
        for line in (ROOT / "docker" / "requirements-serving.txt").read_text().splitlines()
        if line and line[0].isalpha()
    }
    assert not {"mlflow", "mlflow-skinny", "mlflow-tracing", "pandera"} & names
    assert {"fastapi", "scikit-learn", "psycopg"} <= names


def test_no_secret_or_data_path_is_copied_into_an_image() -> None:
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    for needed in (".env", ".git", "data", "project-notes", "*.joblib", ".venv"):
        assert needed in ignore, needed


# -- the full Compose stack's identity (Phase 6, Step 35) --------------------------------------


def test_compose_database_is_isolated_only_inside_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.engine import URL

    from automl_ops import bind, isolation

    url = URL.create("postgresql+psycopg", host="postgres", port=5432, database="automl_compose")
    env = {
        "AUTOML_JOBS_MLFLOW_DIR": "/data/mlflow",
        isolation.TOKEN_SETTING: "t" * 32,
        "AUTOML_IN_CONTAINER": "1",
    }
    wiring = isolation.Wiring(url, Path("/data/store"), Path("/data/models"), env=env)
    # on this machine there is no /.dockerenv marker: refused even with the variable set
    monkeypatch.setattr(bind, "CONTAINER_MARKER", tmp_path / "missing")
    refused = isolation.assess(wiring)
    assert not refused.isolated and "database_not_throwaway" in refused.reasons
    marker = tmp_path / "dockerenv"
    marker.write_text("")
    monkeypatch.setattr(bind, "CONTAINER_MARKER", marker)
    assert isolation.assess(wiring).isolated
    # and without the variable, even with the marker
    no_flag = isolation.Wiring(
        url, Path("/data/store"), Path("/data/models"), env={**env, "AUTOML_IN_CONTAINER": "0"}
    )
    assert not isolation.assess(no_flag).isolated


def test_protected_names_stay_protected_even_in_a_container() -> None:
    from sqlalchemy.engine import URL

    from automl_ops import isolation

    for name in ("automl_ops", "mlflow", "postgres"):
        url = URL.create("postgresql+psycopg", host="postgres", port=5432, database=name)
        report = isolation.assess(
            isolation.Wiring(
                url,
                Path("/data/store"),
                Path("/data/models"),
                env={"AUTOML_IN_CONTAINER": "1", isolation.TOKEN_SETTING: "t" * 32},
            )
        )
        assert not report.isolated


def test_compose_expectation_needs_a_token_and_matches_the_declared_database() -> None:
    from automl_ops import isolation

    assert not isolation.compose_expectation("short").isolated
    good = isolation.compose_expectation("t" * 32)
    assert good.isolated and good.run_id == isolation.run_id("t" * 32)
    other = isolation.compose_expectation("u" * 32)
    assert other.database_id == good.database_id and other.run_id != good.run_id
    # the development database (127.0.0.1:55432, automl_ops) has a different identifier
    from sqlalchemy.engine import URL

    dev = URL.create("postgresql+psycopg", host="127.0.0.1", port=55432, database="automl_ops")
    assert isolation.database_id(dev) != good.database_id


# -- the serving app stays reachable when the API container restarts (Phase 6 review fix) ------


def _in_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from automl_ops import bind

    marker = tmp_path / "dockerenv"
    marker.write_text("")
    monkeypatch.setattr(bind, "CONTAINER_MARKER", marker)
    monkeypatch.setenv("AUTOML_IN_CONTAINER", "1")


def test_serving_host_is_loopback_or_a_service_name_inside_the_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from automl_ops.serving_client import host_problem

    monkeypatch.delenv("AUTOML_IN_CONTAINER", raising=False)
    for host in ("127.0.0.1", "localhost", "::1"):
        assert host_problem(host) is None
    assert host_problem("serving")  # a service name is refused outside the image
    _in_container(monkeypatch, tmp_path)
    assert host_problem("serving") is None
    assert host_problem("my-serving-2") is None
    for bad in (
        "",
        "Serving",
        "a.b",
        "10.0.0.5",
        "-x",
        "x-",
        "serving:8200",
        "a b",
        "evil.example",
    ):
        assert host_problem(bad), bad


def test_settings_reject_a_bad_serving_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from automl_ops.api_control.settings import ApiSettings, SettingsError

    monkeypatch.delenv("AUTOML_IN_CONTAINER", raising=False)
    assert ApiSettings.from_environment({}).serving_host == "127.0.0.1"
    with pytest.raises(SettingsError):
        ApiSettings.from_environment({"AUTOML_SERVING_HOST": "serving"})
    with pytest.raises(SettingsError):
        ApiSettings.from_environment({"AUTOML_SERVING_HOST": "example.com"})


def test_the_client_connects_by_name_but_always_sends_the_loopback_host_header() -> None:
    """The serving app accepts only loopback Host values, so the name used to connect (the Compose
    service name in a container) must never reach the Host header."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from automl_ops.serving_client import IN_SYNC, ServingClient

    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            seen.append(self.headers.get("Host", ""))
            body = json.dumps(
                {"loaded_deployment_id": None, "model_state": "no_deployment"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        report = ServingClient(port, timeout=5, host="localhost").report(None, reload=False)
        assert report.state == IN_SYNC
        assert seen == [f"127.0.0.1:{port}"]  # not "localhost:<port>"
    finally:
        server.shutdown()
        server.server_close()


def test_the_client_refuses_an_address_a_caller_could_choose() -> None:
    from automl_ops.serving_client import ServingClient

    with pytest.raises(ValueError):
        ServingClient(8200, host="example.com")


def test_compose_gives_serving_its_own_network_not_the_apis() -> None:
    """Regression for the verified flaw: `network_mode: service:api` left serving unreachable after
    the API container restarted."""
    text = (ROOT / "compose.full.yaml").read_text(encoding="utf-8")
    assert "network_mode" not in text
    assert "AUTOML_SERVING_HOST: serving" in text
