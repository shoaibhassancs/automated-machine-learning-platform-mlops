"""P4-7: the serving app stays a separate, slim application; the control plane's view of it.

No database needed. The import checks run in a fresh interpreter so they see only what the serving
app itself pulls in.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from automl_ops.serving_client import IN_SYNC, OUT_OF_SYNC, UNREACHABLE, ServingClient


def imported_by(statement: str) -> set[str]:
    code = f"import sys; {statement}; print('\\n'.join(sorted(sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return set(out.split())


def test_the_serving_app_does_not_load_mlflow_or_the_control_plane_app() -> None:
    modules = imported_by("import automl_ops.api_serving.app, automl_ops.api_serving.runtime")
    assert "mlflow" not in modules and "automl_ops.tracking" not in modules
    assert "automl_ops.api_control.app" not in modules
    assert "automl_ops.registry_mirror" not in modules and "automl_ops.worker" not in modules
    for monitoring in ("monitor", "monitoring", "reference"):  # P4-9 stays out of serving
        assert f"automl_ops.{monitoring}" not in modules
    assert "automl_ops.api_serving.runtime" in modules


def test_the_control_plane_does_not_load_the_serving_runtime_or_a_model() -> None:
    modules = imported_by("import automl_ops.api_control.app")
    assert "automl_ops.api_serving.runtime" not in modules
    assert "automl_ops.api_serving.app" not in modules


class Fake(BaseHTTPRequestHandler):
    reply: dict[str, Any] = {}
    seen: list[tuple[str, str]] = []

    def _answer(self) -> None:
        type(self).seen.append((self.command, self.path))
        data = json.dumps(type(self).reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = _answer  # noqa: N815
    do_POST = _answer  # noqa: N815

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def fake() -> Iterator[tuple[int, type[Fake]]]:
    Fake.seen = []
    server = HTTPServer(("127.0.0.1", 0), Fake)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], Fake
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_the_client_compares_what_the_serving_app_serves_with_what_is_active(
    fake: tuple[int, type[Fake]],
) -> None:
    port, handler = fake
    client = ServingClient(port, timeout=5)
    handler.reply = {"model_state": "ready", "loaded_deployment_id": 7, "error": None}
    assert client.report(7, reload=False).state == IN_SYNC
    assert client.report(8, reload=False).state == OUT_OF_SYNC
    handler.reply = {"model_state": "no_deployment", "loaded_deployment_id": None, "error": "x"}
    assert client.report(None, reload=False).state == IN_SYNC
    assert client.report(7, reload=False).state == OUT_OF_SYNC
    handler.reply = {"model_state": "load_failed", "loaded_deployment_id": None, "error": "bad"}
    report = client.report(7, reload=True)
    assert (report.state, report.model_state, report.error) == (OUT_OF_SYNC, "load_failed", "bad")
    assert ("POST", "/reload") in handler.seen and ("GET", "/health") in handler.seen


def test_the_client_reports_an_unreachable_app_and_a_garbled_answer_instead_of_raising() -> None:
    closed = ServingClient(1, timeout=1)
    assert closed.report(1, reload=True).state == UNREACHABLE
    assert closed.report(None, reload=False).error is not None


def test_an_answer_without_the_expected_fields_is_out_of_sync_not_a_crash(
    fake: tuple[int, type[Fake]],
) -> None:
    port, handler = fake
    handler.reply = {}
    report = ServingClient(port).report(1, reload=False)
    assert report.state == OUT_OF_SYNC and report.loaded_deployment_id is None
