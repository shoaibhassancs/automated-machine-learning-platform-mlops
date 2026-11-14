"""Shared helpers for the control-plane API tests (no PostgreSQL needed)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from automl_ops.api_control.app import create_app
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import catalog as catalog_module
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.db.migrate import DbStatus

PORT = 8100
BASE = f"http://127.0.0.1:{PORT}"
UI_ORIGIN = "http://localhost:5173"
HEAD = "0004"
CURRENT = DbStatus(current=HEAD, head=HEAD, pending=(), known=True)


class FakeEngine:
    """Stands in for the SQLAlchemy engine; `record_version` is patched, so this is never used
    for SQL. With `fail=True`, `begin()` raises, like an unreachable database."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    @contextlib.contextmanager
    def begin(self) -> Iterator[None]:
        if self.fail:
            raise RuntimeError("database down")
        yield None


def make_app(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: ApiSettings | None = None,
    fail_catalog: bool = False,
    db_status: Any = None,
) -> FastAPI:
    monkeypatch.setattr(
        catalog_module, "record_version", lambda connection, version: ("added", version.ref)
    )
    settings = settings or ApiSettings(port=PORT, store_root=root / "store")
    store = CatalogedDatasetStore(
        FileDatasetStore(settings.store_root, settings.limits),
        FakeEngine(fail=fail_catalog),  # type: ignore[arg-type]
    )
    return create_app(settings, store, db_status or (lambda: CURRENT))


def client(app: FastAPI) -> TestClient:
    return TestClient(app, base_url=BASE, raise_server_exceptions=False)


def upload(
    c: TestClient, name: str, data: bytes, *, add_version: bool = False, filename: str = "d.csv"
) -> Any:
    form = {"name": name, "add_version": "true" if add_version else "false"}
    return c.post("/datasets", data=form, files={"file": (filename, data, "text/csv")})


def assert_envelope(
    response: Any, status: int | None = None, code: str | None = None
) -> dict[str, Any]:
    """Every error response has exactly the agreed shape and says nothing internal."""
    if status is not None:
        assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/json"), response.text
    body = response.json()
    assert set(body) == {"error"}, body
    error = body["error"]
    assert set(error) <= {"code", "message", "details"}
    assert isinstance(error["code"], str) and isinstance(error["message"], str)
    if code is not None:
        assert error["code"] == code, body
    text = response.text
    for forbidden in ("Traceback", 'File "', "/Users/", "/private/", "/tmp/", "site-packages"):
        assert forbidden not in text, text
    return error  # type: ignore[no-any-return]


# -- raw ASGI driver: exact headers and chunked bodies, with a count of what was consumed -------


class Outcome:
    def __init__(self) -> None:
        self.status = 0
        self.headers: dict[str, str] = {}
        self.body = b""
        self.delivered = 0  # request-body bytes the app actually received
        self.receive_calls = 0

    def json(self) -> Any:
        return json.loads(self.body)


def asgi_call(
    app: FastAPI,
    method: str,
    path: str,
    headers: Iterable[tuple[str, str]],
    chunks: Iterable[bytes] = (),
) -> Outcome:
    outcome = Outcome()
    iterator = iter(chunks)
    pending: list[bytes] = []
    finished = False

    async def receive() -> dict[str, Any]:
        nonlocal finished
        outcome.receive_calls += 1
        if finished:
            return {"type": "http.disconnect"}
        try:
            chunk = pending.pop() if pending else next(iterator)
        except StopIteration:
            finished = True
            return {"type": "http.request", "body": b"", "more_body": False}
        outcome.delivered += len(chunk)
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            outcome.status = message["status"]
            outcome.headers = {
                k.decode().lower(): v.decode() for k, v in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            outcome.body += message.get("body", b"")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", PORT),
    }
    asyncio.run(app(scope, receive, send))  # type: ignore[arg-type]
    return outcome
