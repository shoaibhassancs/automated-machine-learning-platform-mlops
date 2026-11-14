"""P4-3: Host, Origin, CORS, content-type and body-size rules (D-036). No PostgreSQL needed."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI

from api_support import BASE, PORT, UI_ORIGIN, asgi_call, client, make_app
from automl_ops.api_control.settings import (
    JSON_BODY_LIMIT_BYTES,
    MULTIPART_OVERHEAD_BYTES,
    ApiSettings,
)
from automl_ops.ingest import IngestLimits
from conftest import toy_bytes

HOST = f"127.0.0.1:{PORT}"
JSON = ("content-type", "application/json")
MULTIPART_START = (
    b'--b\r\ncontent-disposition: form-data; name="name"\r\n\r\nbig\r\n'
    b'--b\r\ncontent-disposition: form-data; name="file"; filename="a.csv"\r\n'
    b"content-type: text/csv\r\n\r\n"
)
MULTIPART = ("content-type", "multipart/form-data; boundary=b")


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    return make_app(tmp_path, monkeypatch)


def call(app: FastAPI, method: str, path: str, *headers: tuple[str, str], body: bytes = b""):  # type: ignore[no-untyped-def]
    all_headers = [h for h in headers]
    if not any(k.lower() == "host" for k, _ in all_headers):
        all_headers.insert(0, ("host", HOST))
    if body:
        all_headers.append(("content-length", str(len(body))))
    return asgi_call(app, method, path, all_headers, [body] if body else [])


# -- Host ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}", f"LocalHost:{PORT}"]
)
def test_expected_hosts_are_accepted(app: FastAPI, host: str) -> None:
    assert call(app, "GET", "/health", ("host", host)).status == 200


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        f"evil.example:{PORT}",
        f"127.0.0.1.evil.example:{PORT}",
        f"localhost.evil.example:{PORT}",
        "127.0.0.1",  # no port
        "localhost:80",
        "127.0.0.1:8101",  # another port
        f"127.0.0.1:{PORT} ",
        f"0.0.0.0:{PORT}",
        f"[::ffff:127.0.0.1]:{PORT}",
        "",
    ],
)
def test_other_hosts_are_refused(app: FastAPI, host: str) -> None:
    out = call(app, "GET", "/health", ("host", host))
    assert out.status == 400
    assert out.json()["error"]["code"] == "host_not_allowed"


def test_missing_or_repeated_host_is_refused(app: FastAPI) -> None:
    assert asgi_call(app, "GET", "/health", []).status == 400
    assert asgi_call(app, "GET", "/health", [("host", HOST), ("host", HOST)]).status == 400


def test_host_is_checked_before_anything_else(app: FastAPI) -> None:
    out = call(
        app, "POST", "/datasets", ("host", "evil.example"), ("origin", "http://evil.example")
    )
    assert out.json()["error"]["code"] == "host_not_allowed"
    assert "access-control-allow-origin" not in out.headers


def test_host_check_follows_the_configured_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = make_app(tmp_path, monkeypatch, settings=ApiSettings(port=8222, store_root=tmp_path))
    assert call(other, "GET", "/health", ("host", "127.0.0.1:8222")).status == 200
    assert call(other, "GET", "/health", ("host", HOST)).status == 400


# -- Origin on state-changing requests ----------------------------------------------------------

UPLOAD_HEADERS = ("content-type", "multipart/form-data; boundary=x")


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "null",
        "",
        "*",
        "http://localhost",  # right host, wrong port
        "http://localhost:5174",
        "https://localhost:5173",  # right host and port, wrong scheme
        "http://127.0.0.1:5173",  # not the configured host name
        "http://localhost:5173/",  # not an origin (trailing slash)
        "http://localhost:5173/path",
        "http://localhost:5173@evil.example",
        "localhost:5173",
        "http://",
        "http://localhost:5173:80",
        f"http://localhost:{PORT}",  # the API's own origin is not the UI
    ],
)
def test_state_changing_requests_with_a_bad_origin_are_refused(app: FastAPI, origin: str) -> None:
    for path, content_type in (
        ("/datasets", UPLOAD_HEADERS),
        ("/datasets/a/versions/1/task-check", JSON),
    ):
        out = call(app, "POST", path, content_type, ("origin", origin), body=b"{}")
        assert out.status == 403, (origin, out.body)
        assert out.json()["error"]["code"] == "origin_not_allowed"
        assert "access-control-allow-origin" not in out.headers
    assert call(app, "DELETE", "/datasets", ("origin", origin)).status == 403


def test_repeated_origin_headers_are_refused(app: FastAPI) -> None:
    out = call(
        app,
        "POST",
        "/datasets/a/versions/1/task-check",
        JSON,
        ("origin", UI_ORIGIN),
        ("origin", UI_ORIGIN),
        body=b"{}",
    )
    assert out.status == 403


def test_the_configured_origin_and_no_origin_are_accepted(app: FastAPI) -> None:
    body = b'{"target": "y"}'
    path = "/datasets/a/versions/1/task-check"
    with_origin = call(app, "POST", path, JSON, ("origin", UI_ORIGIN), body=body)
    without = call(app, "POST", path, JSON, body=body)
    assert with_origin.status == without.status == 404  # passed the checks; the version is absent
    assert with_origin.json()["error"]["code"] == "version_not_found"


def test_origin_is_compared_as_a_parsed_origin(app: FastAPI) -> None:
    """Scheme and host case do not matter (browsers send lower case); the port does."""
    body = b'{"target": "y"}'
    path = "/datasets/a/versions/1/task-check"
    assert (
        call(app, "POST", path, JSON, ("origin", "HTTP://LOCALHOST:5173"), body=body).status == 404
    )
    assert (
        call(app, "POST", path, JSON, ("origin", "http://localhost:5173"), body=body).status == 404
    )


# -- CORS ---------------------------------------------------------------------------------------


def test_cors_headers_only_for_the_configured_origin(app: FastAPI) -> None:
    ok = call(app, "GET", "/health", ("origin", UI_ORIGIN))
    assert ok.headers["access-control-allow-origin"] == UI_ORIGIN
    assert ok.headers["vary"] == "Origin"
    assert "access-control-allow-credentials" not in ok.headers
    for other in ("http://evil.example", "null", "http://localhost:5174", "*"):
        out = call(app, "GET", "/health", ("origin", other))
        assert out.status == 200  # reads still work; the browser just may not show them
        assert "access-control-allow-origin" not in out.headers
    assert "access-control-allow-origin" not in call(app, "GET", "/health").headers


def test_error_responses_carry_cors_headers_for_the_ui(app: FastAPI) -> None:
    out = call(app, "GET", "/datasets/ghost", ("origin", UI_ORIGIN))
    assert out.status == 404 and out.headers["access-control-allow-origin"] == UI_ORIGIN


def test_preflight_for_the_configured_origin(app: FastAPI) -> None:
    out = call(
        app,
        "OPTIONS",
        "/datasets",
        ("origin", UI_ORIGIN),
        ("access-control-request-method", "POST"),
        ("access-control-request-headers", "Content-Type"),
    )
    assert out.status == 204
    assert out.headers["access-control-allow-origin"] == UI_ORIGIN
    assert out.headers["access-control-allow-methods"] == "GET, POST"
    assert out.headers["access-control-allow-headers"] == "content-type"
    assert "access-control-allow-credentials" not in out.headers


@pytest.mark.parametrize(
    ("origin", "method", "headers"),
    [
        ("http://evil.example", "POST", "content-type"),
        ("null", "POST", ""),
        (UI_ORIGIN, "DELETE", ""),
        (UI_ORIGIN, "PUT", ""),
        (UI_ORIGIN, "POST", "authorization"),
        (UI_ORIGIN, "POST", "content-type, x-custom"),
    ],
)
def test_other_preflights_are_refused_in_the_envelope(
    app: FastAPI, origin: str, method: str, headers: str
) -> None:
    request = [("origin", origin), ("access-control-request-method", method)]
    if headers:
        request.append(("access-control-request-headers", headers))
    out = call(app, "OPTIONS", "/datasets", *request)
    assert out.status == 403 and out.json()["error"]["code"] == "cors_not_allowed"
    assert "access-control-allow-origin" not in out.headers


def test_plain_options_is_just_405(app: FastAPI) -> None:
    out = call(app, "OPTIONS", "/datasets")
    assert out.status == 405 and out.json()["error"]["code"] == "method_not_allowed"


# -- content type -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content_type",
    [
        "text/plain",
        "application/x-www-form-urlencoded",
        "multipart/form-data; boundary=x",
        "",
        "json",
    ],
)
def test_json_endpoints_need_json(app: FastAPI, content_type: str) -> None:
    headers = [("content-type", content_type)] if content_type else []
    out = call(app, "POST", "/datasets/a/versions/1/task-check", *headers, body=b'{"target":"y"}')
    assert out.status == 415 and out.json()["error"]["code"] == "unsupported_media_type"


@pytest.mark.parametrize(
    "content_type", ["application/json", "text/plain", "application/x-www-form-urlencoded", ""]
)
def test_the_upload_needs_multipart(app: FastAPI, content_type: str) -> None:
    headers = [("content-type", content_type)] if content_type else []
    out = call(app, "POST", "/datasets", *headers, body=b"a,b\n1,2\n")
    assert out.status == 415


def test_json_with_charset_is_accepted(app: FastAPI) -> None:
    out = call(
        app,
        "POST",
        "/datasets/a/versions/1/task-check",
        ("content-type", "application/json; charset=utf-8"),
        body=b'{"target":"y"}',
    )
    assert out.status == 404


# -- body size, enforced while streaming --------------------------------------------------------


def test_declared_oversize_is_refused_without_reading_the_body(app: FastAPI) -> None:
    cap = ApiSettings(port=PORT).upload_body_limit
    out = asgi_call(
        app,
        "POST",
        "/datasets",
        [("host", HOST), UPLOAD_HEADERS, ("content-length", str(cap + 1))],
        iter([b"x" * 1024] * 3),
    )
    assert out.status == 413 and out.json()["error"]["code"] == "body_too_large"
    assert out.delivered == 0 and out.receive_calls == 0


def test_chunked_oversize_is_cut_off_at_the_limit(app: FastAPI) -> None:
    cap = ApiSettings(port=PORT).upload_body_limit
    chunk = 256 * 1024

    def endless() -> Iterator[bytes]:
        yield MULTIPART_START
        while True:
            yield b"a,b\n" * (chunk // 4)

    out = asgi_call(
        app,
        "POST",
        "/datasets",
        [("host", HOST), MULTIPART, ("transfer-encoding", "chunked")],
        endless(),
    )
    assert out.status == 413 and out.json()["error"]["code"] == "body_too_large"
    assert cap < out.delivered <= cap + 2 * chunk  # stopped within a chunk of the limit


def test_an_understated_content_length_does_not_get_around_the_limit(app: FastAPI) -> None:
    chunk = 256 * 1024
    limit = JSON_BODY_LIMIT_BYTES
    out = asgi_call(
        app,
        "POST",
        "/datasets/a/versions/1/task-check",
        [("host", HOST), JSON, ("content-length", "10")],
        iter([b" " * chunk] * 100),
    )
    assert out.status == 413
    assert out.delivered <= limit + 2 * chunk


def test_json_endpoints_have_a_small_limit(app: FastAPI) -> None:
    out = call(
        app,
        "POST",
        "/datasets/a/versions/1/task-check",
        JSON,
        body=b" " * (JSON_BODY_LIMIT_BYTES + 1),
    )
    assert out.status == 413


def test_bad_content_length_header(app: FastAPI) -> None:
    out = asgi_call(
        app,
        "POST",
        "/datasets/a/versions/1/task-check",
        [("host", HOST), JSON, ("content-length", "-5")],
        [],
    )
    assert out.status == 400 and out.json()["error"]["code"] == "bad_request"


def test_limit_is_file_limit_plus_a_separate_allowance() -> None:
    settings = ApiSettings(port=PORT)
    assert settings.limits == IngestLimits()  # the 50 MB / 100,000 rows / 100 columns defaults
    assert settings.limits.max_bytes == 50 * 1024 * 1024
    assert settings.upload_body_limit == settings.limits.max_bytes + MULTIPART_OVERHEAD_BYTES


def test_an_oversized_stream_is_not_spooled_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a 1 MiB file limit, a 40 MiB chunked upload is cut off near the limit (the server
    stops reading) and no temporary file larger than the limit is left behind. The test client
    cannot show this (it hands over the whole body at once); the raw ASGI driver feeds chunks."""
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    settings = ApiSettings(
        port=PORT, store_root=tmp_path / "s", limits=IngestLimits(max_bytes=1 << 20)
    )
    app = make_app(tmp_path, monkeypatch, settings=settings)
    chunk = 256 * 1024

    def body() -> Iterator[bytes]:
        yield MULTIPART_START
        for _ in range(160):  # 160 x 256 KiB = 40 MiB, if it were all consumed
            yield b"a,b\n" * (chunk // 4)
        yield b"\r\n--b--\r\n"

    out = asgi_call(
        app,
        "POST",
        "/datasets",
        [("host", HOST), MULTIPART, ("transfer-encoding", "chunked")],
        body(),
    )
    assert out.status == 413 and out.json()["error"]["code"] == "body_too_large"
    assert out.delivered <= settings.upload_body_limit + 2 * chunk  # not 40 MiB
    leftovers = [p for p in spool.rglob("*") if p.is_file()]
    assert all(p.stat().st_size <= settings.upload_body_limit + 2 * chunk for p in leftovers)
    assert client(app).get("/datasets").json() == {"datasets": []}


def test_a_valid_upload_still_works_through_the_middleware(app: FastAPI) -> None:
    c = client(app)
    response = c.post(
        "/datasets",
        data={"name": "ok"},
        files={"file": ("a.csv", toy_bytes(), "text/csv")},
        headers={"origin": UI_ORIGIN},
    )
    assert response.status_code == 201
    assert response.headers["access-control-allow-origin"] == UI_ORIGIN
    assert BASE.startswith("http://127.0.0.1")
