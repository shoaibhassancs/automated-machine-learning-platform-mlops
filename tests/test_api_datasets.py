"""P4-3: the dataset endpoints, error handling, contract and CLI parity (no PostgreSQL needed;
the catalog write is patched here and tested against the real server in test_api_postgres.py)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api_support import (
    BASE,
    CURRENT,
    PORT,
    assert_envelope,
    client,
    make_app,
    upload,
)
from automl_ops import dataset_reports
from automl_ops.api_control import datasets as routes
from automl_ops.api_control.app import StartupError, require_current_schema
from automl_ops.api_control.openapi import OPENAPI_PATH, render_openapi
from automl_ops.api_control.settings import ApiSettings, SettingsError, origin_of
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db.migrate import DbStatus
from automl_ops.ingest import IngestLimits
from automl_ops.profile import profile_version
from conftest import toy_bytes

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
VERSION = "/datasets/toy/versions/1"


@pytest.fixture
def c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return client(make_app(tmp_path, monkeypatch))


@pytest.fixture
def stored(c: TestClient) -> TestClient:
    assert upload(c, "toy", toy_bytes()).status_code == 201
    return c


# -- happy paths --------------------------------------------------------------------------------


def test_upload_creates_then_is_idempotent(c: TestClient) -> None:
    first = upload(c, "toy", toy_bytes(), filename="my file.csv")
    assert first.status_code == 201
    body = first.json()
    assert body["created"] is True and body["catalogued"] is True
    version = body["version"]
    assert version["ref"] == "toy@v1" and version["rows"] == 150
    assert version["columns"] == ["x", "z", "color", "y"]
    assert version["source_filename"] == "my file.csv"
    assert body["upload"]["delimiter"] == ","
    again = upload(c, "toy", toy_bytes())
    assert again.status_code == 200
    assert again.json()["created"] is False
    assert again.json()["version"]["sha256"] == version["sha256"]


def test_upload_converts_other_formats_and_says_so(c: TestClient) -> None:
    data = "a;b\n1;2\n3;4\n".encode("utf-16")
    response = upload(c, "semi", data)
    assert response.status_code == 201
    info = response.json()["upload"]
    assert info["encoding"] == "utf-16" and info["delimiter"] == ";" and info["reencoded"] is True


def test_new_version_needs_the_explicit_flag(stored: TestClient) -> None:
    other = toy_bytes("regression")
    assert_envelope(upload(stored, "toy", other), 409, "dataset_exists")
    second = upload(stored, "toy", other, add_version=True)
    assert second.status_code == 201 and second.json()["version"]["ref"] == "toy@v2"
    assert_envelope(upload(stored, "nope", other, add_version=True), 404, "dataset_not_found")


def test_list_and_get(stored: TestClient) -> None:
    listing = stored.get("/datasets").json()
    assert [d["name"] for d in listing["datasets"]] == ["toy"]
    assert listing["datasets"][0]["versions"][0]["ref"] == "toy@v1"
    assert stored.get("/datasets/toy").json()["versions"][0]["number"] == 1
    assert stored.get(VERSION).json()["sha256"] == listing["datasets"][0]["versions"][0]["sha256"]


def test_empty_store_lists_nothing(c: TestClient) -> None:
    assert c.get("/datasets").json() == {"datasets": []}


def test_reports_happy_path(stored: TestClient) -> None:
    profile = stored.get(f"{VERSION}/profile").json()
    assert profile["dataset_ref"] == "toy@v1" and profile["rows"] == 150
    assert [p["name"] for p in profile["column_profiles"]] == ["x", "z", "color", "y"]
    validation = stored.get(f"{VERSION}/validation", params={"target": "y"}).json()
    assert validation["target"] == "y" and "findings" in validation
    preview = stored.get(f"{VERSION}/preview", params={"target": "y"}).json()
    assert preview["class_balance"]["total_classes"] == 2
    assert len(preview["sample_rows"]) == 10
    plain = stored.get(f"{VERSION}/preview").json()
    assert plain["class_balance"] is None
    task = stored.post(f"{VERSION}/task-check", json={"target": "y", "excluded": ["color"]}).json()
    assert task["ok"] is True
    assert task["config"] == {"target": "y", "task": "classification", "excluded": ["color"]}
    assert task["feature_columns"] == ["x", "z"]


def test_task_check_stores_nothing(stored: TestClient, tmp_path: Path) -> None:
    before = sorted(p.name for p in (tmp_path / "store").rglob("*"))
    stored.post(f"{VERSION}/task-check", json={"target": "y", "task": "regression"})
    assert sorted(p.name for p in (tmp_path / "store").rglob("*")) == before


def test_task_check_reports_problems_as_a_result_not_an_http_error(stored: TestClient) -> None:
    response = stored.post(f"{VERSION}/task-check", json={"target": "ghost", "task": "regression"})
    assert response.status_code == 200
    assert response.json()["ok"] is False
    codes = {f["code"] for f in response.json()["findings"]}
    assert codes


# -- each rejection -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"", 422, "empty_file"),
        (b"\n\n  \n", 422, "empty_file"),
        (b"a,b\n", 422, "no_data_rows"),
        (b"\xff\xfe\x00\x00binary", 422, "bad_encoding"),
        (b"a,b\n1,\xe9\n", 422, "bad_encoding"),
        (b"a,b\n1,2\x00\n", 422, "nul_byte"),
        (b"a,b\n1,2,3\n", 422, "ragged_row"),
        (b"a,a\n1,2\n", 422, "bad_header"),
        (b"onlyone\n1\n2\n", 422, "bad_delimiter"),
        (
            (",".join(f"c{i}" for i in range(101)) + "\n" + ",".join("1" * 101) + "\n").encode(),
            422,
            "too_many_columns",
        ),
    ],
)
def test_upload_rejections(c: TestClient, data: bytes, status: int, code: str) -> None:
    assert_envelope(upload(c, "bad", data), status, code)
    assert c.get("/datasets").json() == {"datasets": []}  # nothing was stored


def test_too_many_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ApiSettings(port=PORT, store_root=tmp_path / "s", limits=IngestLimits(max_rows=5))
    c = client(make_app(tmp_path, monkeypatch, settings=settings))
    rows = "a,b\n" + "1,2\n" * 6
    assert_envelope(upload(c, "bad", rows.encode()), 422, "too_many_rows")


def test_file_over_the_limit_is_413_even_inside_the_body_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The multipart allowance is separate from the file limit: the body fits, the file does not."""
    settings = ApiSettings(
        port=PORT, store_root=tmp_path / "s", limits=IngestLimits(max_bytes=1000)
    )
    c = client(make_app(tmp_path, monkeypatch, settings=settings))
    ok = ("a,b\n" + "1,2\n" * 200)[:1000]
    ok = ok[: ok.rindex("\n") + 1]
    assert upload(c, "fits", ok.encode()).status_code == 201
    big = ("a,b\n" + "1,2\n" * 300).encode()
    error = assert_envelope(upload(c, "big", big), 413, "too_large")
    assert "limit" in error["message"]


def test_bad_names_and_unknown_things(stored: TestClient) -> None:
    for bad in ("Has Space", "UPPER", "../x", "", "a" * 65):
        response = upload(stored, bad, toy_bytes())
        assert response.status_code == 422, bad
        assert_envelope(response)
    assert_envelope(stored.get("/datasets/ghost"), 404, "dataset_not_found")
    assert_envelope(stored.get("/datasets/toy/versions/9"), 404, "version_not_found")
    assert_envelope(stored.get("/datasets/ghost/versions/1/profile"), 404, "version_not_found")
    assert_envelope(stored.get("/datasets/Bad_Name/versions/1"), 422, "invalid_name")
    assert_envelope(stored.get("/datasets/toy/versions/0"), 422, "validation_error")
    assert_envelope(stored.get("/datasets/toy/versions/abc"), 422, "validation_error")


def test_bad_targets_and_tasks(stored: TestClient) -> None:
    assert_envelope(
        stored.get(f"{VERSION}/preview", params={"target": "ghost"}), 422, "invalid_task_config"
    )
    assert_envelope(
        stored.get(f"{VERSION}/preview", params={"target": "y", "task": "clustering"}),
        422,
        "validation_error",
    )
    assert_envelope(stored.get(f"{VERSION}/validation", params={"target": ""}), 422)
    bodies: list[Any] = [
        {},
        {"target": ""},
        {"target": "y", "task": "clustering"},
        {"target": "y", "surprise": 1},
        {"target": "y", "excluded": "color"},
        {"target": "y", "excluded": ["c"] * 101},
        [],
        "text",
    ]
    for body in bodies:
        assert_envelope(stored.post(f"{VERSION}/task-check", json=body), 422, "validation_error")


def test_missing_and_malformed_upload_fields(c: TestClient) -> None:
    assert_envelope(c.post("/datasets", data={"name": "x"}, files={}), 415)  # not multipart
    assert_envelope(
        c.post("/datasets", data={"name": "x"}, files={"other": ("a.csv", b"a,b\n1,2\n")}),
        422,
        "validation_error",
    )
    assert_envelope(
        c.post("/datasets", files={"file": ("a.csv", b"a,b\n1,2\n")}), 422, "validation_error"
    )
    broken = c.post(
        "/datasets",
        content=b'--nope\r\ncontent-disposition: form-data; name="name"\r\n\r\nx',
        headers={"content-type": "multipart/form-data; boundary=zzz"},
    )
    assert broken.status_code in (400, 422)
    assert_envelope(broken)
    no_boundary = c.post(
        "/datasets", content=b"abc", headers={"content-type": "multipart/form-data"}
    )
    assert_envelope(no_boundary)


def test_malformed_json_body(stored: TestClient) -> None:
    response = stored.post(
        f"{VERSION}/task-check",
        content=b'{"target": ',
        headers={"content-type": "application/json"},
    )
    assert_envelope(response, 422, "validation_error")
    assert "input" not in response.text


def test_framework_errors_use_the_envelope(c: TestClient) -> None:
    assert_envelope(c.get("/nope"), 404, "not_found")
    assert_envelope(c.get("/datasets/toy/versions"), 404, "not_found")
    response = c.delete("/datasets")
    assert_envelope(response, 405, "method_not_allowed")
    assert response.headers["allow"] == "GET, POST"
    assert_envelope(c.put("/health", json={}), 405)
    assert_envelope(c.get("/datasets/toy/versions/1/task-check"), 405)


def test_unexpected_errors_are_generic(
    stored: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"secret detail at {tmp_path} password=hunter2")

    monkeypatch.setattr(routes, "profile_version", boom)
    response = stored.get(f"{VERSION}/profile")
    error = assert_envelope(response, 500, "internal_error")
    assert "hunter2" not in response.text and str(tmp_path) not in response.text
    assert error["message"] == "An internal error occurred."


def test_store_problems_never_show_paths(stored: TestClient, tmp_path: Path) -> None:
    record = tmp_path / "store" / "toy" / "v0001.json"
    record.chmod(0o644)
    record.write_text("{not json")
    response = stored.get(VERSION)
    assert_envelope(response, 500, "store_error")
    assert str(tmp_path) not in response.text and "v0001" not in response.text


def test_altered_file_is_reported_as_a_store_error(stored: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "store" / "toy" / "v0001.csv"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"x,x,x,x\n")
    response = stored.get(f"{VERSION}/profile")
    assert_envelope(response, 500, "store_error")
    assert str(tmp_path) not in response.text


# -- catalog failure ----------------------------------------------------------------------------


def test_catalog_failure_keeps_the_file_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = client(make_app(tmp_path, monkeypatch, fail_catalog=True))
    response = upload(c, "toy", toy_bytes())
    assert response.status_code == 201
    assert response.json()["catalogued"] is False
    assert [d["name"] for d in c.get("/datasets").json()["datasets"]] == ["toy"]
    assert c.get(f"{VERSION}/profile").status_code == 200


# -- health and start-up ------------------------------------------------------------------------


def test_health_ok_degraded_and_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ok = client(make_app(tmp_path, monkeypatch)).get("/health")
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    assert ok.json()["database"] == {
        "reachable": True,
        "at_head": True,
        "revision": "0004",
        "head": "0004",
    }
    behind = DbStatus(
        current=None, head="0004", pending=("0001", "0002", "0003", "0004"), known=True
    )
    response = client(make_app(tmp_path, monkeypatch, db_status=lambda: behind)).get("/health")
    assert response.status_code == 503
    assert (
        response.json()["status"] == "degraded" and response.json()["database"]["at_head"] is False
    )

    def down() -> DbStatus:
        raise ConnectionError("could not connect to postgresql://u:hunter2@127.0.0.1:55432/x")

    gone = client(make_app(tmp_path, monkeypatch, db_status=down)).get("/health")
    assert gone.status_code == 503 and "hunter2" not in gone.text
    assert gone.json()["database"]["reachable"] is False


def test_startup_refuses_a_behind_or_unreachable_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behind = DbStatus(current="0000", head="0001", pending=("0001",), known=True)
    with (
        pytest.raises(StartupError, match="db.py upgrade"),
        TestClient(make_app(tmp_path, monkeypatch, db_status=lambda: behind), base_url=BASE),
    ):
        pass

    def down() -> DbStatus:
        raise ConnectionError("postgresql://u:hunter2@host/db")

    with pytest.raises(StartupError) as caught:
        require_current_schema(down)
    assert "hunter2" not in str(caught.value) and "make db-up" in str(caught.value)
    with TestClient(make_app(tmp_path, monkeypatch), base_url=BASE) as started:
        assert started.get("/health").status_code == 200
    assert require_current_schema(lambda: CURRENT) is CURRENT


# -- concurrency --------------------------------------------------------------------------------


def test_heavy_work_is_limited_to_two_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(tmp_path, monkeypatch)
    real = profile_version
    lock = threading.Lock()
    running = 0
    peak = 0

    def slow(store: Any, ref: str) -> Any:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.15)
        try:
            return real(store, ref)
        finally:
            with lock:
                running -= 1

    monkeypatch.setattr(routes, "profile_version", slow)
    results: list[int] = []
    with TestClient(app, base_url=BASE) as c:
        assert upload(c, "toy", toy_bytes()).status_code == 201

        def get() -> None:
            results.append(c.get(f"{VERSION}/profile").status_code)

        threads = [threading.Thread(target=get) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert results == [200] * 6
    assert 1 <= peak <= 2


# -- contract and parity with the command line --------------------------------------------------


def _cli(store: Path, *args: str) -> dict[str, Any]:
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "datasets.py"),
            "--store",
            str(store),
            *args,
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert done.returncode in (0, 1), done.stderr
    parsed: dict[str, Any] = json.loads(done.stdout)
    return parsed


def test_api_json_equals_cli_json(stored: TestClient, tmp_path: Path) -> None:
    store = tmp_path / "store"
    assert stored.get(f"{VERSION}/profile").json() == _cli(store, "profile", "toy@v1")
    assert stored.get(f"{VERSION}/validation", params={"target": "y"}).json() == _cli(
        store, "validate", "toy@v1", "--target", "y"
    )
    task = stored.post(f"{VERSION}/task-check", json={"target": "y", "excluded": ["color"]}).json()
    assert task == _cli(store, "task", "toy@v1", "--target", "y", "--exclude", "color")
    assert stored.get(f"{VERSION}/preview", params={"target": "y"}).json() == _cli(
        store, "explore", "toy@v1", "--target", "y"
    )


@pytest.mark.parametrize(
    ("file", "target"),
    [("abalone_train_pool.csv", "Rings"), ("adult_train_pool.csv", "income")],
)
def test_models_accept_the_real_reports_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file: str, target: str
) -> None:
    """The typed models fit the library's real output (extra fields are forbidden, so a field
    added to a report would fail here)."""
    source = DEMO / file
    if not source.is_file():
        pytest.skip(f"NOT RUN: {source.name} is not prepared (make demo-data)")
    c = client(make_app(tmp_path, monkeypatch))
    name = file.split("_")[0]
    assert upload(c, name, source.read_bytes()).status_code == 201
    base = f"/datasets/{name}/versions/1"
    store = FileDatasetStore(tmp_path / "store")
    assert c.get(f"{base}/profile").json() == json.loads(
        json.dumps(dataset_reports.profile_payload(profile_version(store, f"{name}@v1")))
    )
    assert c.get(f"{base}/validation", params={"target": target}).status_code == 200
    assert c.get(f"{base}/preview", params={"target": target}).status_code == 200
    assert c.post(f"{base}/task-check", json={"target": target}).status_code == 200


# -- settings and OpenAPI -----------------------------------------------------------------------


def test_origin_parsing() -> None:
    assert origin_of("http://localhost:5173") == ("http", "localhost", 5173)
    assert origin_of("http://LOCALHOST:5173") == ("http", "localhost", 5173)
    assert origin_of("https://a.example") == ("https", "a.example", 443)
    assert origin_of("http://a") == ("http", "a", 80)
    for bad in (
        "null",
        "",
        " http://a",
        "http://a/",
        "http://a/x",
        "http://a?q",
        "ftp://a",
        "http://",
        "localhost:5173",
        "http://u@a",
        "http://a:99999",
        "http://a b",
        "http://a#f",
        "*",
    ):
        assert origin_of(bad) is None, bad


def test_settings_validation() -> None:
    assert ApiSettings.from_environment({}).port == 8100
    env = {"AUTOML_API_PORT": "8200", "AUTOML_UI_ORIGIN": "http://127.0.0.1:3000"}
    configured = ApiSettings.from_environment(env)
    assert configured.port == 8200 and "127.0.0.1:8200" in configured.allowed_hosts
    for bad in (
        {"AUTOML_API_PORT": "abc"},
        {"AUTOML_API_PORT": "80"},
        {"AUTOML_UI_ORIGIN": "http://x/path"},
        {"AUTOML_UI_ORIGIN": "null"},
    ):
        with pytest.raises(SettingsError):
            ApiSettings.from_environment(bad)


def test_openapi_file_is_current_and_describes_every_error() -> None:
    assert OPENAPI_PATH.read_text(encoding="utf-8") == render_openapi(), (
        "docs/openapi-control.json is stale: run `uv run python scripts/export_openapi.py`"
    )
    spec = json.loads(render_openapi())
    assert "HTTPValidationError" not in json.dumps(spec)
    assert set(spec["paths"]) == {
        "/health",
        "/datasets",
        "/datasets/{name}",
        "/datasets/{name}/versions/{number}",
        "/datasets/{name}/versions/{number}/profile",
        "/datasets/{name}/versions/{number}/validation",
        "/datasets/{name}/versions/{number}/preview",
        "/datasets/{name}/versions/{number}/task-check",
        "/algorithms",
        "/jobs",
        "/jobs/{job_id}",
        "/jobs/{job_id}/cancel",
        "/models",
        "/models/{name}",
        "/models/{name}/versions",
        "/models/{name}/versions/{number}",
        "/models/{name}/versions/{number}/promote",
        "/models/{name}/versions/{number}/archive",
        "/models/{name}/rollback",
        "/models/{name}/events",
        "/registry/sync",
        "/models/{name}/sync",
        "/models/{name}/mirror/check",
        "/models/{name}/reconcile",
    }
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            for status in ("404", "422"):
                if path != "/health" or status == "404":
                    ref = operation["responses"][status]["content"]["application/json"]["schema"]
                    assert ref == {"$ref": "#/components/schemas/ErrorEnvelope"}, (path, method)
