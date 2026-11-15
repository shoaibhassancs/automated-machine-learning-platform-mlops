"""Follow-up to P4-10: the end-to-end story only ever runs on a stack made for it.

Pure tests cover the isolation report and the refusal checks (every refusal is proved to make only
GET requests). The PostgreSQL tests (throwaway databases only) cover the health fields on both
services, real zero-write refusals, and the wrapper's create / start / verify / cleanup, including
ownership proofs. Nothing here connects to the real `automl_ops` or `mlflow` databases or touches
`data/`; the real database name only appears in pure checks of the name rule.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

from automl_ops import isolation as iso
from automl_ops.e2e_stack import COMMENT_PREFIX, MARKER, IsolatedStack, clean_environment
from automl_ops.isolation import Isolation, Wiring, assess, database_id, run_id, verify_stack
from automl_ops.simulator import Reply

TOKEN = "0123456789abcdef0123456789abcdef"
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"


def good_env(tmp: Path, **over: str) -> dict[str, str]:
    return {
        "AUTOML_JOBS_MLFLOW_DIR": str(tmp / "mlflow"),
        "AUTOML_MLFLOW_BACKEND": "sqlite",
        iso.TOKEN_SETTING: TOKEN,
        **over,
    }


def url(name: str = "automl_test_e2e_abcd1234") -> URL:
    return make_url(f"postgresql+psycopg://automl:s3cretpw@127.0.0.1:55432/{name}")


def wiring(tmp: Path, name: str = "automl_test_e2e_abcd1234", **over: Any) -> Wiring:
    env = good_env(tmp, **{k: v for k, v in over.items() if isinstance(v, str)})
    return Wiring(
        url(name), tmp / "store", tmp / "models", env=env, project_data=over.get("data", DATA)
    )


# -- the isolation report ---------------------------------------------------------------------


def test_the_default_stack_is_not_isolated_and_says_why() -> None:
    default = Wiring(url("automl_ops"), DATA / "store", DATA / "models", env={})
    report = assess(default)
    assert report.isolated is False and not any(report.checks.values())
    assert set(report.reasons) == {
        "database_not_throwaway",
        "store_inside_protected_data",
        "models_inside_protected_data",
        "mlflow_not_configured",
        "run_token_missing",
    }


def test_a_stack_made_for_the_story_is_isolated(tmp_path: Path) -> None:
    report = assess(wiring(tmp_path))
    assert report.isolated and report.reasons == [] and all(report.checks.values())
    assert report.database_id == database_id(url()) and report.run_id == run_id(TOKEN)


def test_the_report_carries_no_path_host_user_password_or_token(tmp_path: Path) -> None:
    blob = json.dumps(assess(wiring(tmp_path)).to_dict())
    for secret in (str(tmp_path), "127.0.0.1", "55432", "automl:", "s3cretpw", TOKEN, "tmp"):
        assert secret not in blob
    assert "automl_test_e2e" not in blob  # not even the database name: only a one-way id


@pytest.mark.parametrize("name", ["automl_ops", "mlflow", "postgres", "template1", "automl_test_x"])
def test_only_throwaway_database_names_pass(tmp_path: Path, name: str) -> None:
    report = assess(wiring(tmp_path, name))
    assert not report.isolated and report.checks["database"] is False


def test_paths_inside_the_protected_data_directory_are_refused(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "store").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(data / "store")
    cases = {
        "store": Wiring(url(), data, tmp_path / "m", env=good_env(tmp_path), project_data=data),
        "models": Wiring(
            url(), tmp_path / "s", data / "models", env=good_env(tmp_path), project_data=data
        ),
        "store (via a symbolic link)": Wiring(
            url(), link, tmp_path / "m", env=good_env(tmp_path), project_data=data
        ),
    }
    for label, w in cases.items():
        assert not assess(w).isolated, label
    archive = good_env(tmp_path, AUTOML_JOBS_MLFLOW_DIR=str(data / "mlflow"))
    archived = Wiring(url(), tmp_path / "s", tmp_path / "m", env=archive, project_data=data)
    assert assess(archived).reasons == ["mlflow_inside_protected_data"]


def test_a_postgresql_mlflow_backend_is_never_isolated(tmp_path: Path) -> None:
    postgres = good_env(tmp_path, AUTOML_MLFLOW_BACKEND="postgresql")
    w = Wiring(url(), tmp_path / "s", tmp_path / "m", env=postgres)
    assert assess(w).reasons == ["mlflow_not_sqlite"]
    assert assess(Wiring(url(), tmp_path / "s", tmp_path / "m", env={})).checks["mlflow"] is False


def test_the_token_must_be_present_and_long_enough(tmp_path: Path) -> None:
    short = good_env(tmp_path, **{iso.TOKEN_SETTING: "short"})
    w = Wiring(url(), tmp_path / "s", tmp_path / "m", env=short)
    assert assess(w).reasons == ["run_token_missing"]


# -- verify_stack: refusals make GET requests only ---------------------------------------------


class Fake:
    """A service double that records every request it receives."""

    def __init__(self, replies: dict[str, Reply]) -> None:
        self.replies, self.calls = replies, []  # type: ignore[var-annotated]

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Reply:
        self.calls.append((method, path))
        return self.replies.get(path, Reply(404, None))


def health(tmp: Path, **over: Any) -> Reply:
    block = assess(wiring(tmp)).to_dict()
    block.update(over)
    return Reply(200, {"status": "ok", "isolation": block})


def empty() -> dict[str, Reply]:
    return {
        "/datasets": Reply(200, {"datasets": []}),
        "/models": Reply(200, {"models": []}),
        "/jobs": Reply(200, {"jobs": [], "next_before_id": None}),
        "/deployments": Reply(200, {"deployments": [], "next_before_id": None}),
    }


def pair(tmp: Path, control: dict[str, Reply] | None = None, serving: Reply | None = None) -> Any:
    c = Fake({"/health": health(tmp), **empty(), **(control or {})})
    s = Fake({"/health": serving or health(tmp)})
    return c, s


def local_ok(tmp: Path) -> Isolation:
    return assess(wiring(tmp))


def only_reads(*fakes: Fake) -> bool:
    return all(method == "GET" for f in fakes for method, _ in f.calls)


def test_an_isolated_empty_stack_passes(tmp_path: Path) -> None:
    c, s = pair(tmp_path)
    assert verify_stack(local_ok(tmp_path), c, s) == [] and only_reads(c, s)


def test_a_script_with_default_settings_contacts_nothing(tmp_path: Path) -> None:
    c, s = pair(tmp_path)
    default = assess(Wiring(url("automl_ops"), DATA / "store", DATA / "models", env={}))
    problems = verify_stack(default, c, s)
    assert problems and c.calls == [] and s.calls == []


def test_a_default_control_plane_is_refused(tmp_path: Path) -> None:
    default = assess(Wiring(url("automl_ops"), DATA / "store", DATA / "models", env={}))
    c, s = pair(tmp_path, {"/health": Reply(200, {"isolation": default.to_dict()})})
    problems = verify_stack(local_ok(tmp_path), c, s)
    assert any("control plane: not isolated" in p for p in problems) and only_reads(c, s)
    assert ("GET", "/datasets") not in c.calls  # nothing past the health check


def test_a_default_serving_app_is_refused(tmp_path: Path) -> None:
    default = assess(Wiring(url("automl_ops"), DATA / "store", DATA / "models", env={}))
    c, s = pair(tmp_path, serving=Reply(200, {"isolation": default.to_dict()}))
    problems = verify_stack(local_ok(tmp_path), c, s)
    assert any("serving app: not isolated" in p for p in problems) and only_reads(c, s)


def test_services_on_a_different_database_or_run_are_refused(tmp_path: Path) -> None:
    other_db = database_id(url("automl_test_e2e_ffff0000"))
    c, s = pair(tmp_path, serving=health(tmp_path, database_id=other_db))
    assert any("different database" in p for p in verify_stack(local_ok(tmp_path), c, s))
    c, s = pair(tmp_path, {"/health": health(tmp_path, run_id=run_id("someone else's"))})
    assert any("different run" in p for p in verify_stack(local_ok(tmp_path), c, s))
    assert only_reads(c, s)


def test_a_stack_that_does_not_report_isolation_is_refused(tmp_path: Path) -> None:
    c, s = pair(tmp_path, {"/health": Reply(200, {"status": "ok", "database": {}})})
    problems = verify_stack(local_ok(tmp_path), c, s)
    assert any("does not report isolation" in p for p in problems) and only_reads(c, s)
    c, s = pair(tmp_path, {"/health": Reply(0, None)})
    assert any("not answering" in p for p in verify_stack(local_ok(tmp_path), c, s))


@pytest.mark.parametrize("key", ["datasets", "models", "jobs", "deployments"])
def test_a_stack_with_existing_data_is_refused(tmp_path: Path, key: str) -> None:
    c, s = pair(tmp_path, {f"/{key}": Reply(200, {key: [{"id": 1}], "next_before_id": None})})
    problems = verify_stack(local_ok(tmp_path), c, s)
    assert problems == [f"the stack is not empty ({key} exist)"] and only_reads(c, s)


def test_an_unreadable_list_is_a_refusal_not_a_pass(tmp_path: Path) -> None:
    c, s = pair(tmp_path, {"/jobs": Reply(500, None)})
    assert verify_stack(local_ok(tmp_path), c, s) == ["control plane: could not list jobs"]


# -- the script itself -------------------------------------------------------------------------


def test_the_script_has_no_force_or_bypass_option() -> None:
    out = subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "e2e_demo.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    options = {w for w in out.replace(",", " ").split() if w.startswith("--")}
    assert options == {
        "--help",
        "--api-port",
        "--serve-port",
        "--rows",
        "--seed",
        "--timeout",
        "--model",
    }
    source = (PROJECT / "scripts" / "e2e_demo.py").read_text()
    for word in ("--force", "--yes", "--skip", "--unsafe", "--override", "--i-know"):
        assert word not in source


class Counting(BaseHTTPRequestHandler):
    seen: list[tuple[str, str]] = []

    def _count(self) -> None:
        type(self).seen.append((self.command, self.path))
        self.send_response(500)
        self.send_header("content-length", "0")
        self.end_headers()

    do_GET = do_POST = do_PUT = do_DELETE = _count  # noqa: N815

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def counting() -> Iterator[tuple[int, type[Counting]]]:
    Counting.seen = []
    server = HTTPServer(("127.0.0.1", 0), Counting)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], Counting
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.mark.skipif(not (DATA / "demo" / "manifest.json").exists(), reason="demo data not prepared")
def test_run_directly_with_default_settings_the_script_refuses_and_contacts_nothing(
    counting: tuple[int, type[Counting]],
) -> None:
    """The script is started with the project's normal settings (no isolation variables): it must
    refuse before a single request reaches the stack. A dummy URL keeps it from reading `.env`."""
    port, handler = counting
    env = {
        **clean_environment(),
        "AUTOML_DATABASE_URL": "postgresql://automl:x@127.0.0.1:1/automl_ops",
    }
    done = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts" / "e2e_demo.py"),
            "abalone",
            "--api-port",
            str(port),
            "--serve-port",
            str(port),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT,
    )
    assert done.returncode == 2 and "REFUSED" in done.stderr
    assert "database_not_throwaway" in done.stderr and "store_inside_protected_data" in done.stderr
    assert handler.seen == []


def test_the_child_environment_is_built_without_inherited_settings() -> None:
    env = clean_environment(
        {
            "PATH": "/bin",
            "AUTOML_DATABASE_URL": "x",
            "MLFLOW_TRACKING_URI": "y",
            "POSTGRES_PASSWORD": "z",
        }
    )
    assert env == {"PATH": "/bin"}


# -- real PostgreSQL ----------------------------------------------------------------------------

pg = pytest.mark.postgres


def databases(engine: Any) -> list[str]:
    with engine.connect() as c:
        return [r[0] for r in c.execute(text("SELECT datname FROM pg_database ORDER BY 1"))]


def tree_fingerprint(root: Path) -> tuple[int, int, int]:
    count = size = newest = 0
    for current, _, files in os.walk(root):
        for name in files:
            st = os.stat(Path(current, name))
            count, size, newest = count + 1, size + st.st_size, max(newest, int(st.st_mtime))
    return count, size, newest


@pg
def test_both_services_report_isolation_on_health_and_a_non_throwaway_database_is_refused(
    job_env: Any, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from api_support import BASE
    from automl_ops.api_control.app import create_app as create_control
    from automl_ops.api_control.settings import ApiSettings
    from automl_ops.api_serving.app import create_app as create_serving
    from automl_ops.api_serving.runtime import ServingRuntime
    from automl_ops.api_serving.settings import ServingSettings
    from automl_ops.dataset_store import FileDatasetStore
    from automl_ops.db import migrate
    from automl_ops.db.catalog import CatalogedDatasetStore
    from test_simulator import AppHttp

    engine = job_env.engine
    env = good_env(tmp_path)
    report = iso.reporter(Wiring(engine.url, job_env.store_root, job_env.models, env=env))
    settings = ApiSettings(store_root=job_env.store_root, models_root=job_env.models)
    control = create_control(
        settings,
        CatalogedDatasetStore(FileDatasetStore(job_env.store_root, settings.limits), engine),
        lambda: migrate.status(engine),
        isolation=report,
    )
    runtime = ServingRuntime(engine, job_env.models, retry_seconds=1.0)
    serving = create_serving(
        ServingSettings(port=8200, models_root=job_env.models, poll_seconds=60), runtime, report
    )
    c = AppHttp(TestClient(control, base_url=BASE, raise_server_exceptions=False))
    s = AppHttp(
        TestClient(serving, base_url="http://127.0.0.1:8200", raise_server_exceptions=False)
    )
    with engine.connect() as conn:
        before = conn.execute(text("SELECT count(*) FROM datasets")).scalar_one()
    for client in (c, s):
        reply = client.request("GET", "/health")
        block = (reply.body or {})["isolation"]
        assert block["isolated"] is False and "database_not_throwaway" in block["reasons"]
        blob = json.dumps(reply.body)
        assert str(job_env.store_root) not in blob and "password" not in blob.lower()
    # the toy fixture's data exists: the story would refuse even if the names matched; and the
    # refusal itself wrote nothing
    problems = verify_stack(assess(wiring(tmp_path)), c, s)
    assert any("not isolated (database_not_throwaway" in p for p in problems)
    assert only_reads(c, s)  # type: ignore[arg-type]
    assert all(m == "GET" for m, _ in c.calls + s.calls)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM datasets")).scalar_one() == before


@pg
def test_an_app_without_an_injected_report_says_not_assessed(job_env: Any, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from api_support import BASE
    from automl_ops.api_control.app import create_app as create_control
    from automl_ops.api_control.settings import ApiSettings
    from automl_ops.dataset_store import FileDatasetStore
    from automl_ops.db import migrate
    from automl_ops.db.catalog import CatalogedDatasetStore

    engine = job_env.engine
    settings = ApiSettings(store_root=job_env.store_root, models_root=job_env.models)
    app = create_control(
        settings,
        CatalogedDatasetStore(FileDatasetStore(job_env.store_root, settings.limits), engine),
        lambda: migrate.status(engine),
    )
    body = TestClient(app, base_url=BASE).get("/health").json()
    assert body["isolation"]["isolated"] is False and body["isolation"]["reasons"] == [
        "not_assessed"
    ]


@pg
def test_the_wrapper_creates_verifies_and_removes_only_its_own_resources(
    pg_engine: Any, pg_url: URL, tmp_path: Path
) -> None:
    from automl_ops.simulator import LocalHttp

    before_databases = databases(pg_engine)
    before_data = tree_fingerprint(DATA) if DATA.exists() else None
    stack = IsolatedStack(base=pg_url, parent=tmp_path)
    left: list[str] = ["not run"]
    try:
        stack.create()
        assert stack.database.startswith("automl_test_e2e_") and stack.root is not None
        assert stack.root.parent == tmp_path.resolve() and (stack.root / MARKER).is_file()
        assert stack.database in databases(pg_engine)
        # packages are read-only by design: cleanup must still remove them
        sealed = stack.root / "models" / "pkg" / "model.joblib"
        sealed.parent.mkdir(parents=True)
        sealed.write_bytes(b"x")
        sealed.chmod(0o444)
        sealed.parent.chmod(0o555)
        stack.start()
        control, serving = LocalHttp(stack.api_port), LocalHttp(stack.serve_port)
        local = assess(
            Wiring(
                stack.database_url,
                stack.root / "store",
                stack.root / "models",
                env=stack.environment(),
            )
        )
        assert local.isolated, local.reasons
        assert verify_stack(local, control, serving) == []  # real services, both report isolation
        # the real services report their own view, with no path or credential in it
        raw = json.dumps(control.request("GET", "/health").body)
        assert str(stack.root) not in raw and stack.token not in raw and "s3cret" not in raw
        # a stack with data in it is refused: upload a dataset through the API
        body = (
            "x,y\n" + "\n".join(f"{i},{'a' if i % 2 else 'b'}" for i in range(40)) + "\n"
        ).encode()
        up = control.upload("/datasets", {"name": "stray"}, "stray.csv", body)
        assert up.status in (200, 201), up
        assert verify_stack(local, control, serving) == ["the stack is not empty (datasets exist)"]
    finally:
        left = stack.cleanup()
    assert left == []
    assert stack.database not in databases(pg_engine)
    assert stack.root is not None and not stack.root.exists()
    assert databases(pg_engine) == before_databases  # nothing else was created or dropped
    assert (tree_fingerprint(DATA) if DATA.exists() else None) == before_data
    assert stack.processes == {}


@pg
def test_cleanup_deletes_nothing_when_ownership_cannot_be_proved(
    pg_engine: Any, pg_url: URL, tmp_path: Path
) -> None:
    stack = IsolatedStack(base=pg_url, parent=tmp_path)
    stack.create()
    assert stack.root is not None
    try:
        marker = stack.root / MARKER
        marker.write_text("someone else", encoding="utf-8")
        left = stack.cleanup()
        assert any("owner" in p or "belongs" in p for p in left)
        assert stack.root.exists() and stack.database in databases(pg_engine)  # nothing deleted
        marker.write_text(run_id(stack.token), encoding="utf-8")
        admin = create_engine(pg_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text(f"COMMENT ON DATABASE \"{stack.database}\" IS 'not mine'"))
        left = stack.cleanup()
        assert any("not stamped" in p for p in left)
        assert stack.root.exists() and stack.database in databases(pg_engine)
        with admin.connect() as c:
            stamp = COMMENT_PREFIX + run_id(stack.token)
            c.execute(text(f"COMMENT ON DATABASE \"{stack.database}\" IS '{stamp}'"))
        admin.dispose()
    finally:
        assert stack.cleanup() == []  # the test's own resources are removed once proved
    assert stack.database not in databases(pg_engine) and not stack.root.exists()


@pg
def test_cleanup_never_drops_a_database_it_did_not_create(
    pg_engine: Any, pg_url: URL, tmp_path: Path
) -> None:
    decoy = "automl_test_e2e_decoy0001"
    admin = create_engine(pg_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{decoy}"'))
        c.execute(text(f'CREATE DATABASE "{decoy}"'))
    try:
        stack = IsolatedStack(base=pg_url, parent=tmp_path, database=decoy)
        left = stack.cleanup()
        assert any("not stamped" in p for p in left) and decoy in databases(pg_engine)
        for protected in ("automl_ops", "mlflow", "postgres"):
            other = IsolatedStack(base=pg_url, parent=tmp_path, database=protected)
            assert other.ownership_problems() == ["the database name is not a throwaway name"]
        stranger = tmp_path / "automl-e2e-stranger"
        stranger.mkdir()
        (stranger / "keep.txt").write_text("mine")
        victim = IsolatedStack(base=pg_url, parent=tmp_path, root=stranger)
        assert any("marker" in p for p in victim.cleanup())
        assert (stranger / "keep.txt").read_text() == "mine"  # an unmarked folder is never removed
        project_folder = IsolatedStack(base=pg_url, parent=tmp_path, root=PROJECT / "data")
        assert project_folder.ownership_problems()  # `data/` can never be proved to be owned
    finally:
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{decoy}" WITH (FORCE)'))
        admin.dispose()


def test_the_ui_stack_environment_allows_exactly_the_ui_origin_and_stays_isolated(
    tmp_path: Path,
) -> None:
    """`scripts/ui_stack.py` (Phase 5) builds on `IsolatedStack`: given ports and a UI origin, the
    environment it hands the services is still isolated and carries that one browser origin."""
    stack = IsolatedStack(
        base=url("postgres"),
        parent=tmp_path,
        api_port=18101,
        serve_port=18201,
        ui_origin="http://localhost:5301",
    )
    stack.database = "automl_test_e2e_feedbeef"
    stack.root = tmp_path / "automl-e2e-x"
    env = stack.environment()
    assert env["AUTOML_UI_ORIGIN"] == "http://localhost:5301"
    assert env["AUTOML_API_PORT"] == "18101" and env["AUTOML_SERVING_PORT"] == "18201"
    assert env["AUTOML_MLFLOW_BACKEND"] == "sqlite" and env["AUTOML_REGISTRY_MIRROR"] == "off"
    report = assess(
        Wiring(stack.database_url, stack.root / "store", stack.root / "models", env=env)
    )
    assert report.isolated, report.reasons
