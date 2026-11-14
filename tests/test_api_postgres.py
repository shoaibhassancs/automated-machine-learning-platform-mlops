"""P4-3 tests of the API against a real PostgreSQL server (`make db-up`).

Each test gets its own throwaway database (created and dropped here, names `automl_test_*` only).
If the server is unreachable the tests are skipped, which is NOT RUN, never a pass.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL

from api_support import BASE, PORT, client, upload
from automl_ops.api_control.app import StartupError, create_app
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore, sync_catalog
from automl_ops.db.config import DatabaseConfigError, database_url, make_engine
from conftest import toy_bytes

pytestmark = pytest.mark.postgres


def _admin(url: URL) -> Engine:
    return create_engine(
        url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 3},
    )


@pytest.fixture(scope="module")
def server() -> URL:
    try:
        url = database_url()
    except DatabaseConfigError as error:
        pytest.skip(f"NOT RUN: no database settings ({error})")
    try:
        with _admin(url).connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as error:
        pytest.skip(f"NOT RUN: PostgreSQL not reachable ({type(error).__name__}; make db-up)")
    return url


@pytest.fixture
def engine(server: URL) -> Iterator[Engine]:
    name = f"automl_test_{secrets.token_hex(6)}"
    admin = _admin(server)
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    engine = make_engine(server.set(database=name))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def build(engine: Engine, root: Path) -> tuple[TestClient, CatalogedDatasetStore]:
    settings = ApiSettings(port=PORT, store_root=root / "store")
    store = CatalogedDatasetStore(FileDatasetStore(settings.store_root, settings.limits), engine)
    app = create_app(settings, store, lambda: migrate.status(engine))
    return TestClient(app, base_url=BASE, raise_server_exceptions=False), store


def count(engine: Engine, table: str) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())


def test_upload_writes_the_file_then_the_catalog_row(engine: Engine, tmp_path: Path) -> None:
    migrate.upgrade(engine)
    c, _ = build(engine, tmp_path)
    response = upload(c, "toy", toy_bytes())
    assert response.status_code == 201 and response.json()["catalogued"] is True
    sha = response.json()["version"]["sha256"]
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT d.name, v.number, v.sha256, v.rows FROM dataset_versions v "
                "JOIN datasets d ON d.id = v.dataset_id"
            )
        ).one()
    assert tuple(row) == ("toy", 1, sha, 150)
    again = upload(c, "toy", toy_bytes())  # identical content: nothing new, still one row
    assert again.status_code == 200 and again.json()["catalogued"] is True
    assert count(engine, "dataset_versions") == 1
    second = upload(c, "toy", toy_bytes("regression"), add_version=True)
    assert second.status_code == 201 and count(engine, "dataset_versions") == 2


def test_a_catalog_failure_keeps_the_version_and_sync_repairs_it(
    engine: Engine, tmp_path: Path
) -> None:
    migrate.upgrade(engine)
    c, store = build(engine, tmp_path)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE dataset_versions RENAME TO dataset_versions_away"))
    response = upload(c, "toy", toy_bytes())
    assert response.status_code == 201
    assert response.json()["catalogued"] is False
    assert store.unsynced == ["toy@v1"]
    assert c.get("/datasets/toy/versions/1/profile").status_code == 200  # the file store serves it
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE dataset_versions_away RENAME TO dataset_versions"))
    result = sync_catalog(engine, FileDatasetStore(tmp_path / "store"))
    assert result.ok and result.added == ("toy@v1",)
    assert count(engine, "dataset_versions") == 1
    assert migrate.check(engine).ok  # the rename round trip left the structure as designed


def test_start_up_refuses_until_the_database_is_at_the_head(engine: Engine, tmp_path: Path) -> None:
    c, _ = build(engine, tmp_path)
    with pytest.raises(StartupError, match="db.py upgrade"), c:
        pass
    migrate.upgrade(engine)
    c, _ = build(engine, tmp_path)
    with c:
        health = c.get("/health")
        assert health.status_code == 200
        assert health.json()["database"] == {
            "reachable": True,
            "at_head": True,
            "revision": "0004",
            "head": "0004",
        }


def test_start_up_refuses_when_the_database_is_unreachable(server: URL, tmp_path: Path) -> None:
    dead = make_engine(server.set(port=1, database="automl_ops"))  # nothing listens on port 1
    c, _ = build(dead, tmp_path)
    with pytest.raises(StartupError) as caught, c:
        pass
    assert "make db-up" in str(caught.value)
    assert str(server.password) not in str(caught.value)
    health = client(
        create_app(
            ApiSettings(port=PORT, store_root=tmp_path / "s"),
            CatalogedDatasetStore(FileDatasetStore(tmp_path / "s"), dead),
            lambda: migrate.status(dead),
        )
    ).get("/health")
    assert health.status_code == 503 and health.json()["database"]["reachable"] is False
    assert str(server.password) not in health.text
    dead.dispose()
