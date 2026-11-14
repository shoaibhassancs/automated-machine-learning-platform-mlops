"""P4-1 tests against a real PostgreSQL server (`make db-up`).

Each test gets its own empty database, created and dropped here (only names this file created are
ever dropped). If the server cannot be reached the tests are *skipped*, and pytest's summary says
so: a skip is NOT RUN, never a pass. Real PostgreSQL is required because the constraints and
`ON CONFLICT` behaviour under test are PostgreSQL-specific.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DataError, IntegrityError

from automl_ops.dataset_store import DatasetStore, FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore, sync_catalog
from automl_ops.db.config import DatabaseConfigError, database_url, make_engine
from conftest import toy_bytes

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def _server_url() -> URL:
    try:
        return database_url()
    except DatabaseConfigError as error:
        pytest.skip(f"NOT RUN: no database settings ({error})")


def _admin(url: URL) -> Engine:
    return create_engine(
        url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 3},
    )


@pytest.fixture(scope="module")
def server() -> URL:
    url = _server_url()
    try:
        with _admin(url).connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as error:
        pytest.skip(
            f"NOT RUN: PostgreSQL not reachable at {url.host}:{url.port} (start it: make db-up): "
            f"{type(error).__name__}"
        )
    return url


@pytest.fixture
def url(server: URL) -> Iterator[URL]:
    """A fresh, empty database for one test."""
    name = f"automl_test_{secrets.token_hex(6)}"
    admin = _admin(server)
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield server.set(database=name)
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def engine(url: URL) -> Iterator[Engine]:
    engine = make_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture
def migrated(engine: Engine) -> Engine:
    migrate.upgrade(engine)
    return engine


def make_store(root: Path) -> FileDatasetStore:
    store = FileDatasetStore(root)
    store.ingest("alpha", toy_bytes("classification", 40), source_filename="a.csv")
    store.ingest(
        "alpha", toy_bytes("classification", 41), source_filename="a2.csv", add_version=True
    )
    store.ingest("beta", toy_bytes("regression", 30), source_filename="b.csv")
    return store


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
            digest.update(str(stat.S_IMODE(path.stat().st_mode)).encode())
    return digest.hexdigest()


def count(engine: Engine, table: str) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())


# --- migrations --------------------------------------------------------------------------------


def test_upgrade_builds_the_catalog_and_job_tables_on_an_empty_database(engine: Engine) -> None:
    before = migrate.status(engine)
    assert (before.current, before.pending, before.up_to_date) == (
        None,
        ("0001", "0002", "0003", "0004"),
        False,
    )
    after = migrate.upgrade(engine)
    assert after.up_to_date and after.current == "0004" and after.pending == ()
    tables = set(inspect(engine).get_table_names())
    assert tables == {
        "alembic_version", "datasets", "dataset_versions", "training_jobs", "job_events",
        "registered_models", "model_versions", "registry_events", "registry_sync",
        "registry_sync_log",
    }  # fmt: skip


def test_upgrading_twice_is_a_no_op(migrated: Engine) -> None:
    assert migrate.upgrade(migrated).up_to_date
    assert migrate.check(migrated).ok


def test_up_down_up(engine: Engine) -> None:
    migrate.upgrade(engine)
    down = migrate.downgrade(engine, "base")
    assert down.current is None
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    migrate.upgrade(engine)
    assert migrate.upgrade(engine).up_to_date
    assert migrate.check(engine).ok


def test_concurrent_upgrades_serialise_and_all_succeed(url: URL) -> None:
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def run() -> None:
        engine = make_engine(url)
        try:
            barrier.wait()
            migrate.upgrade(engine)
        except BaseException as error:
            errors.append(error)
        finally:
            engine.dispose()

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    engine = make_engine(url)
    try:
        assert migrate.check(engine).ok
    finally:
        engine.dispose()


def test_check_reports_a_database_that_is_not_upgraded(engine: Engine) -> None:
    result = migrate.check(engine)
    assert not result.ok and "db upgrade" in result.problems[0]


def test_check_reports_structure_drift(migrated: Engine) -> None:
    with migrated.begin() as connection:
        connection.execute(text("ALTER TABLE datasets ADD COLUMN extra text"))
    result = migrate.check(migrated)
    assert not result.ok and any("extra" in p for p in result.problems)


def test_check_reports_a_missing_check_constraint(migrated: Engine) -> None:
    with migrated.begin() as connection:
        connection.execute(
            text("ALTER TABLE dataset_versions DROP CONSTRAINT ck_dataset_versions_sha256_format")
        )
    result = migrate.check(migrated)
    assert not result.ok
    assert any("ck_dataset_versions_sha256_format" in p for p in result.problems)


def test_an_unknown_revision_is_reported_not_guessed(migrated: Engine) -> None:
    with migrated.begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num = 'zzz'"))
    state = migrate.status(migrated)
    assert not state.known and not state.up_to_date
    assert not migrate.check(migrated).ok


def test_the_actual_constraint_names_are_the_declared_ones(migrated: Engine) -> None:
    with migrated.connect() as connection:
        names = set(
            connection.execute(
                text(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conrelid IN ('datasets'::regclass, 'dataset_versions'::regclass)"
                )
            ).scalars()
        )
    assert names == {
        "pk_datasets", "uq_datasets_name", "ck_datasets_name_format", "pk_dataset_versions",
        "fk_dataset_versions_dataset_id_datasets", "uq_dataset_versions_dataset_number",
        "uq_dataset_versions_dataset_sha256", "ck_dataset_versions_number_positive",
        "ck_dataset_versions_sha256_format", "ck_dataset_versions_original_sha256_format",
        "ck_dataset_versions_sizes", "ck_dataset_versions_counts",
    }  # fmt: skip


# --- constraints -------------------------------------------------------------------------------

VERSION_SQL = (
    "INSERT INTO dataset_versions (dataset_id, number, sha256, original_sha256, size_bytes, "
    "original_size_bytes, rows, column_names, source_filename, original_encoding, "
    "original_delimiter, reencoded, trailing_blank_lines_ignored, store_format, created_at) "
    "VALUES (:d, :n, :s, :s, :b, :b, :r, '[]', 'f.csv', 'utf-8', ',', false, 0, 1, now())"
)


def add_version(engine: Engine, **overrides: object) -> None:
    params: dict[str, object] = {"d": 1, "n": 1, "s": SHA, "b": 10, "r": 3, **overrides}
    with engine.begin() as connection:
        connection.execute(text(VERSION_SQL), params)


@pytest.fixture
def with_dataset(migrated: Engine) -> Engine:
    with migrated.begin() as connection:
        connection.execute(text("INSERT INTO datasets (name) VALUES ('alpha')"))
    return migrated


@pytest.mark.parametrize("name", ["", "Upper", "-lead", "has space", "x" * 65, "a/b"])
def test_dataset_names_are_validated_by_the_database(migrated: Engine, name: str) -> None:
    with pytest.raises((IntegrityError, DataError)), migrated.begin() as connection:
        connection.execute(text("INSERT INTO datasets (name) VALUES (:n)"), {"n": name})


def test_dataset_names_are_unique(with_dataset: Engine) -> None:
    with pytest.raises(IntegrityError), with_dataset.begin() as connection:
        connection.execute(text("INSERT INTO datasets (name) VALUES ('alpha')"))


@pytest.mark.parametrize(
    "bad",
    [
        {"s": "A" * 64},  # upper-case hex
        {"s": "a" * 63},
        {"s": "g" * 64},
        {"n": 0},
        {"b": -1},
        {"r": -1},
        {"d": 999},  # no such dataset
    ],
)
def test_version_rows_are_validated(with_dataset: Engine, bad: dict[str, object]) -> None:
    with pytest.raises(IntegrityError):
        add_version(with_dataset, **bad)


def test_a_version_number_and_a_hash_are_unique_per_dataset(with_dataset: Engine) -> None:
    add_version(with_dataset)
    with pytest.raises(IntegrityError):
        add_version(with_dataset, s="b" * 64)  # same number
    with pytest.raises(IntegrityError):
        add_version(with_dataset, n=2)  # same hash


def test_a_dataset_with_versions_cannot_be_deleted(with_dataset: Engine) -> None:
    add_version(with_dataset)
    with pytest.raises(IntegrityError), with_dataset.begin() as connection:
        connection.execute(text("DELETE FROM datasets"))


# --- catalog sync ------------------------------------------------------------------------------


def test_sync_catalogs_every_version_from_the_records(migrated: Engine, tmp_path: Path) -> None:
    store = make_store(tmp_path / "store")
    result = sync_catalog(migrated, store)
    assert result.ok and result.present == 0
    assert result.added == ("alpha@v1", "alpha@v2", "beta@v1")
    assert (count(migrated, "datasets"), count(migrated, "dataset_versions")) == (2, 3)
    with migrated.connect() as connection:
        row = connection.execute(
            text(
                "SELECT d.name, v.number, v.sha256, v.rows, v.column_names, v.size_bytes, "
                "v.source_filename, v.created_at FROM dataset_versions v "
                "JOIN datasets d ON d.id = v.dataset_id WHERE d.name='alpha' AND v.number=2"
            )
        ).one()
    version = store.get_version("alpha@v2")
    assert (row.sha256, row.rows, row.size_bytes) == (
        version.sha256,
        version.rows,
        version.size_bytes,
    )
    assert row.column_names == list(version.columns) and row.source_filename == "a2.csv"
    assert row.created_at.isoformat() == version.created_at


def test_sync_is_idempotent(migrated: Engine, tmp_path: Path) -> None:
    store = make_store(tmp_path / "store")
    sync_catalog(migrated, store)
    again = sync_catalog(migrated, store)
    assert again.ok and again.added == () and again.present == 3
    assert count(migrated, "dataset_versions") == 3


def test_sync_rebuilds_missing_rows(migrated: Engine, tmp_path: Path) -> None:
    store = make_store(tmp_path / "store")
    sync_catalog(migrated, store)
    with migrated.begin() as connection:  # lose one version row, and one dataset with its row
        connection.execute(text("DELETE FROM dataset_versions WHERE number = 2"))
        connection.execute(
            text(
                "DELETE FROM dataset_versions WHERE dataset_id IN "
                "(SELECT id FROM datasets WHERE name = 'beta')"
            )
        )
        connection.execute(text("DELETE FROM datasets WHERE name = 'beta'"))
    assert count(migrated, "dataset_versions") == 1
    result = sync_catalog(migrated, store)
    assert result.ok and result.added == ("alpha@v2", "beta@v1") and result.present == 1
    assert (count(migrated, "datasets"), count(migrated, "dataset_versions")) == (2, 3)


def test_sync_reports_an_orphan_row_and_deletes_nothing(migrated: Engine, tmp_path: Path) -> None:
    store = make_store(tmp_path / "store")
    sync_catalog(migrated, store)
    with migrated.begin() as connection:
        connection.execute(text("INSERT INTO datasets (name) VALUES ('ghost')"))
    add_version(migrated, d=_dataset_id(migrated, "ghost"), n=1, s="c" * 64)
    result = sync_catalog(migrated, store)
    assert result.orphans == ("ghost@v1",) and not result.ok
    assert count(migrated, "dataset_versions") == 4


def _dataset_id(engine: Engine, name: str) -> int:
    with engine.connect() as connection:
        return int(
            connection.execute(
                text("SELECT id FROM datasets WHERE name = :n"), {"n": name}
            ).scalar_one()
        )


def test_sync_reports_a_hash_conflict_and_changes_nothing(migrated: Engine, tmp_path: Path) -> None:
    store = make_store(tmp_path / "store")
    sync_catalog(migrated, store)
    with migrated.begin() as connection:
        connection.execute(
            text(
                "UPDATE dataset_versions SET sha256 = :s FROM datasets d "
                "WHERE d.id = dataset_id AND d.name='beta'"
            ),
            {"s": "d" * 64},
        )
    result = sync_catalog(migrated, store)
    assert len(result.conflicts) == 1 and "beta@v1" in result.conflicts[0] and not result.ok
    with migrated.connect() as connection:
        shas = (
            connection.execute(
                text(
                    "SELECT sha256 FROM dataset_versions v "
                    "JOIN datasets d ON d.id = v.dataset_id WHERE d.name = 'beta'"
                )
            )
            .scalars()
            .all()
        )
    assert shas == ["d" * 64]  # reported, not overwritten


def test_sync_does_not_catalog_a_version_whose_file_fails_verification(
    migrated: Engine, tmp_path: Path
) -> None:
    store = make_store(tmp_path / "store")
    victim = store.version_path("beta@v1")
    victim.chmod(0o644)
    victim.write_bytes(victim.read_bytes() + b"tampered\n")
    result = sync_catalog(migrated, store)
    assert any(u.startswith("beta@v1") for u in result.unverified) and not result.ok
    assert result.added == ("alpha@v1", "alpha@v2")
    assert count(migrated, "dataset_versions") == 2


def test_sync_never_modifies_the_file_store(migrated: Engine, tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = make_store(root)
    before = tree_digest(root)
    sync_catalog(migrated, store)
    sync_catalog(migrated, store)
    assert tree_digest(root) == before


def test_concurrent_syncs_leave_exactly_one_row_per_version(url: URL, tmp_path: Path) -> None:
    setup = make_engine(url)
    migrate.upgrade(setup)
    store = make_store(tmp_path / "store")
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def run() -> None:
        engine = make_engine(url)
        try:
            barrier.wait()
            sync_catalog(engine, store)
        except BaseException as error:
            errors.append(error)
        finally:
            engine.dispose()

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert errors == []
        assert (count(setup, "datasets"), count(setup, "dataset_versions")) == (2, 3)
        assert sync_catalog(setup, store).ok
    finally:
        setup.dispose()


def test_the_catalog_holds_no_dataset_contents(migrated: Engine, tmp_path: Path) -> None:
    store = make_store(tmp_path / "store")
    sync_catalog(migrated, store)
    csv_text = store.version_path("alpha@v1").read_text()
    sample_cells = [line.split(",")[0] for line in csv_text.splitlines()[1:6]]
    with migrated.connect() as connection:
        dump = (
            connection.execute(text("SELECT row_to_json(v)::text FROM dataset_versions v"))
            .scalars()
            .all()
        )
    blob = "\n".join(dump)
    assert not any(len(cell) > 3 and cell in blob for cell in sample_cells)
    assert max(len(row) for row in dump) < 2000  # rows are small: metadata only


def test_the_real_demo_store_catalogs_when_present(migrated: Engine) -> None:
    root = ROOT / "data" / "store"
    if not root.is_dir():
        pytest.skip("NOT RUN: data/store has no demo datasets")
    store = FileDatasetStore(root)
    expected = sum(len(store.list_versions(n)) for n in store.list_datasets())
    before = tree_digest(root)
    result = sync_catalog(migrated, store)
    assert result.ok and len(result.added) == expected and expected > 0
    assert count(migrated, "dataset_versions") == expected
    assert tree_digest(root) == before


# --- the cataloged store -----------------------------------------------------------------------


def test_ingest_through_the_cataloged_store_writes_the_row_after_the_file(
    migrated: Engine, tmp_path: Path
) -> None:
    inner = FileDatasetStore(tmp_path / "store")
    store: DatasetStore = CatalogedDatasetStore(inner, migrated)
    first = store.ingest("gamma", toy_bytes(), source_filename="g.csv")
    assert first.created and count(migrated, "dataset_versions") == 1
    again = store.ingest("gamma", toy_bytes(), source_filename="g.csv")
    assert not again.created and count(migrated, "dataset_versions") == 1  # idempotent
    assert store.get_version("gamma@v1").sha256 == first.version.sha256
    assert store.version_path("gamma@v1").is_file()
    assert [r.ok for r in store.verify()] == [True]
    assert sync_catalog(migrated, inner).present == 1


def test_a_catalog_failure_does_not_lose_the_stored_version_and_sync_repairs_it(
    migrated: Engine, tmp_path: Path, url: URL
) -> None:
    inner = FileDatasetStore(tmp_path / "store")
    dead = create_engine(url.set(port=1), connect_args={"connect_timeout": 1})
    broken = CatalogedDatasetStore(inner, dead)
    result = broken.ingest("delta", toy_bytes(), source_filename="d.csv")
    assert result.created and inner.version_path("delta@v1").is_file()  # the bytes are safe
    assert broken.unsynced == ["delta@v1"] and count(migrated, "dataset_versions") == 0
    repaired = sync_catalog(migrated, inner)
    assert repaired.added == ("delta@v1",) and repaired.ok
    dead.dispose()


# --- the command line --------------------------------------------------------------------------


def run_cli(url: URL, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "AUTOML_DATABASE_URL": url.render_as_string(hide_password=False),
    }
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "db.py"), *args],
        capture_output=True, text=True, cwd=ROOT, env=env, check=False,
    )  # fmt: skip


def test_cli_status_upgrade_check_sync(url: URL, tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    make_store(store_root)
    status = run_cli(url, "status")
    assert status.returncode == 1 and "pending: 0001" in status.stdout
    assert run_cli(url, "check").returncode == 1
    assert run_cli(url, "sync", "--store", str(store_root)).returncode == 2  # not upgraded
    up = run_cli(url, "upgrade")
    assert up.returncode == 0 and "applied: 0001" in up.stdout
    again = run_cli(url, "upgrade")
    assert again.returncode == 0 and "already up to date" in again.stdout
    assert run_cli(url, "status").returncode == 0
    assert run_cli(url, "check").returncode == 0
    synced = run_cli(url, "sync", "--store", str(store_root), "--json")
    assert synced.returncode == 0
    payload = json.loads(synced.stdout)
    assert payload["added"] == ["alpha@v1", "alpha@v2", "beta@v1"] and payload["ok"]
    second = json.loads(run_cli(url, "sync", "--store", str(store_root), "--json").stdout)
    assert second["added"] == [] and second["present"] == 3
    password = url.password or "unreachable-sentinel"
    for result in (status, up, synced):
        assert password not in result.stdout + result.stderr
