"""P4-2b tests against a real PostgreSQL server (`make db-up`): MLflow's own database, the artifact
URI limit, explicit initialisation, the tracking behaviour on both backends, the UI and the
migration script.

Each test gets its own throwaway MLflow database, cloned from a template that was initialised once
(creating MLflow's 71 tables takes about 2 s). Only databases created here are ever dropped. If the
server cannot be reached the PostgreSQL cases are *skipped* and pytest says
`NOT RUN: PostgreSQL not reachable ...`; a skip is never a pass.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from mlflow import MlflowClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL

import automl_ops.tracking as tracking
from automl_ops.compare import compare_jobs
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db.config import DatabaseConfigError, database_url
from automl_ops.finalize import DatasetInfo, finalize
from automl_ops.package import library_versions
from automl_ops.profile import read_version_frame
from automl_ops.reproduce import reproduce_job, sealed_manifest_selection_sha, sealed_test_result
from automl_ops.selection import freeze_selection, select_model
from automl_ops.tracking import (
    AlreadyTrackedError,
    ArtifactLocationError,
    JobNotFoundError,
    TrackingError,
    TrackingStore,
    initialise_store,
    list_jobs,
    log_job,
    read_job,
    record_from_package,
    record_from_training,
    run_ids,
    store_status,
)
from automl_ops.train import run_training
from conftest import CLF, FAST, SETTINGS, Run, toy_bytes

ROOT = Path(__file__).resolve().parents[1]
MIGRATE = ROOT / "scripts" / "tracking_migrate.py"


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


def _drop(url: URL, name: str) -> None:
    admin = _admin(url)
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()


@pytest.fixture(scope="module")
def mlflow_template(server: URL, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A database with MLflow's schema and no experiment, initialised once."""
    name = f"automl_test_mlt_{secrets.token_hex(5)}"
    patch = pytest.MonkeyPatch()
    patch.setenv("AUTOML_MLFLOW_DB_NAME", name)
    patch.setenv("AUTOML_MLFLOW_ARTIFACT_ROOT", str(tmp_path_factory.mktemp("tpl") / "a"))
    try:
        initialise_store(TrackingStore.from_settings(backend="postgresql"), create_experiment=False)
    finally:
        patch.undo()
    yield name
    _drop(server, name)


@pytest.fixture
def pg_clone(
    server: URL, mlflow_template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TrackingStore]:
    """A fresh MLflow database (schema, no experiment) and a short artifact root in `tmp_path`."""
    name = f"automl_test_ml_{secrets.token_hex(5)}"
    admin = _admin(server)
    try:
        with admin.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": mlflow_template},
            )
            connection.execute(text(f'CREATE DATABASE "{name}" TEMPLATE "{mlflow_template}"'))
        monkeypatch.setenv("AUTOML_MLFLOW_DB_NAME", name)
        monkeypatch.setenv("AUTOML_MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
        yield TrackingStore.from_settings(backend="postgresql")
    finally:
        admin.dispose()
        _drop(server, name)


@pytest.fixture
def pg_store(pg_clone: TrackingStore) -> TrackingStore:
    initialise_store(pg_clone)
    return pg_clone


def sql(store: TrackingStore, statement: str) -> Any:
    engine = create_engine(database_url().set(database=store.database_name))
    try:
        with engine.connect() as connection:
            return connection.execute(text(statement)).scalar_one()
    finally:
        engine.dispose()


def row_counts(store: TrackingStore) -> dict[str, int]:
    engine = create_engine(database_url().set(database=store.database_name))
    try:
        with engine.connect() as connection:
            names = [
                r[0]
                for r in connection.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                )
            ]
            return {
                n: int(connection.execute(text(f'SELECT count(*) FROM "{n}"')).scalar_one())
                for n in sorted(names)
            }
    finally:
        engine.dispose()


def root_with_uri_length(base: Path, length: int) -> Path:
    """A directory under `base` whose file URI is exactly `length` characters long."""
    probe = (base / "x").resolve().as_uri()
    root = base / ("x" * (1 + length - len(probe)))
    assert len(root.resolve().as_uri()) == length
    return root


def record_of(run: Run) -> tracking.JobRecord:
    return record_from_package(run.root, run.result.package_id)


# --- explicit initialisation: nothing else may create the store --------------------------------


@pytest.mark.postgres
def test_init_is_idempotent_and_creates_the_experiment(pg_clone: TrackingStore) -> None:
    first = initialise_store(pg_clone)
    assert first["initialised"] and first["experiment_exists"] and first["runs"] == 0
    assert first["location_matches_configuration"] and first["run_uri_margin"] >= 0
    second = initialise_store(pg_clone)
    assert second["database_created"] is False
    assert second["experiment_artifact_location"] == first["experiment_artifact_location"]
    assert sql(pg_clone, "SELECT count(*) FROM experiments WHERE name = 'automl-ops'") == 1


@pytest.mark.postgres
def test_init_creates_a_missing_database(
    server: URL, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = f"automl_test_ml_{secrets.token_hex(5)}"
    monkeypatch.setenv("AUTOML_MLFLOW_DB_NAME", name)
    monkeypatch.setenv("AUTOML_MLFLOW_ARTIFACT_ROOT", str(tmp_path / "a"))
    store = TrackingStore.from_settings(backend="postgresql")
    try:
        assert not store.exists()
        info = initialise_store(store)
        assert info["database_created"] is True and info["initialised"] and store.exists()
    finally:
        _drop(server, name)


@pytest.mark.postgres
def test_reads_and_status_never_create_a_database(
    server: URL, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clf_run: Run
) -> None:
    name = f"automl_test_ml_missing_{secrets.token_hex(5)}"
    monkeypatch.setenv("AUTOML_MLFLOW_DB_NAME", name)
    monkeypatch.setenv("AUTOML_MLFLOW_ARTIFACT_ROOT", str(tmp_path / "a"))
    store = TrackingStore.from_settings(backend="postgresql")
    assert not store.exists()
    assert list_jobs(store) == []  # the documented contract: a missing store is an empty history
    with pytest.raises(JobNotFoundError):
        read_job("m-none", store)
    assert store_status(store)["initialised"] is False
    with pytest.raises(TrackingError, match="mlflow-init"):
        log_job(record_of(clf_run), store)  # a write does not create it either
    admin = _admin(server)
    try:
        with admin.connect() as connection:
            found = connection.execute(
                text("SELECT count(*) FROM pg_database WHERE datname = :n"), {"n": name}
            ).scalar_one()
    finally:
        admin.dispose()
    assert found == 0 and not (tmp_path / "a").exists()


@pytest.mark.postgres
def test_an_empty_database_gets_no_schema_from_reads_or_writes(
    server: URL, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clf_run: Run
) -> None:
    name = f"automl_test_ml_empty_{secrets.token_hex(5)}"
    admin = _admin(server)
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    try:
        monkeypatch.setenv("AUTOML_MLFLOW_DB_NAME", name)
        monkeypatch.setenv("AUTOML_MLFLOW_ARTIFACT_ROOT", str(tmp_path / "a"))
        store = TrackingStore.from_settings(backend="postgresql")
        assert list_jobs(store) == []
        with pytest.raises(JobNotFoundError):
            read_job("m-none", store)
        with pytest.raises(TrackingError, match="not initialised"):
            log_job(record_of(clf_run), store)
        assert not store.exists() and store_status(store)["database_exists"] is True
        assert (
            sql(store, "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")
            == 0
        )
    finally:
        _drop(server, name)


# --- the artifact URI limit --------------------------------------------------------------------


@pytest.mark.postgres
def test_an_overlong_location_is_refused_before_anything_is_created(
    pg_clone: TrackingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clf_run: Run
) -> None:
    monkeypatch.setenv("AUTOML_MLFLOW_ARTIFACT_ROOT", str(root_with_uri_length(tmp_path, 158)))
    store = TrackingStore.from_settings(backend="postgresql")
    with pytest.raises(ArtifactLocationError, match="201"):
        initialise_store(store)
    with pytest.raises(ArtifactLocationError):
        log_job(record_of(clf_run), store)
    status = store_status(store)
    assert status["configured_location_ok"] is False and status["experiment_exists"] is False
    assert sql(store, "SELECT count(*) FROM runs") == 0
    assert sql(store, "SELECT count(*) FROM experiments WHERE name = 'automl-ops'") == 0


@pytest.mark.postgres
def test_a_location_at_the_limit_works_end_to_end(
    pg_clone: TrackingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clf_run: Run
) -> None:
    monkeypatch.setenv("AUTOML_MLFLOW_ARTIFACT_ROOT", str(root_with_uri_length(tmp_path, 157)))
    store = TrackingStore.from_settings(backend="postgresql")
    assert initialise_store(store)["configured_location_longest_run_uri"] == 200
    log_job(record_of(clf_run), store)
    assert sql(store, "SELECT max(length(artifact_uri)) FROM runs") == 200
    assert store_status(store)["run_uri_margin"] == 0
    assert read_job(clf_run.result.package_id, store).state == tracking.COMPLETE


@pytest.mark.postgres
def test_postgresql_itself_rejects_a_201_character_uri(
    pg_store: TrackingStore, tmp_path: Path
) -> None:
    """Why the pre-flight exists: MLflow does not check, PostgreSQL does."""
    location = root_with_uri_length(tmp_path, 158).as_uri()
    client = MlflowClient(tracking_uri=pg_store.tracking_uri)
    experiment = client.create_experiment("over-the-limit", artifact_location=location)
    with pytest.raises(Exception, match="too long"):
        client.create_run(experiment)


@pytest.mark.postgres
def test_a_changed_location_blocks_writes_but_not_reads(
    pg_store: TrackingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clf_run: Run
) -> None:
    log_job(record_of(clf_run), pg_store)
    monkeypatch.setenv("AUTOML_MLFLOW_ARTIFACT_ROOT", str(tmp_path / "moved"))
    moved = TrackingStore.from_settings(backend="postgresql")
    with pytest.raises(ArtifactLocationError, match="records the artifact location"):
        log_job(record_of(clf_run), moved)
    assert [j.job_key for j in list_jobs(moved)] == [clf_run.result.package_id]  # reads still work
    assert store_status(moved)["location_matches_configuration"] is False


# --- the tracking behaviour on both backends ---------------------------------------------------


@pytest.fixture(params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgres)])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> TrackingStore:
    if request.param == "sqlite":
        return TrackingStore(tmp_path / "m")
    return request.getfixturevalue("pg_store")  # type: ignore[no-any-return]


def test_log_read_and_never_twice(store: TrackingStore, clf_run: Run) -> None:
    record = record_of(clf_run)
    log_job(record, store)
    job = read_job(record.job_key, store)
    assert job.state == tracking.COMPLETE and job.kind == "final"
    assert set(job.algorithms) == {r.key for r in clf_run.training.results}
    chosen = job.tags["platform.selected_algorithm"]
    assert job.algorithms[chosen].test_result is not None
    with pytest.raises(AlreadyTrackedError):
        log_job(record, store)
    assert [j.job_key for j in list_jobs(store)] == [record.job_key]


def test_resume_after_an_injected_failure(
    store: TrackingStore, clf_run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = record_of(clf_run)
    original = tracking._log_child
    calls = {"n": 0}

    def failing(*args: Any, **kwargs: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure")
        return str(original(*args, **kwargs))

    monkeypatch.setattr(tracking, "_log_child", failing)
    with pytest.raises(TrackingError):
        log_job(record, store)
    monkeypatch.setattr(tracking, "_log_child", original)
    assert run_ids(record.job_key, store)["state"] == tracking.INCOMPLETE
    resumed = log_job(record, store)
    assert resumed.resumed
    ids = run_ids(record.job_key, store)
    assert ids["state"] == tracking.COMPLETE
    assert len(ids["children"]) == len(clf_run.training.results)  # no duplicates
    assert len(list_jobs(store)) == 1


def test_history_comparison_and_reproduction(store: TrackingStore, clf_run: Run) -> None:
    final = record_of(clf_run)
    cv_only = record_from_training(
        clf_run.training.to_dict(), clf_run.dataset.to_dict(), library_versions()
    )
    log_job(final, store)
    log_job(cv_only, store)
    assert {j.kind for j in list_jobs(store)} == {"final", "train"}
    jobs = [read_job(final.job_key, store), read_job(cv_only.job_key, store)]
    result = compare_jobs(jobs)
    assert result["rows"] and all(r["cv_mean"] is not None for r in result["rows"])
    assert "no test scores" in result["basis"]
    pid = clf_run.result.package_id
    report = reproduce_job(
        jobs[0],
        clf_run.frame,
        clf_run.dataset.sha256,
        library_versions(),
        package_test_result=sealed_test_result(clf_run.root, pid),
        package_selection_sha256=sealed_manifest_selection_sha(clf_run.root, pid),
        package_checked=True,
    )
    assert report.ok and report.bit_identical


def fingerprint(store: TrackingStore) -> Any:
    if store.backend == "sqlite":
        return hashlib.sha256(store.db_path.read_bytes()).hexdigest()
    return row_counts(store)


def test_read_only_operations_write_nothing(store: TrackingStore, clf_run: Run) -> None:
    record = record_of(clf_run)
    log_job(record, store)
    before = fingerprint(store)
    list_jobs(store)
    job = read_job(record.job_key, store)
    tracking.run_ids(record.job_key, store)
    tracking.store_status(store)
    reproduce_job(
        job, clf_run.frame, clf_run.dataset.sha256, library_versions(),
        package_test_result=sealed_test_result(clf_run.root, record.job_key),
        package_selection_sha256=sealed_manifest_selection_sha(clf_run.root, record.job_key),
        package_checked=True,
    )  # fmt: skip
    assert fingerprint(store) == before


# --- concurrency, secrets and the UI -----------------------------------------------------------


@pytest.mark.postgres
def test_two_writers_and_a_reader_see_no_partial_job(
    pg_store: TrackingStore, clf_run: Run, reg_run: Run
) -> None:
    stop = threading.Event()
    anomalies: list[str] = []
    failures: list[BaseException] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                for summary in list_jobs(pg_store):
                    if summary.state == tracking.COMPLETE:
                        job = read_job(summary.job_key, pg_store)
                        if len(job.algorithms) != summary.algorithms_ok + summary.algorithms_failed:
                            anomalies.append(f"partial read of {summary.job_key}")
            except Exception as error:
                anomalies.append(repr(error)[:120])
            time.sleep(0.02)

    def writer(run: Run) -> None:
        try:
            log_job(record_of(run), pg_store)
        except BaseException as error:
            failures.append(error)

    watcher = threading.Thread(target=reader)
    watcher.start()
    writers = [threading.Thread(target=writer, args=(run,)) for run in (clf_run, reg_run)]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    watcher.join()
    assert failures == [] and anomalies == []
    keys = {j.job_key for j in list_jobs(pg_store, state=tracking.COMPLETE)}
    assert keys == {clf_run.result.package_id, reg_run.result.package_id}
    assert sql(pg_store, "SELECT count(*) FROM runs WHERE name LIKE 'job %'") == 2


@pytest.mark.postgres
def test_a_wrong_password_error_does_not_contain_the_password(
    server: URL, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "wrong-canary-xyz-81")
    with pytest.raises(TrackingError) as error:
        TrackingStore(backend="postgresql").exists()
    assert "wrong-canary-xyz-81" not in str(error.value)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.postgres
def test_the_ui_serves_the_postgresql_store_without_the_password_in_its_arguments(
    pg_store: TrackingStore, clf_run: Run
) -> None:
    log_job(record_of(clf_run), pg_store)
    port = _free_port()
    env = {**os.environ, **tracking.quiet_environment(), **pg_store.ui_environment()}
    command = [sys.executable, "-m", "mlflow", *pg_store.ui_command(port)[1:]]
    server = subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        up = False
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                up = True
                break
            except Exception:
                time.sleep(1)
        assert up, "the MLflow UI did not start"
        shown = subprocess.run(
            ["ps", "-o", "args=", "-p", str(server.pid)], capture_output=True, text=True
        ).stdout
        secret = database_url().password or "unreachable-sentinel"
        assert secret not in shown and "postgresql" not in shown
        experiment = tracking._Api(pg_store, read_only=True).experiment_id
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/2.0/mlflow/runs/search",
            data=json.dumps({"experiment_ids": [experiment], "max_results": 100}).encode(),
            headers={"Content-Type": "application/json"},
        )
        served = json.loads(urllib.request.urlopen(request, timeout=20).read())["runs"]
        assert len(served) == 1 + len(clf_run.training.results)
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()


# --- the migration script ----------------------------------------------------------------------


def load_migrate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tracking_migrate_under_test", MIGRATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class World:
    def __init__(self, tmp_path: Path) -> None:
        self.store_root = tmp_path / "datasets"
        self.models = tmp_path / "models"
        self.archive = tmp_path / "archive"
        store = FileDatasetStore(self.store_root)
        store.ingest("toy", toy_bytes(), source_filename="toy.csv")
        version, frame = read_version_frame(store, "toy@v1")
        info = DatasetInfo.from_version(version)
        training = run_training(frame, CLF, FAST, SETTINGS)
        frozen = freeze_selection(training, select_model(training))
        result = finalize(frame, CLF, training, frozen, info, self.models, confirm_final_test=True)
        self.final_key = result.package_id
        archive = TrackingStore(self.archive)
        log_job(record_from_package(self.models, self.final_key), archive)
        other = run_training(frame, CLF, FAST, SETTINGS.__class__(folds=3, seed=7))
        cv_only = record_from_training(other.to_dict(), info.to_dict(), library_versions())
        log_job(cv_only, archive)
        self.cv_key = cv_only.job_key

    def common(self) -> list[str]:
        return ["--archive", str(self.archive)]


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path / "w")


def mapping_of(target: TrackingStore) -> list[dict[str, Any]]:
    path = target.artifact_path.parent / "migration" / "mapping.json"
    return list(json.loads(path.read_text()))


@pytest.mark.postgres
def test_relog_rerun_and_verify_then_run_again(
    world: World, pg_store: TrackingStore, capsys: pytest.CaptureFixture[str]
) -> None:
    migrate = load_migrate()
    relog = ["relog", *world.common(), "--models", str(world.models)]
    rerun = ["rerun", *world.common(), "--store", str(world.store_root)]
    assert migrate.main(relog) == 0 and migrate.main(rerun) == 0
    mapping = {e["original_job_key"]: e for e in mapping_of(pg_store)}
    assert mapping[world.final_key]["new_job_key"] == world.final_key  # same key, from the package
    assert mapping[world.final_key]["origin"] == "relogged-from-package"
    new_cv = mapping[world.cv_key]["new_job_key"]
    assert new_cv != world.cv_key and new_cv.startswith("t-")
    assert mapping[world.cv_key]["origin"] == "rerun-from-archive"
    assert (
        mapping[world.final_key]["original_run_ids"]["parent"]
        != mapping[world.final_key]["new_run_ids"]["parent"]
    )
    new_final, new_rerun = read_job(world.final_key, pg_store), read_job(new_cv, pg_store)
    assert new_final.tags["platform.source_store"] == "sqlite-archive:data/mlflow"
    assert new_rerun.tags["platform.replaces_job_key"] == world.cv_key
    assert {new_final.tags["platform.migration"], new_rerun.tags["platform.migration"]} == {"P4-2b"}
    archive = TrackingStore(world.archive)  # the originals carry none of the new tags
    for key in (world.final_key, world.cv_key):
        assert not {t for t in read_job(key, archive).tags if t in tracking_tags()}
    before = row_counts(pg_store)
    capsys.readouterr()
    verify = [
        "verify",
        *world.common(),
        "--models",
        str(world.models),
        "--store",
        str(world.store_root),
    ]
    assert migrate.main(verify) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and out.count("[PASS]") >= 8
    assert row_counts(pg_store) == before  # verification wrote nothing
    # a second pass is a no-op: nothing is logged twice, the same keys are reused
    assert migrate.main(relog) == 0 and migrate.main(rerun) == 0
    assert row_counts(pg_store) == before
    assert {e["original_job_key"]: e["new_job_key"] for e in mapping_of(pg_store)} == {
        k: v["new_job_key"] for k, v in mapping.items()
    }


def tracking_tags() -> set[str]:
    return {
        "platform.origin",
        "platform.source_store",
        "platform.migration",
        "platform.replaces_job_key",
    }


@pytest.mark.postgres
def test_an_interrupted_rerun_resumes_the_same_job(
    world: World, pg_store: TrackingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrate = load_migrate()
    rerun = ["rerun", *world.common(), "--store", str(world.store_root)]
    original = tracking._log_child
    calls = {"n": 0}

    def failing(*args: Any, **kwargs: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure")
        return str(original(*args, **kwargs))

    monkeypatch.setattr(tracking, "_log_child", failing)
    assert migrate.main(rerun) == 2  # the failure is reported, nothing is hidden
    saved = json.loads(
        next((pg_store.artifact_path.parent / "migration").glob("rerun-*.json")).read_text()
    )
    monkeypatch.setattr(tracking, "_log_child", original)
    assert run_ids(saved["job_key"], pg_store)["state"] == tracking.INCOMPLETE
    assert migrate.main(rerun) == 0  # resumes the same job: same key, the saved result is reused
    ids = run_ids(saved["job_key"], pg_store)
    assert ids["state"] == tracking.COMPLETE and len(ids["children"]) == len(
        saved["training"]["results"]
    )
    assert sql(pg_store, "SELECT count(*) FROM runs WHERE name LIKE 'job t-%'") == 1
    assert read_job(saved["job_key"], pg_store).tags["platform.replaces_job_key"] == world.cv_key
