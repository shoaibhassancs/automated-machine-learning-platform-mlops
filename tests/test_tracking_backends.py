"""P4-2b tests that need no PostgreSQL server: the artifact-URI pre-flight, backend configuration,
password handling, provenance tags, SQLite behaviour and the archive backup tool."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mlflow import MlflowClient

import automl_ops.tracking as tracking
from automl_ops.tracking import (
    DEFAULT_PG_ARTIFACT_ROOT,
    RUN_URI_LIMIT,
    RUN_URI_SUFFIX_LENGTH,
    ArtifactLocationError,
    TrackingError,
    TrackingStore,
    check_artifact_location,
    list_jobs,
    log_job,
    record_from_package,
    record_from_training,
)
from conftest import Run

ROOT = Path(__file__).resolve().parents[1]
MIGRATE = ROOT / "scripts" / "tracking_migrate.py"
CANARY = "leak-canary-9f3a"


@pytest.fixture
def canary_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", CANARY)
    monkeypatch.setenv("AUTOML_DB_PORT", "1")  # nothing listens here
    monkeypatch.delenv("AUTOML_DATABASE_URL", raising=False)


# --- the artifact URI pre-flight ---------------------------------------------------------------


def test_the_boundary_is_exactly_157_characters_of_location() -> None:
    longest_allowed = 157
    assert RUN_URI_LIMIT - RUN_URI_SUFFIX_LENGTH == longest_allowed
    fits = "file:///" + "a" * (longest_allowed - len("file:///"))
    assert len(fits) == longest_allowed
    assert check_artifact_location(fits) == RUN_URI_LIMIT
    assert check_artifact_location(fits + "/") == RUN_URI_LIMIT  # a trailing slash is not counted
    with pytest.raises(ArtifactLocationError, match="201") as error:
        check_artifact_location(fits + "a")
    assert "200" in str(error.value) and "AUTOML_MLFLOW_ARTIFACT_ROOT" in str(error.value)


def test_a_location_over_256_characters_is_refused() -> None:
    with pytest.raises(ArtifactLocationError):
        check_artifact_location("file:///" + "a" * 300)


def test_the_suffix_matches_the_run_uri_mlflow_really_builds(tmp_path: Path) -> None:
    """The formula the pre-flight relies on, checked against MLflow itself (SQLite backend)."""
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'x.db'}")
    location = (tmp_path / "art").as_uri()
    experiment = client.create_experiment("suffix", artifact_location=location)
    run = client.create_run(experiment)
    assert len(run.info.artifact_uri) == len(location) + RUN_URI_SUFFIX_LENGTH
    assert check_artifact_location(location) == len(run.info.artifact_uri)


def test_this_projects_default_location_fits_with_room_to_spare() -> None:
    longest = check_artifact_location(DEFAULT_PG_ARTIFACT_ROOT.resolve().as_uri())
    assert longest < RUN_URI_LIMIT


# --- configuration -----------------------------------------------------------------------------


def test_sqlite_is_the_default_and_unchanged(tmp_path: Path) -> None:
    store = TrackingStore.from_settings(tmp_path / "m", None, env={})
    assert (store.backend, store.artifact_root) == ("sqlite", None)
    assert store == TrackingStore(tmp_path / "m")
    assert store.tracking_uri == f"sqlite:///{(tmp_path / 'm').resolve().as_posix()}/mlflow.db"
    assert store.artifact_path == (tmp_path / "m").resolve() / "artifacts"
    assert store.describe() == str(store.db_path) and not store.exists()
    assert store.ui_environment() == {}
    command = store.ui_command(5123)
    assert command[:4] == ["mlflow", "server", "--backend-store-uri", store.tracking_uri]
    assert "--default-artifact-root" in command and command[-2:] == ["--port", "5123"]


def test_postgresql_must_be_selected_explicitly(tmp_path: Path) -> None:
    chosen = TrackingStore.from_settings(None, None, env={"AUTOML_MLFLOW_BACKEND": "postgresql"})
    assert chosen.backend == "postgresql"
    assert TrackingStore.from_settings(None, "sqlite", env={"AUTOML_MLFLOW_BACKEND": "postgresql"})
    assert TrackingStore.from_settings(None, "sqlite", env={}).backend == "sqlite"
    with pytest.raises(TrackingError, match="unknown tracking backend"):
        TrackingStore.from_settings(None, "mysql", env={})


def test_the_postgresql_artifact_root_default_override_and_absolute_rule(tmp_path: Path) -> None:
    default = TrackingStore.from_settings(None, "postgresql", env={})
    assert default.artifact_path == DEFAULT_PG_ARTIFACT_ROOT.resolve()
    assert default.artifact_path == (ROOT / "data" / "mlflow-pg" / "artifacts").resolve()
    env = {"AUTOML_MLFLOW_ARTIFACT_ROOT": str(tmp_path / "elsewhere")}
    assert (
        TrackingStore.from_settings(None, "postgresql", env=env).artifact_path
        == (tmp_path / "elsewhere").resolve()
    )
    with pytest.raises(TrackingError, match="absolute"):
        TrackingStore.from_settings(None, "postgresql", env={"AUTOML_MLFLOW_ARTIFACT_ROOT": "rel"})
    # a SQLite store ignores the override
    assert TrackingStore.from_settings(tmp_path, "sqlite", env=env).artifact_root is None


def test_the_database_name_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    store = TrackingStore(backend="postgresql")
    assert store.database_name == "mlflow"
    monkeypatch.setenv("AUTOML_MLFLOW_DB_NAME", 'x"; DROP DATABASE automl_ops; --')
    with pytest.raises(TrackingError, match="invalid MLflow database name"):
        _ = store.database_name


def test_the_password_is_in_the_uri_only_where_mlflow_needs_it(canary_env: None) -> None:
    store = TrackingStore(backend="postgresql")
    assert CANARY in store.tracking_uri  # the in-process URI handed to MLflow
    assert CANARY not in repr(store) and CANARY not in store.describe()
    assert "***" in store.describe()
    command = " ".join(store.ui_command())
    assert CANARY not in command and "postgresql" not in command  # nothing secret in argv
    environment = store.ui_environment()
    assert set(environment) == {"MLFLOW_BACKEND_STORE_URI", "MLFLOW_DEFAULT_ARTIFACT_ROOT"}
    assert CANARY in environment["MLFLOW_BACKEND_STORE_URI"]
    assert environment["MLFLOW_DEFAULT_ARTIFACT_ROOT"] == store.artifact_uri


def test_errors_never_contain_the_password(canary_env: None, clf_run: Run) -> None:
    store = TrackingStore(backend="postgresql")
    with pytest.raises(TrackingError, match="cannot reach") as read_error:
        store.exists()
    record = record_from_package(clf_run.root, clf_run.result.package_id)
    with pytest.raises(TrackingError) as write_error:
        log_job(record, store)
    with pytest.raises(TrackingError) as list_error:
        list_jobs(store)
    for error in (read_error, write_error, list_error):
        assert CANARY not in str(error.value) and CANARY not in repr(error.value)
        assert error.value.__cause__ is None  # no chained exception to leak through a traceback


def test_scrub_hides_the_password_and_its_url_encoded_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/word")
    text = "failed for p@ss/word and p%40ss%2Fword"
    assert tracking.scrub(text) == "failed for *** and ***"


def test_the_ui_script_hides_the_password_in_its_printed_command(tmp_path: Path) -> None:
    import os

    env = {**os.environ, "POSTGRES_PASSWORD": CANARY}
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "mlflow_ui.py")]
        + ["--backend", "postgresql", "--print-command"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert done.returncode == 0 and CANARY not in done.stdout + done.stderr
    assert "MLFLOW_BACKEND_STORE_URI=<set, not shown>" in done.stdout


# --- provenance tags ---------------------------------------------------------------------------


def test_extra_tags_are_added_and_default_to_none(clf_run: Run) -> None:
    plain = record_from_package(clf_run.root, clf_run.result.package_id)
    assert plain.extra_tags == {}
    assert "platform.origin" not in tracking._parent_tags(plain)
    tagged = record_from_package(
        clf_run.root, clf_run.result.package_id, extra_tags={"platform.origin": "x"}
    )
    assert tracking._parent_tags(tagged)["platform.origin"] == "x"
    base = {k: v for k, v in tracking._parent_tags(tagged).items() if k != "platform.origin"}
    assert base == tracking._parent_tags(plain)  # nothing else changes


def test_extra_tags_are_validated(clf_run: Run) -> None:
    for bad in ({"origin": "x"}, {"platform.origin": 1}):
        with pytest.raises(TrackingError, match="extra tags"):
            record_from_package(clf_run.root, clf_run.result.package_id, extra_tags=bad)  # type: ignore[arg-type]
    clash = record_from_package(
        clf_run.root, clf_run.result.package_id, extra_tags={"platform.kind": "train"}
    )
    with pytest.raises(TrackingError, match="would replace"):
        tracking._parent_tags(clash)
    training = clf_run.training.to_dict()
    record = record_from_training(
        training,
        {"ref": "toy@v1", "sha256": "0" * 64},
        {},
        extra_tags={"platform.migration": "P4-2b"},
    )
    assert record.extra_tags == {"platform.migration": "P4-2b"}  # fmt: skip


def test_sqlite_logging_with_tags_round_trips(clf_run: Run, tmp_path: Path) -> None:
    store = TrackingStore(tmp_path / "m")
    record = record_from_package(
        clf_run.root,
        clf_run.result.package_id,
        extra_tags={"platform.origin": "relogged-from-package"},
    )
    log_job(record, store)
    job = tracking.read_job(clf_run.result.package_id, store)
    assert job.tags["platform.origin"] == "relogged-from-package"
    again = log_job.__wrapped__ if hasattr(log_job, "__wrapped__") else None
    assert again is None  # (no hidden wrapper)
    with pytest.raises(tracking.AlreadyTrackedError):
        log_job(record, store)
    assert (
        store.exists() and tracking.run_ids(clf_run.result.package_id, store)["state"] == "complete"
    )


# --- the archive backup ------------------------------------------------------------------------


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_archive(tmp_path: Path, clf_run: Run) -> Path:
    """A SQLite archive made in a *separate process*, so this test process holds no handle on it."""
    archive = tmp_path / "archive"
    code = (
        "from pathlib import Path\n"
        "from automl_ops.tracking import TrackingStore, log_job, record_from_package\n"
        f"root, package = Path({str(clf_run.root)!r}), {clf_run.result.package_id!r}\n"
        "record = record_from_package(root, package)\n"
        f"log_job(record, TrackingStore(Path({str(archive)!r})))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
    return archive


def run_backup(archive: Path, dest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MIGRATE), "backup", "--archive", str(archive), "--dest", str(dest)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )  # fmt: skip


@pytest.mark.skipif(shutil.which("lsof") is None, reason="NOT RUN: lsof is not available")
def test_backup_is_consistent_verified_and_leaves_the_archive_untouched(
    tmp_path: Path, clf_run: Run
) -> None:
    archive = make_archive(tmp_path, clf_run)
    db = archive / "mlflow.db"
    before = (digest(db), db.stat().st_mtime_ns, db.stat().st_size)
    files_before = sorted(p.name for p in archive.iterdir())
    done = run_backup(archive, tmp_path / "backups")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (digest(db), db.stat().st_mtime_ns, db.stat().st_size) == before
    assert sorted(p.name for p in archive.iterdir()) == files_before  # no -wal/-shm/-journal made
    (backup,) = (tmp_path / "backups").iterdir()
    manifest = json.loads((backup / "MANIFEST.json").read_text())
    assert manifest["problems"] == [] and manifest["backup"]["integrity_check"] == "ok"
    assert manifest["archive"]["db_sha256"] == before[0]
    # the backup is itself a usable store with the same history
    original = [j.to_dict() for j in list_jobs(TrackingStore(archive))]
    copied = [j.to_dict() for j in list_jobs(TrackingStore(backup))]
    assert copied == original and len(original) == 1
    # a backup never overwrites (the same second is refused, a later one is a new folder)
    again = run_backup(archive, tmp_path / "backups")
    assert again.returncode in (0, 2)


@pytest.mark.skipif(shutil.which("lsof") is None, reason="NOT RUN: lsof is not available")
def test_backup_refuses_a_wal_or_journal_file_and_creates_nothing(
    tmp_path: Path, clf_run: Run
) -> None:
    archive = make_archive(tmp_path, clf_run)
    (archive / "mlflow.db-wal").write_bytes(b"")
    done = run_backup(archive, tmp_path / "backups")
    assert done.returncode == 2 and "WAL or journal" in done.stderr
    assert not (tmp_path / "backups").exists()


@pytest.mark.skipif(shutil.which("lsof") is None, reason="NOT RUN: lsof is not available")
def test_backup_refuses_while_a_process_has_the_database_open(tmp_path: Path, clf_run: Run) -> None:
    archive = make_archive(tmp_path, clf_run)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import time; f = open({str(archive / 'mlflow.db')!r}, 'rb'); time.sleep(60)",
        ]
    )
    try:
        time.sleep(1.0)
        done = run_backup(archive, tmp_path / "backups")
        assert done.returncode == 2 and "open" in done.stderr
        assert not (tmp_path / "backups").exists()
    finally:
        holder.kill()
        holder.wait()


def test_backup_refuses_something_that_is_not_a_store(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    done = run_backup(tmp_path / "empty", tmp_path / "b")
    assert done.returncode == 2 and "not a tracking store" in done.stderr


def test_plan_lists_the_archive_jobs_without_writing(tmp_path: Path, clf_run: Run) -> None:
    archive = make_archive(tmp_path, clf_run)
    before = digest(archive / "mlflow.db")
    done = subprocess.run(
        [sys.executable, str(MIGRATE), "plan", "--archive", str(archive)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )  # fmt: skip
    assert done.returncode == 0 and clf_run.result.package_id in done.stdout
    assert "relog from sealed package, same key" in done.stdout
    assert digest(archive / "mlflow.db") == before
