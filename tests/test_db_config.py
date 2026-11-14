"""P4-1 tests that need no PostgreSQL server: settings, migration scripts, models, isolation."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory

from automl_ops.db import migrate
from automl_ops.db.config import (
    DatabaseConfigError,
    database_url,
    describe,
    read_env_file,
)
from automl_ops.db.models import Base

ROOT = Path(__file__).resolve().parents[1]
NO_FILE = Path("/nonexistent/.env")


def test_default_url_is_local_non_default_port_with_psycopg3() -> None:
    url = database_url({"POSTGRES_PASSWORD": "pw"}, NO_FILE)
    assert (url.drivername, url.host, url.port) == ("postgresql+psycopg", "127.0.0.1", 55432)
    assert (url.username, url.database) == ("automl", "automl_ops")


def test_describe_hides_the_password() -> None:
    url = database_url({"POSTGRES_PASSWORD": "s3cret-value"}, NO_FILE)
    assert "s3cret-value" not in describe(url)
    assert "***" in describe(url)


def test_a_missing_password_is_a_clear_error() -> None:
    with pytest.raises(DatabaseConfigError, match="POSTGRES_PASSWORD"):
        database_url({}, NO_FILE)


def test_the_env_file_supplies_the_password_and_the_environment_wins(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\n\nPOSTGRES_PASSWORD='from-file'\nOTHER=1\n")
    assert read_env_file(env) == {"POSTGRES_PASSWORD": "from-file", "OTHER": "1"}
    assert database_url({}, env).password == "from-file"
    assert database_url({"POSTGRES_PASSWORD": "from-env"}, env).password == "from-env"
    assert read_env_file(tmp_path / "missing") == {}


def test_an_explicit_url_is_used_but_must_be_postgresql() -> None:
    url = database_url({"AUTOML_DATABASE_URL": "postgresql://u:p@db.local:1234/x"}, NO_FILE)
    assert (url.drivername, url.host, url.port, url.database) == (
        "postgresql+psycopg",
        "db.local",
        1234,
        "x",
    )
    with pytest.raises(DatabaseConfigError, match="postgresql"):
        database_url({"AUTOML_DATABASE_URL": "sqlite:///x.db"}, NO_FILE)
    with pytest.raises(DatabaseConfigError):
        database_url({"AUTOML_DATABASE_URL": "not a url"}, NO_FILE)
    with pytest.raises(DatabaseConfigError, match="PORT"):
        database_url({"POSTGRES_PASSWORD": "p", "AUTOML_DB_PORT": "x"}, NO_FILE)


def test_there_is_one_migration_head_and_a_linear_history() -> None:
    scripts = ScriptDirectory(str(migrate.MIGRATIONS))
    assert scripts.get_heads() == ["0007"] == [migrate.head_revision()]
    revisions = list(scripts.walk_revisions())
    assert [r.revision for r in revisions] == [
        "0007",
        "0006",
        "0005",
        "0004",
        "0003",
        "0002",
        "0001",
    ]
    assert revisions[0].down_revision == "0006"
    assert revisions[1].down_revision == "0005"
    assert revisions[2].down_revision == "0004"
    assert revisions[3].down_revision == "0003"
    assert revisions[4].down_revision == "0002"
    assert revisions[5].down_revision == "0001"
    assert revisions[6].down_revision is None


def test_the_models_declare_exactly_the_catalog_job_and_registry_tables() -> None:
    tables = {t.name: {c.name for c in t.columns} for t in Base.metadata.sorted_tables}
    assert set(tables) == {
        "datasets",
        "dataset_versions",
        "training_jobs",
        "job_events",
        "registered_models",
        "model_versions",
        "registry_events",
        "registry_sync",
        "registry_sync_log",
        "deployments",
        "predictions",
        "feedback",
        "reference_samples",
        "monitoring_reports",
        "alerts",
        "alert_events",
    }
    assert tables["datasets"] == {"id", "name", "created_at"}
    # identity, shape, column *names* and provenance; no column can hold cell values or file bytes
    assert tables["dataset_versions"] == {
        "id", "dataset_id", "number", "sha256", "original_sha256", "size_bytes",
        "original_size_bytes", "rows", "column_names", "source_filename", "original_encoding",
        "original_delimiter", "reencoded", "trailing_blank_lines_ignored", "store_format",
        "created_at", "catalogued_at",
    }  # fmt: skip


def test_constraint_names_follow_the_convention() -> None:
    names = {
        str(c.name)
        for t in Base.metadata.sorted_tables
        for c in t.constraints
        if c.name is not None
    }
    assert "ck_datasets_name_format" in names
    assert "uq_dataset_versions_dataset_sha256" in names
    assert "ck_dataset_versions_sha256_format" in names


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_the_db_package_imports_no_mlflow_and_no_model_or_test_code() -> None:
    for path in (ROOT / "src" / "automl_ops" / "db").rglob("*.py"):
        modules = _imports(path)
        assert not {m for m in modules if m.split(".")[0] == "mlflow"}, path
        assert not modules & {"automl_ops.finalize", "automl_ops.package", "joblib"}, path


def test_dependencies_are_declared_directly() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    for name in ("psycopg[binary]", "sqlalchemy", "alembic"):
        assert f'"{name}>=' in text


def test_the_cli_reports_an_unreachable_server_without_leaking_the_password() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "POSTGRES_PASSWORD": "leak-canary-4711",
        "AUTOML_DB_PORT": "1",  # nothing listens here
    }
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "db.py"), "status"],
        capture_output=True, text=True, cwd=ROOT, env=env, check=False,
    )  # fmt: skip
    assert done.returncode == 2
    assert "cannot use the database" in done.stderr
    assert "leak-canary-4711" not in done.stdout + done.stderr
