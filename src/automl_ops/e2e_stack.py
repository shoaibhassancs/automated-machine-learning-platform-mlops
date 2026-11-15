"""A throwaway stack for the end-to-end story, owned by one run (follow-up to P4-10).

`IsolatedStack` creates everything the story may write to, starts the three processes on it, and
removes only what it created:

* a new PostgreSQL database `automl_test_e2e_<random>`, migrated, and stamped with a comment that
  names this run (the real `automl_ops` and `mlflow` databases are never connected to for anything
  but listing nothing: the only other connection is to the maintenance database `postgres` to
  create and drop *this* database);
* a new temporary folder (outside the project) holding the dataset store, the model packages and a
  SQLite MLflow store, with an owner marker file;
* the API, serving and worker processes, started by this object on free local ports with an
  environment built from scratch (no inherited `AUTOML_*`, `MLFLOW_*` or `POSTGRES_*` variable).

Cleanup first proves ownership of each resource (database comment, marker file, location, no
symbolic link). If any proof fails, nothing is deleted and the remaining resources are reported.
There is no force option.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from automl_ops.db import migrate
from automl_ops.db.config import database_url, make_engine
from automl_ops.isolation import (
    PROTECTED_DATABASES,
    THROWAWAY_DB_PREFIX,
    TOKEN_SETTING,
    inside,
    run_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLDER_PREFIX = "automl-e2e-"
MARKER = ".automl-e2e-owner"
COMMENT_PREFIX = "automl-e2e-owner:"
INHERITED_PREFIXES = ("AUTOML_", "MLFLOW_", "POSTGRES_")


class StackError(RuntimeError):
    """The stack could not be created or started."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def clean_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """The current environment without anything that could point a child at the real stack."""
    return {
        k: v
        for k, v in (os.environ if source is None else source).items()
        if not k.startswith(INHERITED_PREFIXES)
    }


@dataclass
class IsolatedStack:
    base: URL | None = None  # the server to create the database on (default: the configured one)
    parent: Path | None = None  # where the temporary folder goes (default: the system temp dir)
    token: str = field(default_factory=lambda: secrets.token_hex(16))
    database: str = ""
    root: Path | None = None
    api_port: int = 0
    serve_port: int = 0
    ui_origin: str | None = None  # the browser origin allowed by CORS (the UI's address)
    processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)

    # -- creation ---------------------------------------------------------------------------

    @property
    def database_url(self) -> URL:
        assert self.base is not None and self.database
        return self.base.set(database=self.database)

    def create(self) -> None:
        self.base = self.base or database_url()
        parent = Path(self.parent or tempfile.gettempdir()).resolve()
        if inside(parent, PROJECT_ROOT):
            raise StackError("the temporary folder must be outside the project")
        self.root = Path(tempfile.mkdtemp(prefix=FOLDER_PREFIX, dir=parent)).resolve()
        (self.root / MARKER).write_text(run_id(self.token), encoding="utf-8")
        for name in ("store", "models", "mlflow"):
            (self.root / name).mkdir()
        self.database = f"{THROWAWAY_DB_PREFIX}{secrets.token_hex(4)}"
        admin = create_engine(self.base.set(database="postgres"), isolation_level="AUTOCOMMIT")
        try:
            with admin.connect() as c:
                c.execute(text(f'CREATE DATABASE "{self.database}"'))
                # COMMENT takes no bind parameters; the text is the prefix plus 16 hex digits
                stamp = COMMENT_PREFIX + run_id(self.token)
                assert re.fullmatch(r"[a-z0-9:-]+", stamp)
                c.execute(text(f"COMMENT ON DATABASE \"{self.database}\" IS '{stamp}'"))
        finally:
            admin.dispose()
        engine = make_engine(self.database_url)
        try:
            if engine.url.database != self.database:
                raise StackError("refusing to migrate a database this run did not create")
            migrate.upgrade(engine)
        finally:
            engine.dispose()

    def environment(self) -> dict[str, str]:
        assert self.root is not None
        return {
            **clean_environment(),
            "AUTOML_DATABASE_URL": self.database_url.render_as_string(hide_password=False),
            "AUTOML_STORE_DIR": str(self.root / "store"),
            "AUTOML_MODELS_DIR": str(self.root / "models"),
            "AUTOML_JOBS_MLFLOW_DIR": str(self.root / "mlflow"),
            "AUTOML_MLFLOW_BACKEND": "sqlite",
            "AUTOML_REGISTRY_MIRROR": "off",
            "AUTOML_API_PORT": str(self.api_port),
            "AUTOML_SERVING_PORT": str(self.serve_port),
            TOKEN_SETTING: self.token,
            **({"AUTOML_UI_ORIGIN": self.ui_origin} if self.ui_origin else {}),
        }

    # -- processes --------------------------------------------------------------------------

    def start(self, wait: float = 90.0) -> None:
        assert self.root is not None
        self.api_port = self.api_port or free_port()
        self.serve_port = self.serve_port or free_port()
        while self.serve_port == self.api_port:
            self.serve_port = free_port()
        env = self.environment()
        for key, script in (
            ("api", "serve_control.py"),
            ("serve", "serve_model.py"),
            ("worker", "worker.py"),
        ):
            log = open(self.root / f"{key}.log", "wb")  # noqa: SIM115 (closed with the process)
            self.processes[key] = subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "scripts" / script)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
        deadline = time.monotonic() + wait
        for key, port in (("api", self.api_port), ("serve", self.serve_port)):
            while True:
                if self.processes[key].poll() is not None:
                    raise StackError(f"the {key} process exited while starting")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise StackError(f"the {key} process did not become healthy") from None
                    time.sleep(0.5)

    def stop(self) -> None:
        """Stop only the processes this object started."""
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            try:
                process.wait(20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)
        self.processes.clear()

    # -- cleanup ----------------------------------------------------------------------------

    def ownership_problems(self) -> list[str]:
        """Why a resource cannot be proved to belong to this run (empty: all proved)."""
        problems: list[str] = []
        if self.root is not None:
            marker = self.root / MARKER
            if self.root.is_symlink() or not self.root.is_dir():
                problems.append("the temporary folder is missing or is a link")
            elif not self.root.name.startswith(FOLDER_PREFIX) or inside(self.root, PROJECT_ROOT):
                problems.append("the temporary folder is not where this run created it")
            elif marker.is_symlink() or not marker.is_file():
                problems.append("the temporary folder has no owner marker")
            elif marker.read_text(encoding="utf-8").strip() != run_id(self.token):
                problems.append("the temporary folder belongs to another run")
        if self.database:
            if self.database in PROTECTED_DATABASES or not self.database.startswith(
                THROWAWAY_DB_PREFIX
            ):
                problems.append("the database name is not a throwaway name")
            else:
                problems.extend(self._database_problems())
        return problems

    def _database_problems(self) -> list[str]:
        assert self.base is not None
        admin = create_engine(self.base.set(database="postgres"), isolation_level="AUTOCOMMIT")
        try:
            with admin.connect() as c:
                found = c.execute(
                    text(
                        "SELECT shobj_description(oid, 'pg_database') FROM pg_database "
                        "WHERE datname = :n"
                    ),
                    {"n": self.database},
                ).all()
        except Exception as error:
            return [f"the database could not be checked ({type(error).__name__})"]
        finally:
            admin.dispose()
        if not found:
            return []  # already gone: nothing to delete
        if found[0][0] != COMMENT_PREFIX + run_id(self.token):
            return ["the database is not stamped as belonging to this run"]
        return []

    def cleanup(self) -> list[str]:
        """Stop the processes, then remove what this run created if (and only if) every resource
        is proved to be its own. Returns the resources that remain (empty: all removed)."""
        self.stop()
        problems = self.ownership_problems()
        if problems:
            return [*problems, *self.remaining()]
        if self.database:
            assert self.base is not None
            admin = create_engine(self.base.set(database="postgres"), isolation_level="AUTOCOMMIT")
            try:
                with admin.connect() as c:
                    c.execute(text(f'DROP DATABASE IF EXISTS "{self.database}" WITH (FORCE)'))
            except Exception as error:
                return [f"the database could not be dropped ({type(error).__name__})"]
            finally:
                admin.dispose()
        if self.root is not None and self.root.exists():
            _make_writable(self.root)
            shutil.rmtree(self.root)
        return self.remaining()

    def remaining(self) -> list[str]:
        left: list[str] = []
        if self.root is not None and self.root.exists():
            left.append(f"folder {self.root}")
        if self.database:
            assert self.base is not None
            admin = create_engine(self.base.set(database="postgres"), isolation_level="AUTOCOMMIT")
            try:
                with admin.connect() as c:
                    exists = c.execute(
                        text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": self.database}
                    ).first()
                if exists:
                    left.append(f"database {self.database}")
            except Exception:
                left.append(f"database {self.database} (could not be checked)")
            finally:
                admin.dispose()
        return left


def _make_writable(root: Path) -> None:
    """Sealed packages are read-only on purpose; make this run's own folder removable. Symbolic
    links are never followed."""
    for current, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(current, name)
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
    root.chmod(root.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
