"""Is this process wired to a throwaway stack? (follow-up to P4-10: E2E script safety)

The end-to-end story writes rows and model packages, so it must only run against a stack made for
it. Every service reports, on `/health`, whether *it* is isolated; the E2E script refuses unless
both services, the script's own settings and the stack's contents all agree. Nothing here opens a
database connection, a model or a data file: it only looks at names and paths.

A process is **isolated** only if every check holds:

* `database`: the application database is a throwaway one (name starts with `automl_test_e2e_`;
  never `automl_ops`, `mlflow`, `postgres` or a template);
* `store`, `models`: the dataset store and model folder are outside the project's `data/`
  directory;
* `mlflow`: tracking uses the SQLite backend in a directory outside `data/` (and not the Phase 3
  archive), never the PostgreSQL `mlflow` database;
* `token`: the per-run token `AUTOML_E2E_RUN_TOKEN` (made by `scripts/e2e_isolated.py`) is
  present.

The report carries booleans, reason codes and two short one-way identifiers (the database and the
run). It never carries a path, a host, a user name, a password, or the token itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.engine import URL

from automl_ops.bind import in_container
from automl_ops.db.config import setting

TOKEN_SETTING = "AUTOML_E2E_RUN_TOKEN"
THROWAWAY_DB_PREFIX = "automl_test_e2e_"
# The full Compose stack (compose.full.yaml) has its own PostgreSQL server in its own volume. Its
# database is accepted as isolated only inside the project's image (see `bind.in_container`), and
# only when the host and port the service reports match what the Compose file declares (the script
# compares the one-way identifiers).
COMPOSE_DB = "automl_compose"
COMPOSE_DB_HOST = "postgres"
COMPOSE_DB_PORT = 5432
PROTECTED_DATABASES = frozenset({"automl_ops", "mlflow", "postgres", "template0", "template1"})
MIN_TOKEN_LENGTH = 16
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKS = ("database", "store", "models", "mlflow", "token")


def _short(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def database_id(url: URL) -> str:
    """A one-way identifier of server + database, safe to print (not reversible to either)."""
    return _short("db", str(url.host or ""), str(url.port or ""), str(url.database or ""))


def run_id(token: str) -> str:
    """A one-way identifier of an E2E run's token."""
    return _short("run", token)


def inside(path: Path, directory: Path) -> bool:
    """True if `path` is `directory` or below it, following symbolic links and ignoring case."""
    try:
        target, base = path.expanduser().resolve(), directory.resolve()
    except (OSError, RuntimeError):
        return True  # cannot tell: treat as unsafe
    if target == base or base in target.parents:
        return True
    folded = target.as_posix().casefold()
    return folded == base.as_posix().casefold() or folded.startswith(
        base.as_posix().casefold() + "/"
    )


@dataclass(frozen=True)
class Isolation:
    isolated: bool
    checks: dict[str, bool]
    reasons: list[str]
    database_id: str | None
    run_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NOT_ASSESSED = Isolation(
    isolated=False,
    checks=dict.fromkeys(CHECKS, False),
    reasons=["not_assessed"],
    database_id=None,
    run_id=None,
)


@dataclass(frozen=True)
class Wiring:
    """What a process is connected to (names and paths only)."""

    database: URL
    store_root: Path
    models_root: Path
    env: Mapping[str, str] | None = None
    project_data: Path = field(default_factory=lambda: PROJECT_ROOT / "data")


def _mlflow_problem(wiring: Wiring) -> str | None:
    """From the settings only (the serving app must not import MLflow): the same backend and
    directory rules that `job_tracking` applies, restated as isolation checks."""
    env = None if wiring.env is None else dict(wiring.env)
    backend = (setting("AUTOML_MLFLOW_BACKEND", "sqlite", env=env) or "sqlite").lower()
    if backend != "sqlite":
        return "mlflow_not_sqlite"  # the PostgreSQL `mlflow` database is never isolated
    directory = setting("AUTOML_JOBS_MLFLOW_DIR", None, env=env)
    if not directory:
        return "mlflow_not_configured"
    root = Path(directory).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if inside(root, wiring.project_data):
        return "mlflow_inside_protected_data"  # includes the Phase 3 archive data/mlflow
    return None


def assess(wiring: Wiring) -> Isolation:
    """The isolation report for one process."""
    env = None if wiring.env is None else dict(wiring.env)
    name = wiring.database.database or ""
    token = setting(TOKEN_SETTING, None, env=env) or ""
    reasons: dict[str, str] = {}
    compose_db = name == COMPOSE_DB and in_container(env)
    if name in PROTECTED_DATABASES or not (name.startswith(THROWAWAY_DB_PREFIX) or compose_db):
        reasons["database"] = "database_not_throwaway"
    if inside(wiring.store_root, wiring.project_data):
        reasons["store"] = "store_inside_protected_data"
    if inside(wiring.models_root, wiring.project_data):
        reasons["models"] = "models_inside_protected_data"
    problem = _mlflow_problem(wiring)
    if problem is not None:
        reasons["mlflow"] = problem
    if len(token) < MIN_TOKEN_LENGTH:
        reasons["token"] = "run_token_missing"
    return Isolation(
        isolated=not reasons,
        checks={check: check not in reasons for check in CHECKS},
        reasons=sorted(reasons.values()),
        database_id=database_id(wiring.database),
        run_id=run_id(token) if token else None,
    )


def reporter(wiring: Wiring) -> Callable[[], Isolation]:
    """A function for a service to call on every `/health`."""
    return lambda: assess(wiring)


# --------------------------------------------------------------------------------------------
# The E2E script's verification (read-only requests only)
# --------------------------------------------------------------------------------------------

EMPTY_LISTS = (
    ("/datasets", "datasets"),
    ("/models", "models"),
    ("/jobs", "jobs"),
    ("/deployments", "deployments"),
)


def _listed(reply: Any, key: str) -> list[Any] | None:
    body = reply.body
    items = body.get(key) if reply.status == 200 and isinstance(body, dict) else None
    return items if isinstance(items, list) else None


def _reported(body: Any) -> dict[str, Any] | None:
    block = body.get("isolation") if isinstance(body, dict) else None
    return block if isinstance(block, dict) else None


def compose_expectation(token: str) -> Isolation:
    """What the E2E script expects of the full Compose stack: the identifiers of the database the
    Compose file declares and of the run token in the git-ignored `.env.full`. It is a statement
    of identity, not a check of this machine; `verify_stack` then compares it with what both
    services report."""
    wiring_url = URL.create(
        "postgresql+psycopg", host=COMPOSE_DB_HOST, port=COMPOSE_DB_PORT, database=COMPOSE_DB
    )
    ok = len(token) >= MIN_TOKEN_LENGTH
    return Isolation(
        isolated=ok,
        checks=dict.fromkeys(CHECKS, ok),
        reasons=[] if ok else ["run_token_missing"],
        database_id=database_id(wiring_url),
        run_id=run_id(token) if ok else None,
    )


def verify_stack(
    local: Isolation,
    control: Any,
    serving: Any,
) -> list[str]:
    """Every reason the stack must not be written to (empty list: safe). Uses GET requests only.

    `control` and `serving` are objects with `request(method, path)` returning `.status` and `.body`
    (the simulator's `Http`). The checks, in order: this script's own settings are isolated; both
    services answer and report isolation; both are on this script's database and run; the
    control plane's lists of datasets, models, jobs and deployments are empty."""
    problems = [f"this script: {reason}" for reason in local.reasons]
    if not local.isolated:
        return problems  # nothing is contacted when the script's own settings are not isolated
    reports: dict[str, dict[str, Any] | None] = {}
    for name, client in (("control plane", control), ("serving app", serving)):
        reply = client.request("GET", "/health")
        block = _reported(reply.body) if reply.status in (200, 503) else None
        reports[name] = block
        if reply.status == 0:
            problems.append(f"{name}: not answering")
        elif block is None:
            problems.append(f"{name}: does not report isolation (an older or unknown stack)")
        elif not block.get("isolated"):
            reasons = ", ".join(map(str, block.get("reasons") or [])) or "unknown"
            problems.append(f"{name}: not isolated ({reasons})")
        else:
            if block.get("database_id") != local.database_id:
                problems.append(f"{name}: uses a different database than this run")
            if block.get("run_id") != local.run_id:
                problems.append(f"{name}: belongs to a different run")
    if problems:
        return problems
    for path, key in EMPTY_LISTS:
        reply = control.request("GET", path)
        items = _listed(reply, key)
        if items is None:
            problems.append(f"control plane: could not list {key}")
        elif items:
            problems.append(f"the stack is not empty ({key} exist)")
    return problems
