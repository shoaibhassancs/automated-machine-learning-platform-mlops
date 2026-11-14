"""Run and inspect Alembic migrations explicitly (`db upgrade`, `db status`, `db check`).

Migrations never run on their own: no application start-up and no import applies them. Upgrades
hold a PostgreSQL advisory lock, so two processes upgrading at once cannot interleave; the second
finds the database already current. Each migration runs in one transaction (PostgreSQL DDL is
transactional), so a failed migration leaves nothing half-applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.script.revision import ResolutionError
from alembic.util import CommandError
from sqlalchemy import CheckConstraint, Engine, inspect, text
from sqlalchemy.engine import Connection

from automl_ops.db.models import Base

MIGRATIONS = Path(__file__).resolve().parent / "migrations"
ADVISORY_LOCK_KEY = 7_265_470_001  # arbitrary, fixed: "automl-ops migrations"


@dataclass(frozen=True)
class DbStatus:
    current: str | None  # revision applied to the database; None for an empty database
    head: str  # revision the code expects
    pending: tuple[str, ...]  # revisions an upgrade would apply, oldest first
    known: bool  # False: the database is at a revision this code does not have

    @property
    def up_to_date(self) -> bool:
        return self.known and self.current == self.head

    def to_dict(self) -> dict[str, object]:
        return {
            "current": self.current,
            "head": self.head,
            "pending": list(self.pending),
            "known": self.known,
            "up_to_date": self.up_to_date,
        }


@dataclass(frozen=True)
class CheckResult:
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def _config(connection: Connection | None = None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def _scripts() -> ScriptDirectory:
    return ScriptDirectory.from_config(_config())


def head_revision() -> str:
    heads = _scripts().get_heads()
    if len(heads) != 1:
        raise RuntimeError(f"expected exactly one migration head, found {heads}")
    return heads[0]


def _status(connection: Connection) -> DbStatus:
    scripts = _scripts()
    head = head_revision()
    current = MigrationContext.configure(connection).get_current_revision()
    if current is not None:
        try:
            known = scripts.get_revision(current) is not None
        except (ResolutionError, CommandError):
            known = False
        if not known:
            return DbStatus(current, head, (), known=False)
    pending = tuple(
        r.revision for r in reversed(list(scripts.iterate_revisions(head, current or "base")))
    )
    if current is not None:
        pending = tuple(p for p in pending if p != current)
    return DbStatus(current, head, pending, known=True)


def status(engine: Engine) -> DbStatus:
    with engine.connect() as connection:
        return _status(connection)


def _locked(engine: Engine, action: str, revision: str) -> DbStatus:
    with engine.connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": ADVISORY_LOCK_KEY})
        connection.commit()
        try:
            config = _config(connection)
            if action == "upgrade":
                command.upgrade(config, revision)
            else:
                command.downgrade(config, revision)
            connection.commit()
            return _status(connection)
        finally:
            connection.rollback()
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": ADVISORY_LOCK_KEY})
            connection.commit()


def upgrade(engine: Engine, revision: str = "head") -> DbStatus:
    """Apply migrations up to `revision` (default: the head). Applying none is not an error."""
    return _locked(engine, "upgrade", revision)


def downgrade(engine: Engine, revision: str) -> DbStatus:
    """Revert to `revision`. Used by tests; deliberately not offered by the command line."""
    return _locked(engine, "downgrade", revision)


def check(engine: Engine) -> CheckResult:
    """Is the database at the head, and does its structure equal what the models declare?"""
    problems: list[str] = []
    with engine.connect() as connection:
        state = _status(connection)
        if not state.known:
            return CheckResult(
                (f"the database is at revision {state.current}, unknown to this code",)
            )
        if not state.up_to_date:
            return CheckResult(
                (
                    f"the database is at {state.current or 'no revision'}, the code expects "
                    f"{state.head}: run `db upgrade`",
                )
            )
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        for difference in compare_metadata(context, Base.metadata):
            problems.append(f"structure differs from the models: {difference}")
        inspector = inspect(connection)
        for table in Base.metadata.sorted_tables:
            actual = {c["name"] for c in inspector.get_check_constraints(table.name)}
            for constraint in table.constraints:
                if isinstance(constraint, CheckConstraint) and constraint.name:
                    expected = str(constraint.name)  # already resolved by the naming convention
                    if expected not in actual:
                        problems.append(f"{table.name}: check constraint {expected} is missing")
    return CheckResult(tuple(problems))
