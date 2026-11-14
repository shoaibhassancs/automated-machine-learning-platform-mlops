"""Alembic environment. Migrations are run only through `automl_ops.db.migrate` (explicit
`db upgrade`), which passes an open connection; there is no offline (SQL script) mode."""

from __future__ import annotations

from alembic import context

from automl_ops.db.models import Base

config = context.config


def run_migrations() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        raise RuntimeError(
            "run migrations through automl_ops.db.migrate, which opens the connection"
        )
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        compare_type=True,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("offline migrations are not supported")
run_migrations()
