"""The model registry (P4-5): models, versions and the append-only event log.

Revision ID: 0003
Revises: 0002

Event integrity is enforced here, not only in Python. A version's status can change only by
inserting a `registry_events` row: a BEFORE INSERT trigger checks the move, allocates the event's
`seq` under a per-model row lock, and applies the new status. A second trigger refuses any direct
status change, and a deferred constraint trigger refuses a version without its `register` event.
Events can never be updated, deleted or truncated.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

NAME_REGEX = "^[a-z0-9][a-z0-9_-]{0,63}$"
SHA256_REGEX = "^[0-9a-f]{64}$"
PACKAGE_ID_REGEX = r"^m-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
STATES = "('development', 'staging', 'production', 'archived')"
ACTIONS = "('register', 'promote', 'archive', 'supersede', 'rollback')"

APPLY_FUNCTION = """
CREATE FUNCTION registry_events_apply() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    n integer;
    cur text;
    owner integer;
    prev registry_events%ROWTYPE;
BEGIN
    UPDATE registered_models SET last_event_seq = last_event_seq + 1
        WHERE id = NEW.model_id RETURNING last_event_seq INTO n;
    IF n IS NULL THEN
        RAISE EXCEPTION 'model % does not exist', NEW.model_id USING ERRCODE = 'check_violation';
    END IF;
    NEW.seq := n;
    SELECT status, model_id INTO cur, owner FROM model_versions
        WHERE id = NEW.version_id FOR UPDATE;
    IF cur IS NULL OR owner <> NEW.model_id THEN
        RAISE EXCEPTION 'version % does not belong to model %', NEW.version_id, NEW.model_id
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.action = 'register' THEN
        IF NEW.from_status IS NOT NULL OR NEW.to_status <> 'development'
           OR cur <> 'development'
           OR EXISTS (SELECT 1 FROM registry_events WHERE version_id = NEW.version_id) THEN
            RAISE EXCEPTION 'invalid register event for version %', NEW.version_id
                USING ERRCODE = 'check_violation';
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.from_status IS DISTINCT FROM cur THEN
        RAISE EXCEPTION 'version % is %, not %', NEW.version_id, cur, NEW.from_status
            USING ERRCODE = 'check_violation';
    END IF;
    IF NOT (
        (NEW.action = 'promote' AND (
            (cur = 'development' AND NEW.to_status = 'staging')
            OR (cur = 'staging' AND NEW.to_status = 'production')
            OR (cur = 'archived' AND NEW.to_status = 'staging')))
        OR (NEW.action = 'archive' AND cur IN ('development', 'staging')
            AND NEW.to_status = 'archived')
        OR (NEW.action = 'supersede' AND cur = 'production' AND NEW.to_status = 'archived')
        OR (NEW.action = 'rollback' AND (
            (cur = 'production' AND NEW.to_status = 'archived')
            OR (cur = 'archived' AND NEW.to_status = 'production')))
    ) THEN
        RAISE EXCEPTION 'move % from % to % is not allowed', NEW.action, cur, NEW.to_status
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.action = 'rollback' AND NEW.to_status = 'production' THEN
        SELECT * INTO prev FROM registry_events
            WHERE model_id = NEW.model_id AND seq = NEW.seq - 1;
        IF NOT FOUND OR prev.action <> 'rollback' OR prev.to_status <> 'archived'
           OR prev.version_id = NEW.version_id
           OR NOT EXISTS (SELECT 1 FROM registry_events
                          WHERE version_id = NEW.version_id AND to_status = 'production') THEN
            RAISE EXCEPTION 'a rollback restore needs the matching rollback archive just before it'
                USING ERRCODE = 'check_violation';
        END IF;
    END IF;
    PERFORM set_config('automl.registry_apply', 'on', true);
    UPDATE model_versions SET status = NEW.to_status, status_changed_at = now()
        WHERE id = NEW.version_id;
    PERFORM set_config('automl.registry_apply', 'off', true);
    RETURN NEW;
END
$$
"""

VERSION_GUARD = """
CREATE FUNCTION model_versions_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'registered versions are never deleted' USING ERRCODE = 'check_violation';
    END IF;
    IF TG_OP = 'INSERT' THEN
        IF NEW.status <> 'development' THEN
            RAISE EXCEPTION 'a new version starts in development' USING ERRCODE = 'check_violation';
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.status <> OLD.status
       AND current_setting('automl.registry_apply', true) IS DISTINCT FROM 'on' THEN
        RAISE EXCEPTION 'a status changes only through a registry event'
            USING ERRCODE = 'check_violation';
    END IF;
    IF (NEW.model_id, NEW.version, NEW.package_id, NEW.job_id, NEW.dataset_version_id,
        NEW.registered_at, NEW.algorithm, NEW.task, NEW.target, NEW.selection_sha256,
        NEW.evaluation)
       IS DISTINCT FROM
       (OLD.model_id, OLD.version, OLD.package_id, OLD.job_id, OLD.dataset_version_id,
        OLD.registered_at, OLD.algorithm, OLD.task, OLD.target, OLD.selection_sha256,
        OLD.evaluation) THEN
        RAISE EXCEPTION 'a registered version is immutable apart from its status'
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$
"""

VERSION_NEEDS_EVENT = """
CREATE FUNCTION model_versions_need_register_event() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM registry_events
                   WHERE version_id = NEW.id AND action = 'register') THEN
        RAISE EXCEPTION 'version % has no register event', NEW.id
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NULL;
END
$$
"""

MODEL_GUARD = """
CREATE FUNCTION registered_models_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'registered models are never deleted' USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.name <> OLD.name OR NEW.created_at <> OLD.created_at
       OR NEW.last_event_seq NOT IN (OLD.last_event_seq, OLD.last_event_seq + 1) THEN
        RAISE EXCEPTION 'a registered model changes only by allocating the next event number'
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$
"""

EVENTS_GUARD = """
CREATE FUNCTION registry_events_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'registry events are append-only' USING ERRCODE = 'check_violation';
END
$$
"""


def upgrade() -> None:
    op.create_table(
        "registered_models",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("last_event_seq", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_registered_models")),
        sa.UniqueConstraint("name", name=op.f("uq_registered_models_name")),
        sa.CheckConstraint(f"name ~ '{NAME_REGEX}'", name=op.f("ck_registered_models_name_format")),
        sa.CheckConstraint(
            "last_event_seq >= 0", name=op.f("ck_registered_models_seq_not_negative")
        ),
    )
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("package_id", sa.String(length=64), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=True),
        sa.Column("dataset_version_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=12), server_default="development", nullable=False),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "status_changed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("algorithm", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=16), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("selection_sha256", sa.String(length=64), nullable=False),
        sa.Column("evaluation", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["registered_models.id"],
            name=op.f("fk_model_versions_model_id_registered_models"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["training_jobs.id"],
            name=op.f("fk_model_versions_job_id_training_jobs"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            name=op.f("fk_model_versions_dataset_version_id_dataset_versions"),
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("model_id", "version", name=op.f("uq_model_versions_model_version")),
        sa.UniqueConstraint("package_id", name=op.f("uq_model_versions_package")),
        sa.CheckConstraint("version > 0", name=op.f("ck_model_versions_version_positive")),
        sa.CheckConstraint(f"status IN {STATES}", name=op.f("ck_model_versions_status")),
        sa.CheckConstraint(
            f"package_id ~ '{PACKAGE_ID_REGEX}'", name=op.f("ck_model_versions_package_id_format")
        ),
        sa.CheckConstraint(
            f"selection_sha256 ~ '{SHA256_REGEX}'",
            name=op.f("ck_model_versions_selection_sha256_format"),
        ),
    )
    op.create_index(
        "uq_model_versions_one_production",
        "model_versions",
        ["model_id"],
        unique=True,
        postgresql_where=sa.text("status = 'production'"),
    )
    op.create_table(
        "registry_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=12), nullable=False),
        sa.Column("from_status", sa.String(length=12), nullable=True),
        sa.Column("to_status", sa.String(length=12), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_registry_events")),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["registered_models.id"],
            name=op.f("fk_registry_events_model_id_registered_models"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["model_versions.id"],
            name=op.f("fk_registry_events_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("model_id", "seq", name=op.f("uq_registry_events_model_seq")),
        sa.CheckConstraint("seq > 0", name=op.f("ck_registry_events_seq_positive")),
        sa.CheckConstraint(f"action IN {ACTIONS}", name=op.f("ck_registry_events_action")),
        sa.CheckConstraint(f"to_status IN {STATES}", name=op.f("ck_registry_events_to_status")),
        sa.CheckConstraint(
            f"from_status IS NULL OR from_status IN {STATES}",
            name=op.f("ck_registry_events_from_status"),
        ),
        sa.CheckConstraint(
            "char_length(reason) BETWEEN 1 AND 500", name=op.f("ck_registry_events_reason_length")
        ),
    )
    for function in (
        APPLY_FUNCTION,
        VERSION_GUARD,
        VERSION_NEEDS_EVENT,
        MODEL_GUARD,
        EVENTS_GUARD,
    ):
        op.execute(function)
    op.execute(
        "CREATE TRIGGER registry_events_apply BEFORE INSERT ON registry_events "
        "FOR EACH ROW EXECUTE FUNCTION registry_events_apply()"
    )
    op.execute(
        "CREATE TRIGGER registry_events_guard BEFORE UPDATE OR DELETE ON registry_events "
        "FOR EACH ROW EXECUTE FUNCTION registry_events_guard()"
    )
    op.execute(
        "CREATE TRIGGER registry_events_no_truncate BEFORE TRUNCATE ON registry_events "
        "FOR EACH STATEMENT EXECUTE FUNCTION registry_events_guard()"
    )
    op.execute(
        "CREATE TRIGGER model_versions_guard BEFORE INSERT OR UPDATE OR DELETE ON model_versions "
        "FOR EACH ROW EXECUTE FUNCTION model_versions_guard()"
    )
    op.execute(
        "CREATE CONSTRAINT TRIGGER model_versions_need_event AFTER INSERT ON model_versions "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
        "EXECUTE FUNCTION model_versions_need_register_event()"
    )
    op.execute(
        "CREATE TRIGGER registered_models_guard BEFORE UPDATE OR DELETE ON registered_models "
        "FOR EACH ROW EXECUTE FUNCTION registered_models_guard()"
    )


def downgrade() -> None:
    op.drop_table("registry_events")
    op.drop_table("model_versions")
    op.drop_table("registered_models")
    for function in (
        "registry_events_apply",
        "registry_events_guard",
        "model_versions_guard",
        "model_versions_need_register_event",
        "registered_models_guard",
    ):
        op.execute(f"DROP FUNCTION {function}()")
