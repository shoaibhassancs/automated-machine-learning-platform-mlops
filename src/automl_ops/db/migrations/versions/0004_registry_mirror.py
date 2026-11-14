"""MLflow mirror state for the registry (P4-6): sync progress and the sync/reconcile audit log.

Revision ID: 0004
Revises: 0003

The registry stays authoritative; nothing here changes a registry rule. `registry_sync.synced_seq`
can only move forward and never past the model's last event number. `registry_sync_log` is
append-only.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

SYNC_GUARD = """
CREATE FUNCTION registry_sync_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    latest integer;
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'mirror progress is never deleted' USING ERRCODE = 'check_violation';
    END IF;
    SELECT last_event_seq INTO latest FROM registered_models WHERE id = NEW.model_id;
    IF NEW.synced_seq > latest THEN
        RAISE EXCEPTION 'synced_seq % is past the last event %', NEW.synced_seq, latest
            USING ERRCODE = 'check_violation';
    END IF;
    IF TG_OP = 'UPDATE' AND (NEW.synced_seq < OLD.synced_seq OR NEW.model_id <> OLD.model_id) THEN
        RAISE EXCEPTION 'synced_seq only moves forward' USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$
"""

LOG_GUARD = """
CREATE FUNCTION registry_sync_log_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'the sync log is append-only' USING ERRCODE = 'check_violation';
END
$$
"""


def upgrade() -> None:
    op.create_table(
        "registry_sync",
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("synced_seq", sa.Integer(), server_default="0", nullable=False),
        sa.Column("synced_state", postgresql.JSONB(), nullable=True),
        sa.Column("attempt_state", postgresql.JSONB(), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_error_kind", sa.String(length=10), nullable=True),
        sa.PrimaryKeyConstraint("model_id", name=op.f("pk_registry_sync")),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["registered_models.id"],
            name=op.f("fk_registry_sync_model_id_registered_models"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("synced_seq >= 0", name=op.f("ck_registry_sync_seq_not_negative")),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_registry_sync_attempts_not_negative")),
        sa.CheckConstraint(
            "last_error_kind IS NULL OR last_error_kind IN ('error', 'drift', 'conflict')",
            name=op.f("ck_registry_sync_error_kind"),
        ),
    )
    op.create_table(
        "registry_sync_log",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column("outcome", sa.String(length=10), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_registry_sync_log")),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["registered_models.id"],
            name=op.f("fk_registry_sync_log_model_id_registered_models"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("kind IN ('sync', 'reconcile')", name=op.f("ck_registry_sync_log_kind")),
        sa.CheckConstraint(
            "outcome IN ('ok', 'failed', 'drift', 'conflict', 'partial')",
            name=op.f("ck_registry_sync_log_outcome"),
        ),
        sa.CheckConstraint(
            "kind <> 'reconcile' OR (reason IS NOT NULL AND char_length(reason) BETWEEN 1 AND 500)",
            name=op.f("ck_registry_sync_log_reconcile_reason"),
        ),
    )
    op.execute(SYNC_GUARD)
    op.execute(LOG_GUARD)
    op.execute(
        "CREATE TRIGGER registry_sync_guard BEFORE INSERT OR UPDATE OR DELETE ON registry_sync "
        "FOR EACH ROW EXECUTE FUNCTION registry_sync_guard()"
    )
    op.execute(
        "CREATE TRIGGER registry_sync_log_guard BEFORE UPDATE OR DELETE ON registry_sync_log "
        "FOR EACH ROW EXECUTE FUNCTION registry_sync_log_guard()"
    )
    op.execute(
        "CREATE TRIGGER registry_sync_log_no_truncate BEFORE TRUNCATE ON registry_sync_log "
        "FOR EACH STATEMENT EXECUTE FUNCTION registry_sync_log_guard()"
    )


def downgrade() -> None:
    op.drop_table("registry_sync_log")
    op.drop_table("registry_sync")
    op.execute("DROP FUNCTION registry_sync_guard()")
    op.execute("DROP FUNCTION registry_sync_log_guard()")
