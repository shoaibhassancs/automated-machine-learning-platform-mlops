"""Training jobs and their progress events (P4-4).

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

SHA256_REGEX = "^[0-9a-f]{64}$"
FINAL = "('succeeded', 'failed', 'cancelled')"

# Defence in depth for the repository's own checks: a finished job never changes, and the only
# status moves are queued -> running | cancelled and running -> a final state.
GUARD_FUNCTION = """
CREATE FUNCTION training_jobs_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.status IN ('succeeded', 'failed', 'cancelled') THEN
        RAISE EXCEPTION 'job % is finished and cannot change', OLD.id
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.status <> OLD.status AND NOT (
        (OLD.status = 'queued' AND NEW.status IN ('running', 'cancelled'))
        OR (OLD.status = 'running' AND NEW.status IN ('succeeded', 'failed', 'cancelled'))
    ) THEN
        RAISE EXCEPTION 'job % cannot move from % to %', OLD.id, OLD.status, NEW.status
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$
"""


def upgrade() -> None:
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=10), server_default="queued", nullable=False),
        sa.Column("phase", sa.String(length=12), server_default="queued", nullable=False),
        sa.Column("dataset_version_id", sa.Integer(), nullable=False),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("track", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("confirm_final_test", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("progress", postgresql.JSONB(), nullable=True),
        sa.Column("worker_id", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("package_id", sa.String(length=64), nullable=True),
        sa.Column("tracking_state", sa.String(length=8), server_default="off", nullable=False),
        sa.Column("tracking_job_key", sa.String(length=64), nullable=True),
        sa.Column("tracking_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_jobs")),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            name=op.f("fk_training_jobs_dataset_version_id_dataset_versions"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("kind IN ('train', 'final')", name=op.f("ck_training_jobs_kind")),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name=op.f("ck_training_jobs_status"),
        ),
        sa.CheckConstraint(
            "phase IN ('queued', 'loading', 'training', 'selecting', 'finalizing', "
            "'tracking', 'done')",
            name=op.f("ck_training_jobs_phase"),
        ),
        sa.CheckConstraint(
            "tracking_state IN ('off', 'logged', 'pending')",
            name=op.f("ck_training_jobs_tracking_state"),
        ),
        sa.CheckConstraint(
            f"dataset_sha256 ~ '{SHA256_REGEX}'",
            name=op.f("ck_training_jobs_dataset_sha256_format"),
        ),
        sa.CheckConstraint(
            "kind <> 'final' OR confirm_final_test",
            name=op.f("ck_training_jobs_final_is_confirmed"),
        ),
        sa.CheckConstraint(
            f"(status IN {FINAL}) = (finished_at IS NOT NULL)",
            name=op.f("ck_training_jobs_finished"),
        ),
        sa.CheckConstraint(
            "status <> 'running' OR (started_at IS NOT NULL AND worker_id IS NOT NULL)",
            name=op.f("ck_training_jobs_running_is_claimed"),
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR error_code IS NOT NULL",
            name=op.f("ck_training_jobs_failed_has_error"),
        ),
    )
    op.create_index("ix_training_jobs_status_id", "training_jobs", ["status", "id"])
    op.create_index(
        "uq_training_jobs_one_running",
        "training_jobs",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )
    op.create_table(
        "job_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_events")),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["training_jobs.id"],
            name=op.f("fk_job_events_job_id_training_jobs"),
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("job_id", "seq", name=op.f("uq_job_events_job_seq")),
        sa.CheckConstraint("seq > 0", name=op.f("ck_job_events_seq_positive")),
    )
    op.execute(GUARD_FUNCTION)
    op.execute(
        "CREATE TRIGGER training_jobs_guard BEFORE UPDATE ON training_jobs "
        "FOR EACH ROW EXECUTE FUNCTION training_jobs_guard()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER training_jobs_guard ON training_jobs")
    op.execute("DROP FUNCTION training_jobs_guard()")
    op.drop_table("job_events")
    op.drop_table("training_jobs")
