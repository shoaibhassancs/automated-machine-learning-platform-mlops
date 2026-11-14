"""Deployments and the prediction log (P4-7).

Revision ID: 0005
Revises: 0004

A deployment row is one period in which one registered version was live. At most one row is
`active` (partial unique index). The database refuses to deploy a version that is not staging or
production, to change who or what a deployment points at, to change an ended row, to delete a row,
and to change a prediction after it was written. Nothing here changes a registry rule.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

DEPLOYMENT_GUARD = """
CREATE FUNCTION deployments_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    v record;
    prev record;
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'deployments are never deleted' USING ERRCODE = 'check_violation';
    END IF;
    IF TG_OP = 'INSERT' THEN
        SELECT model_id, package_id, status INTO v FROM model_versions WHERE id = NEW.version_id;
        IF v.model_id IS DISTINCT FROM NEW.model_id OR v.package_id IS DISTINCT FROM NEW.package_id
        THEN
            RAISE EXCEPTION 'the deployment does not match its model version'
                USING ERRCODE = 'check_violation';
        END IF;
        IF v.status NOT IN ('staging', 'production') THEN
            RAISE EXCEPTION 'a % version cannot be deployed', v.status
                USING ERRCODE = 'check_violation';
        END IF;
        IF NEW.status <> 'active' THEN
            RAISE EXCEPTION 'a deployment starts active' USING ERRCODE = 'check_violation';
        END IF;
        IF NEW.replaced_id IS NOT NULL THEN
            SELECT status INTO prev FROM deployments WHERE id = NEW.replaced_id;
            IF prev.status IS DISTINCT FROM 'ended' THEN
                RAISE EXCEPTION 'the replaced deployment must be ended first'
                    USING ERRCODE = 'check_violation';
            END IF;
        END IF;
        RETURN NEW;
    END IF;
    -- UPDATE: only active -> ended, and only the end fields change.
    IF OLD.status <> 'active' OR NEW.status <> 'ended' THEN
        RAISE EXCEPTION 'a deployment can only move from active to ended'
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.id IS DISTINCT FROM OLD.id OR NEW.model_id IS DISTINCT FROM OLD.model_id
       OR NEW.version_id IS DISTINCT FROM OLD.version_id
       OR NEW.package_id IS DISTINCT FROM OLD.package_id
       OR NEW.action IS DISTINCT FROM OLD.action OR NEW.reason IS DISTINCT FROM OLD.reason
       OR NEW.replaced_id IS DISTINCT FROM OLD.replaced_id
       OR NEW.deployed_at IS DISTINCT FROM OLD.deployed_at THEN
        RAISE EXCEPTION 'only the end of a deployment can be recorded'
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$
"""

NO_TRUNCATE = """
CREATE FUNCTION deployments_no_truncate() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'deployments are never truncated' USING ERRCODE = 'check_violation';
END
$$
"""

PREDICTION_GUARD = """
CREATE FUNCTION predictions_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'a logged prediction is never changed' USING ERRCODE = 'check_violation';
END
$$
"""


def upgrade() -> None:
    op.create_table(
        "deployments",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("package_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=8), server_default="active", nullable=False),
        sa.Column("action", sa.String(length=10), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("replaced_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "deployed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_deployments")),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["registered_models.id"],
            name=op.f("fk_deployments_model_id_registered_models"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["model_versions.id"],
            name=op.f("fk_deployments_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["replaced_id"],
            ["deployments.id"],
            name=op.f("fk_deployments_replaced_id_deployments"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("status IN ('active', 'ended')", name=op.f("ck_deployments_status")),
        sa.CheckConstraint(
            "action IN ('deploy', 'switch', 'rollback')", name=op.f("ck_deployments_action")
        ),
        sa.CheckConstraint(
            "char_length(reason) BETWEEN 1 AND 500", name=op.f("ck_deployments_reason_length")
        ),
        sa.CheckConstraint(
            "end_reason IS NULL OR char_length(end_reason) BETWEEN 1 AND 500",
            name=op.f("ck_deployments_end_reason_length"),
        ),
        sa.CheckConstraint(
            "package_id ~ '^m-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$'",
            name=op.f("ck_deployments_package_id_format"),
        ),
        sa.CheckConstraint(
            "(status = 'active' AND ended_at IS NULL AND end_reason IS NULL) OR "
            "(status = 'ended' AND ended_at IS NOT NULL AND end_reason IS NOT NULL)",
            name=op.f("ck_deployments_end_consistent"),
        ),
        sa.CheckConstraint(
            "(action = 'deploy' AND replaced_id IS NULL) OR "
            "(action <> 'deploy' AND replaced_id IS NOT NULL)",
            name=op.f("ck_deployments_replaced_consistent"),
        ),
    )
    op.create_index(
        "uq_deployments_one_active",
        "deployments",
        [sa.text("(true)")],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("deployment_id", sa.BigInteger(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("inputs", postgresql.JSONB(), nullable=False),
        sa.Column("output", postgresql.JSONB(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("is_simulated", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
        sa.ForeignKeyConstraint(
            ["deployment_id"],
            ["deployments.id"],
            name=op.f("fk_predictions_deployment_id_deployments"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("latency_ms >= 0", name=op.f("ck_predictions_latency_not_negative")),
    )
    op.create_index(
        "ix_predictions_deployment_created", "predictions", ["deployment_id", "created_at"]
    )
    op.execute(DEPLOYMENT_GUARD)
    op.execute(NO_TRUNCATE)
    op.execute(PREDICTION_GUARD)
    op.execute(
        "CREATE TRIGGER deployments_guard BEFORE INSERT OR UPDATE OR DELETE ON deployments "
        "FOR EACH ROW EXECUTE FUNCTION deployments_guard()"
    )
    op.execute(
        "CREATE TRIGGER deployments_no_truncate BEFORE TRUNCATE ON deployments "
        "FOR EACH STATEMENT EXECUTE FUNCTION deployments_no_truncate()"
    )
    op.execute(
        "CREATE TRIGGER predictions_guard BEFORE UPDATE ON predictions "
        "FOR EACH ROW EXECUTE FUNCTION predictions_guard()"
    )


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("deployments")
    op.execute("DROP FUNCTION deployments_guard()")
    op.execute("DROP FUNCTION deployments_no_truncate()")
    op.execute("DROP FUNCTION predictions_guard()")
