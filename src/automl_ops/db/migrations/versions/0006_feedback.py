"""Feedback labels for logged predictions (P4-8).

Revision ID: 0006
Revises: 0005

One label per prediction (unique). The database refuses a label whose deployment or model version
differs from the prediction's own, and refuses to change or remove a label afterwards.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

GUARD = """
CREATE FUNCTION feedback_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    p record;
BEGIN
    IF TG_OP <> 'INSERT' THEN
        RAISE EXCEPTION 'a feedback label is never changed or removed'
            USING ERRCODE = 'check_violation';
    END IF;
    SELECT deployment_id, model_version INTO p FROM predictions WHERE id = NEW.prediction_id;
    IF p.deployment_id IS DISTINCT FROM NEW.deployment_id
       OR p.model_version IS DISTINCT FROM NEW.model_version THEN
        RAISE EXCEPTION 'the label does not match its prediction''s deployment and version'
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$
"""

NO_TRUNCATE = """
CREATE FUNCTION feedback_no_truncate() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'feedback is never truncated' USING ERRCODE = 'check_violation';
END
$$
"""


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=False),
        sa.Column("deployment_id", sa.BigInteger(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("true_value", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name=op.f("fk_feedback_prediction_id_predictions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["deployment_id"],
            ["deployments.id"],
            name=op.f("fk_feedback_deployment_id_deployments"),
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("prediction_id", name=op.f("uq_feedback_prediction")),
        sa.CheckConstraint(
            "char_length(true_value) BETWEEN 1 AND 200", name=op.f("ck_feedback_true_value_length")
        ),
    )
    op.create_index("ix_feedback_deployment_id", "feedback", ["deployment_id", "id"])
    op.execute(GUARD)
    op.execute(NO_TRUNCATE)
    op.execute(
        "CREATE TRIGGER feedback_guard BEFORE INSERT OR UPDATE OR DELETE ON feedback "
        "FOR EACH ROW EXECUTE FUNCTION feedback_guard()"
    )
    op.execute(
        "CREATE TRIGGER feedback_no_truncate BEFORE TRUNCATE ON feedback "
        "FOR EACH STATEMENT EXECUTE FUNCTION feedback_no_truncate()"
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.execute("DROP FUNCTION feedback_guard()")
    op.execute("DROP FUNCTION feedback_no_truncate()")
