"""Reference samples, monitoring reports, alerts and the alert audit log (P4-9).

Revision ID: 0007
Revises: 0006

Reference samples and reports are immutable. An alert moves forward only (open -> acknowledged ->
resolved), a resolved alert never changes, at most one unresolved alert exists per (deployment,
rule, traffic scope), and every state change is an append-only event.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

NO_CHANGE = """
CREATE FUNCTION immutable_rows_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% rows are never changed or removed', TG_TABLE_NAME
        USING ERRCODE = 'check_violation';
END
$$
"""

ALERT_GUARD = """
CREATE FUNCTION alerts_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'alerts are never deleted' USING ERRCODE = 'check_violation';
    END IF;
    IF OLD.status = 'resolved' THEN
        RAISE EXCEPTION 'a resolved alert never changes' USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.id IS DISTINCT FROM OLD.id OR NEW.deployment_id IS DISTINCT FROM OLD.deployment_id
       OR NEW.rule IS DISTINCT FROM OLD.rule OR NEW.scope IS DISTINCT FROM OLD.scope
       OR NEW.severity IS DISTINCT FROM OLD.severity
       OR NEW.first_report_id IS DISTINCT FROM OLD.first_report_id
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'only the state of an alert can change' USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.occurrences < OLD.occurrences THEN
        RAISE EXCEPTION 'occurrences only grow' USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.status <> OLD.status AND NOT (
        (OLD.status = 'open' AND NEW.status IN ('acknowledged', 'resolved'))
        OR (OLD.status = 'acknowledged' AND NEW.status = 'resolved')
    ) THEN
        RAISE EXCEPTION 'an alert cannot move from % to %', OLD.status, NEW.status
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$
"""

TABLES_TRUNCATE = ("reference_samples", "monitoring_reports", "alerts", "alert_events")


def upgrade() -> None:
    op.create_table(
        "reference_samples",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("package_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_ref", sa.Text(), nullable=False),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=False),
        sa.Column("train_sha256", sa.String(length=64), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("train_rows", sa.Integer(), nullable=False),
        sa.Column("sample_rows", sa.Integer(), nullable=False),
        sa.Column("columns", postgresql.JSONB(), nullable=False),
        sa.Column("rows", postgresql.JSONB(), nullable=False),
        sa.Column("predictions", postgresql.JSONB(), nullable=False),
        sa.Column("sample_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reference_samples")),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["model_versions.id"],
            name=op.f("fk_reference_samples_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("version_id", name=op.f("uq_reference_samples_version")),
        sa.CheckConstraint(
            "sample_rows > 0 AND sample_rows <= train_rows",
            name=op.f("ck_reference_samples_sample_size"),
        ),
        sa.CheckConstraint(
            "sample_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_reference_samples_sample_sha256_format"),
        ),
        sa.CheckConstraint(
            "dataset_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_reference_samples_dataset_sha256_format"),
        ),
        sa.CheckConstraint(
            "train_sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_reference_samples_train_sha256_format")
        ),
        sa.CheckConstraint(
            "package_id ~ '^m-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$'",
            name=op.f("ck_reference_samples_package_id_format"),
        ),
    )
    op.create_table(
        "monitoring_reports",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("deployment_id", sa.BigInteger(), nullable=False),
        sa.Column("reference_id", sa.Integer(), nullable=False),
        sa.Column("scope", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=18), nullable=False),
        sa.Column("window_rows", sa.Integer(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("thresholds", postgresql.JSONB(), nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "finished_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_monitoring_reports")),
        sa.ForeignKeyConstraint(
            ["deployment_id"],
            ["deployments.id"],
            name=op.f("fk_monitoring_reports_deployment_id_deployments"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["reference_id"],
            ["reference_samples.id"],
            name=op.f("fk_monitoring_reports_reference_id_reference_samples"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "scope IN ('real', 'simulated')", name=op.f("ck_monitoring_reports_scope")
        ),
        sa.CheckConstraint(
            "status IN ('ok', 'insufficient_data', 'failed')",
            name=op.f("ck_monitoring_reports_status"),
        ),
        sa.CheckConstraint(
            "window_rows >= 0", name=op.f("ck_monitoring_reports_window_rows_not_negative")
        ),
        sa.CheckConstraint(
            "(status = 'failed' AND error IS NOT NULL) OR "
            "(status <> 'failed' AND error IS NULL AND summary IS NOT NULL)",
            name=op.f("ck_monitoring_reports_outcome_consistent"),
        ),
    )
    op.create_index(
        "ix_monitoring_reports_deployment_id", "monitoring_reports", ["deployment_id", "id"]
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("deployment_id", sa.BigInteger(), nullable=False),
        sa.Column("rule", sa.String(length=24), nullable=False),
        sa.Column("scope", sa.String(length=10), nullable=False),
        sa.Column("severity", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=12), server_default="open", nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column("first_report_id", sa.BigInteger(), nullable=False),
        sa.Column("last_report_id", sa.BigInteger(), nullable=False),
        sa.Column("occurrences", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alerts")),
        sa.ForeignKeyConstraint(
            ["deployment_id"],
            ["deployments.id"],
            name=op.f("fk_alerts_deployment_id_deployments"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["first_report_id"],
            ["monitoring_reports.id"],
            name=op.f("fk_alerts_first_report_id_monitoring_reports"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["last_report_id"],
            ["monitoring_reports.id"],
            name=op.f("fk_alerts_last_report_id_monitoring_reports"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "rule IN ('drifted_feature_share', 'prediction_shift', 'performance_drop')",
            name=op.f("ck_alerts_rule"),
        ),
        sa.CheckConstraint("scope IN ('real', 'simulated')", name=op.f("ck_alerts_scope")),
        sa.CheckConstraint("severity IN ('warning', 'critical')", name=op.f("ck_alerts_severity")),
        sa.CheckConstraint(
            "status IN ('open', 'acknowledged', 'resolved')", name=op.f("ck_alerts_status")
        ),
        sa.CheckConstraint("occurrences >= 1", name=op.f("ck_alerts_occurrences_positive")),
        sa.CheckConstraint(
            "(status = 'resolved') = (resolved_at IS NOT NULL)",
            name=op.f("ck_alerts_resolved_consistent"),
        ),
    )
    op.create_index(
        "uq_alerts_one_unresolved",
        "alerts",
        ["deployment_id", "rule", "scope"],
        unique=True,
        postgresql_where=sa.text("status <> 'resolved'"),
    )
    op.create_table(
        "alert_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("alert_id", sa.BigInteger(), nullable=False),
        sa.Column("action", sa.String(length=14), nullable=False),
        sa.Column("report_id", sa.BigInteger(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alert_events")),
        sa.ForeignKeyConstraint(
            ["alert_id"],
            ["alerts.id"],
            name=op.f("fk_alert_events_alert_id_alerts"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["report_id"],
            ["monitoring_reports.id"],
            name=op.f("fk_alert_events_report_id_monitoring_reports"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "action IN ('opened', 'reoccurred', 'acknowledged', 'resolved', 'auto_resolved')",
            name=op.f("ck_alert_events_action"),
        ),
        sa.CheckConstraint(
            "reason IS NULL OR char_length(reason) BETWEEN 1 AND 500",
            name=op.f("ck_alert_events_reason_length"),
        ),
    )
    op.execute(NO_CHANGE)
    op.execute(ALERT_GUARD)
    for table in ("reference_samples", "monitoring_reports", "alert_events"):
        op.execute(
            f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION immutable_rows_guard()"
        )
    op.execute(
        "CREATE TRIGGER alerts_guard BEFORE UPDATE OR DELETE ON alerts "
        "FOR EACH ROW EXECUTE FUNCTION alerts_guard()"
    )
    for table in TABLES_TRUNCATE:
        op.execute(
            f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
            "FOR EACH STATEMENT EXECUTE FUNCTION immutable_rows_guard()"
        )


def downgrade() -> None:
    op.drop_table("alert_events")
    op.drop_table("alerts")
    op.drop_table("monitoring_reports")
    op.drop_table("reference_samples")
    op.execute("DROP FUNCTION alerts_guard()")
    op.execute("DROP FUNCTION immutable_rows_guard()")
