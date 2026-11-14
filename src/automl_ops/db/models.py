"""SQLAlchemy models: the dataset catalog (P4-1), training jobs (P4-4), the registry (P4-5) and
its MLflow mirror state (P4-6). More tables arrive with the chunks that use them.

The catalog is an *index* of the versions in the file store, never a copy of their contents: it
holds identity (name, number, SHA-256), shape (rows, column names) and provenance fields from the
version's JSON record. The file store stays authoritative for bytes and hashes (D-028).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(constraint_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Same rules as `dataset_store.NAME_PATTERN`; PostgreSQL enforces them again.
NAME_REGEX = "^[a-z0-9][a-z0-9_-]{0,63}$"
SHA256_REGEX = "^[0-9a-f]{64}$"

# P4-4: the job lifecycle. Valid moves are queued -> running | cancelled and running -> a final
# state; a trigger (migration 0002) refuses every other change, and the repository checks first.
JOB_KINDS = ("train", "final")
JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled")
JOB_FINAL_STATES = ("succeeded", "failed", "cancelled")
JOB_PHASES = ("queued", "loading", "training", "selecting", "finalizing", "tracking", "done")
TRACKING_STATES = ("off", "logged", "pending")


# P4-5: the registry. Status changes happen only through `registry_events` (a trigger in migration
# 0003 applies each event to its version), so a status can never change without its audit event.
REGISTRY_STATES = ("development", "staging", "production", "archived")
REGISTRY_ACTIONS = ("register", "promote", "archive", "supersede", "rollback")
SYNC_ERROR_KINDS = ("error", "drift", "conflict")
SYNC_LOG_KINDS = ("sync", "reconcile")
SYNC_LOG_OUTCOMES = ("ok", "failed", "drift", "conflict", "partial")
PACKAGE_ID_REGEX = r"^m-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"

# P4-7: deployments. A row is one period during which one registered version was live; at most one
# row is `active`. Rows are never deleted and, once `ended`, never change (migration 0005 triggers).
DEPLOYMENT_STATES = ("active", "ended")
DEPLOYMENT_ACTIONS = ("deploy", "switch", "rollback")
DEPLOYABLE_STATES = ("staging", "production")  # registry statuses a version may be deployed from


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("name", name="name"),
        CheckConstraint(f"name ~ '{NAME_REGEX}'", name="name_format"),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    versions: Mapped[list[DatasetVersionRow]] = relationship(back_populates="dataset")


class DatasetVersionRow(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "number", name="dataset_number"),
        UniqueConstraint("dataset_id", "sha256", name="dataset_sha256"),
        CheckConstraint("number > 0", name="number_positive"),
        CheckConstraint(f"sha256 ~ '{SHA256_REGEX}'", name="sha256_format"),
        CheckConstraint(f"original_sha256 ~ '{SHA256_REGEX}'", name="original_sha256_format"),
        CheckConstraint("size_bytes >= 0 AND original_size_bytes >= 0", name="sizes"),
        CheckConstraint("rows >= 0 AND trailing_blank_lines_ignored >= 0", name="counts"),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="RESTRICT"))
    number: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))  # identity of the stored bytes
    original_sha256: Mapped[str] = mapped_column(String(64))  # provenance of the first upload
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    original_size_bytes: Mapped[int] = mapped_column(BigInteger)
    rows: Mapped[int] = mapped_column(Integer)
    column_names: Mapped[list[Any]] = mapped_column(JSONB)  # names only, never cell values
    source_filename: Mapped[str] = mapped_column(Text)  # information only, never used in a path
    original_encoding: Mapped[str] = mapped_column(String(64))
    original_delimiter: Mapped[str] = mapped_column(String(8))
    reencoded: Mapped[bool] = mapped_column(Boolean)
    trailing_blank_lines_ignored: Mapped[int] = mapped_column(Integer)
    store_format: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # from the file record
    catalogued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    dataset: Mapped[Dataset] = relationship(back_populates="versions")


class TrainingJob(Base):
    """One training job (P4-4). The worker is the only writer after the job is claimed."""

    __tablename__ = "training_jobs"
    __table_args__ = (
        CheckConstraint(_in("kind", JOB_KINDS), name="kind"),
        CheckConstraint(_in("status", JOB_STATES), name="status"),
        CheckConstraint(_in("phase", JOB_PHASES), name="phase"),
        CheckConstraint(_in("tracking_state", TRACKING_STATES), name="tracking_state"),
        CheckConstraint(f"dataset_sha256 ~ '{SHA256_REGEX}'", name="dataset_sha256_format"),
        CheckConstraint("kind <> 'final' OR confirm_final_test", name="final_is_confirmed"),
        CheckConstraint(
            f"({_in('status', JOB_FINAL_STATES)}) = (finished_at IS NOT NULL)", name="finished"
        ),
        CheckConstraint(
            "status <> 'running' OR (started_at IS NOT NULL AND worker_id IS NOT NULL)",
            name="running_is_claimed",
        ),
        CheckConstraint("status <> 'failed' OR error_code IS NOT NULL", name="failed_has_error"),
        Index("ix_training_jobs_status_id", "status", "id"),
        # At most one running job, enforced by the database itself: every running row has the
        # same key, so a second one violates the unique index (even with two workers).
        Index(
            "uq_training_jobs_one_running",
            "status",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    kind: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(10), server_default="queued")
    phase: Mapped[str] = mapped_column(String(12), server_default="queued")
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT")
    )
    dataset_sha256: Mapped[str] = mapped_column(String(64))  # the bytes the job was submitted for
    config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    track: Mapped[bool] = mapped_column(Boolean, server_default="false")
    confirm_final_test: Mapped[bool] = mapped_column(Boolean, server_default="false")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, server_default="false")
    progress: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    package_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tracking_state: Mapped[str] = mapped_column(String(8), server_default="off")
    tracking_job_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tracking_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class JobEvent(Base):
    """One progress or phase event of a job, in order (`seq` counts from 1 per job)."""

    __tablename__ = "job_events"
    __table_args__ = (
        UniqueConstraint("job_id", "seq", name="job_seq"),
        CheckConstraint("seq > 0", name="seq_positive"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id", ondelete="RESTRICT"))
    seq: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(24))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RegisteredModel(Base):
    """A named model (P4-5). `last_event_seq` allocates the per-model event numbers."""

    __tablename__ = "registered_models"
    __table_args__ = (
        UniqueConstraint("name", name="name"),
        CheckConstraint(f"name ~ '{NAME_REGEX}'", name="name_format"),
        CheckConstraint("last_event_seq >= 0", name="seq_not_negative"),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_event_seq: Mapped[int] = mapped_column(Integer, server_default="0")


class ModelVersion(Base):
    """One registered version of a model: a sealed package, plus a snapshot of its evidence."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version", name="model_version"),
        UniqueConstraint("package_id", name="package"),
        CheckConstraint("version > 0", name="version_positive"),
        CheckConstraint(_in("status", REGISTRY_STATES), name="status"),
        CheckConstraint(f"package_id ~ '{PACKAGE_ID_REGEX}'", name="package_id_format"),
        CheckConstraint(f"selection_sha256 ~ '{SHA256_REGEX}'", name="selection_sha256_format"),
        # At most one production version per model, enforced by the database itself.
        Index(
            "uq_model_versions_one_production",
            "model_id",
            unique=True,
            postgresql_where=text("status = 'production'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("registered_models.id", ondelete="RESTRICT"))
    version: Mapped[int] = mapped_column(Integer)
    package_id: Mapped[str] = mapped_column(String(64))
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_jobs.id", ondelete="RESTRICT"), nullable=True
    )
    dataset_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(12), server_default="development")
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    algorithm: Mapped[str] = mapped_column(String(64))
    task: Mapped[str] = mapped_column(String(16))
    target: Mapped[str] = mapped_column(Text)
    selection_sha256: Mapped[str] = mapped_column(String(64))
    evaluation: Mapped[dict[str, Any]] = mapped_column(JSONB)  # from the sealed test_result.json


class RegistryEvent(Base):
    """The append-only audit log. Inserting an event applies it to the version's status."""

    __tablename__ = "registry_events"
    __table_args__ = (
        UniqueConstraint("model_id", "seq", name="model_seq"),
        CheckConstraint("seq > 0", name="seq_positive"),
        CheckConstraint(_in("action", REGISTRY_ACTIONS), name="action"),
        CheckConstraint(_in("to_status", REGISTRY_STATES), name="to_status"),
        CheckConstraint(
            f"from_status IS NULL OR {_in('from_status', REGISTRY_STATES)}", name="from_status"
        ),
        CheckConstraint("char_length(reason) BETWEEN 1 AND 500", name="reason_length"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("registered_models.id", ondelete="RESTRICT"))
    version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    seq: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(12))
    from_status: Mapped[str | None] = mapped_column(String(12), nullable=True)
    to_status: Mapped[str] = mapped_column(String(12))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RegistrySync(Base):
    """Per-model mirror progress (P4-6). The registry is authoritative; `synced_seq` is the highest
    event number whose *complete* desired mirror state was verified in MLflow, and `synced_state`
    is that state. A model is pending while `registered_models.last_event_seq > synced_seq`."""

    __tablename__ = "registry_sync"
    __table_args__ = (
        CheckConstraint("synced_seq >= 0", name="seq_not_negative"),
        CheckConstraint("attempts >= 0", name="attempts_not_negative"),
        CheckConstraint(
            f"last_error_kind IS NULL OR {_in('last_error_kind', SYNC_ERROR_KINDS)}",
            name="error_kind",
        ),
    )

    model_id: Mapped[int] = mapped_column(
        ForeignKey("registered_models.id", ondelete="RESTRICT"), primary_key=True
    )
    synced_seq: Mapped[int] = mapped_column(Integer, server_default="0")
    synced_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    attempt_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error_kind: Mapped[str | None] = mapped_column(String(10), nullable=True)


class RegistrySyncLog(Base):
    """Append-only audit of mirror syncs that changed something or failed, and of reconciles."""

    __tablename__ = "registry_sync_log"
    __table_args__ = (
        CheckConstraint(_in("kind", SYNC_LOG_KINDS), name="kind"),
        CheckConstraint(_in("outcome", SYNC_LOG_OUTCOMES), name="outcome"),
        CheckConstraint(
            "kind <> 'reconcile' OR (reason IS NOT NULL AND char_length(reason) BETWEEN 1 AND 500)",
            name="reconcile_reason",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("registered_models.id", ondelete="RESTRICT"))
    kind: Mapped[str] = mapped_column(String(10))
    outcome: Mapped[str] = mapped_column(String(10))
    seq: Mapped[int] = mapped_column(Integer)  # the registry event number this pass targeted
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Deployment(Base):
    """One period in which one registered version was the live model (P4-7)."""

    __tablename__ = "deployments"
    __table_args__ = (
        CheckConstraint(_in("status", DEPLOYMENT_STATES), name="status"),
        CheckConstraint(_in("action", DEPLOYMENT_ACTIONS), name="action"),
        CheckConstraint("char_length(reason) BETWEEN 1 AND 500", name="reason_length"),
        CheckConstraint(
            "end_reason IS NULL OR char_length(end_reason) BETWEEN 1 AND 500",
            name="end_reason_length",
        ),
        CheckConstraint(f"package_id ~ '{PACKAGE_ID_REGEX}'", name="package_id_format"),
        CheckConstraint(
            "(status = 'active' AND ended_at IS NULL AND end_reason IS NULL) OR "
            "(status = 'ended' AND ended_at IS NOT NULL AND end_reason IS NOT NULL)",
            name="end_consistent",
        ),
        CheckConstraint(
            "(action = 'deploy' AND replaced_id IS NULL) OR "
            "(action <> 'deploy' AND replaced_id IS NOT NULL)",
            name="replaced_consistent",
        ),
        # At most one active deployment: every active row has the same index key.
        Index(
            "uq_deployments_one_active",
            text("(true)"),
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("registered_models.id", ondelete="RESTRICT"))
    version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    package_id: Mapped[str] = mapped_column(String(64))  # copy of the version's package, immutable
    status: Mapped[str] = mapped_column(String(8), server_default="active")
    action: Mapped[str] = mapped_column(String(10))
    reason: Mapped[str] = mapped_column(Text)
    replaced_id: Mapped[int | None] = mapped_column(
        ForeignKey("deployments.id", ondelete="RESTRICT"), nullable=True
    )
    deployed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class Prediction(Base):
    """One served prediction (P4-7). Written by the serving app; never changed afterwards."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("latency_ms >= 0", name="latency_not_negative"),
        Index("ix_predictions_deployment_created", "deployment_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID text, made by the server
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id", ondelete="RESTRICT"))
    model_version: Mapped[int] = mapped_column(Integer)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONB)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB)
    latency_ms: Mapped[float] = mapped_column(Float)
    is_simulated: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    """A ground-truth label for one logged prediction (P4-8). Append-only: a label is never changed
    or removed, so a later, different label for the same prediction is refused, not applied."""

    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("prediction_id", name="prediction"),
        CheckConstraint("char_length(true_value) BETWEEN 1 AND 200", name="true_value_length"),
        Index("ix_feedback_deployment_id", "deployment_id", "id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    prediction_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("predictions.id", ondelete="RESTRICT")
    )
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id", ondelete="RESTRICT"))
    model_version: Mapped[int] = mapped_column(Integer)  # copied from the prediction, checked
    true_value: Mapped[str] = mapped_column(Text)  # class label, or the canonical number text
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
