"""Dataset catalog: an index of the versions in the file store (P4-1).

Revision ID: 0001
Revises: (none)
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

NAME_REGEX = "^[a-z0-9][a-z0-9_-]{0,63}$"
SHA256_REGEX = "^[0-9a-f]{64}$"


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_datasets")),
        sa.UniqueConstraint("name", name=op.f("uq_datasets_name")),
        sa.CheckConstraint(f"name ~ '{NAME_REGEX}'", name=op.f("ck_datasets_name_format")),
    )
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("original_sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("original_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("column_names", postgresql.JSONB(), nullable=False),
        sa.Column("source_filename", sa.Text(), nullable=False),
        sa.Column("original_encoding", sa.String(length=64), nullable=False),
        sa.Column("original_delimiter", sa.String(length=8), nullable=False),
        sa.Column("reencoded", sa.Boolean(), nullable=False),
        sa.Column("trailing_blank_lines_ignored", sa.Integer(), nullable=False),
        sa.Column("store_format", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "catalogued_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_versions")),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name=op.f("fk_dataset_versions_dataset_id_datasets"),
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "dataset_id", "number", name=op.f("uq_dataset_versions_dataset_number")
        ),
        sa.UniqueConstraint(
            "dataset_id", "sha256", name=op.f("uq_dataset_versions_dataset_sha256")
        ),
        sa.CheckConstraint("number > 0", name=op.f("ck_dataset_versions_number_positive")),
        sa.CheckConstraint(
            f"sha256 ~ '{SHA256_REGEX}'", name=op.f("ck_dataset_versions_sha256_format")
        ),
        sa.CheckConstraint(
            f"original_sha256 ~ '{SHA256_REGEX}'",
            name=op.f("ck_dataset_versions_original_sha256_format"),
        ),
        sa.CheckConstraint(
            "size_bytes >= 0 AND original_size_bytes >= 0", name=op.f("ck_dataset_versions_sizes")
        ),
        sa.CheckConstraint(
            "rows >= 0 AND trailing_blank_lines_ignored >= 0",
            name=op.f("ck_dataset_versions_counts"),
        ),
    )


def downgrade() -> None:
    op.drop_table("dataset_versions")
    op.drop_table("datasets")
