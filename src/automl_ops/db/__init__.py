"""Application database (BUILD_PLAN Step 17, chunk P4-1): PostgreSQL via SQLAlchemy and Alembic.

Responsibilities (D-028): dataset *bytes* stay in the file store; MLflow keeps experiment records;
this database keeps mutable platform state. In P4-1 it holds only the dataset catalog, an index of
the stored versions. Nothing here imports `mlflow`.
"""
