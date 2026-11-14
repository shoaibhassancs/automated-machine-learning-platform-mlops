# Application database (PostgreSQL)

BUILD_PLAN Step 17, chunk **P4-1** (D-028, D-029). Code: `src/automl_ops/db/` (`config`, `models`,
`migrate`, `catalog`, `migrations/`); commands: `scripts/db.py`; server: `compose.yaml`.

## What lives where

| Store | Holds | Authoritative for |
| --- | --- | --- |
| Files (`data/store`, `data/models`) | dataset version bytes and records; sealed model packages | bytes, SHA-256, version numbers, seals |
| MLflow (SQLite in `data/mlflow`) | runs, parameters, metrics, artifacts | experiment records |
| PostgreSQL `automl_ops` (this chunk) | the **dataset catalog**, an index of the file store | nothing about bytes |

Later chunks add jobs, registry, deployments, predictions and monitoring tables, each with its own
migration. Nothing in P4-1 touches MLflow (it stays on SQLite until P4-2 proves a move is sound).
The catalog holds identity (name, number, SHA-256), shape (rows, column *names*) and provenance
fields from each version's JSON record. It never holds cell values or file bytes (a test checks
the columns and the stored rows).

## Start, migrate, check

```bash
cp .env.example .env            # set POSTGRES_PASSWORD (the .env file is git-ignored)
make db-up                      # docker compose up -d --wait postgres   (make db-stop keeps the data)
uv run python scripts/db.py upgrade
uv run python scripts/db.py status [--json]
uv run python scripts/db.py check  [--json]
uv run python scripts/db.py sync [--store data/store] [--json]
```

* **Server:** `postgres:17-alpine` from the local image cache (`pull_policy: never`, so starting it
  contacts no registry), container `automl-ops-postgres`, bound to `127.0.0.1:55432` only, named
  volume `automl-ops_automl_ops_pgdata`, memory limit 384 MB. Other projects' containers, the
  Homebrew server (5432) and ports 8000, 8080, 8501 and 5433 are not used.
* **Settings:** `POSTGRES_PASSWORD` (environment, else `.env`); optional `AUTOML_DB_HOST`,
  `AUTOML_DB_PORT`, `AUTOML_DB_USER`, `AUTOML_DB_NAME`, or a full `AUTOML_DATABASE_URL`. The driver is
  psycopg 3. The password is never printed (URLs are shown with `***`).
* **Exit codes:** 0 ok; 1 `status` not up to date, `check` or `sync` found problems; 2 cannot
  connect or bad settings.

## MLflow's own database (P4-2b)

MLflow's tracking records live in a **separate database, `mlflow`**, on the same server; MLflow owns
its schema and migrations (they share nothing with the Alembic history above). It is created only by
`uv run python scripts/db.py mlflow-init` (idempotent; also creates the experiment) and inspected by
`mlflow-status` (read-only). Settings: `AUTOML_MLFLOW_DB_NAME` (default `mlflow`),
`AUTOML_MLFLOW_ARTIFACT_ROOT` (default `data/mlflow-pg/artifacts`) and the credentials above.
Details, the 200-character artifact URI limit, the archive and the migration: `docs/tracking.md`.

## Migrations

Alembic, with the scripts inside the package (`db/migrations/versions/`). They run **only** on an
explicit `db upgrade`; no import and no application start applies them. An upgrade holds a
PostgreSQL advisory lock (two simultaneous upgrades serialise, the second is a no-op), and every
migration is one transaction (PostgreSQL DDL is transactional). `downgrade` exists as a function for
tests; the command line deliberately does not offer it. Revision `0001` creates `datasets` and
`dataset_versions` with named constraints (`pk_`, `uq_`, `ck_`, `fk_`).

`db check` confirms the database is at the head and that its structure equals the models: Alembic's
comparison (tables, columns, types, unique constraints) plus every CHECK constraint by name. It
found a real bug during this chunk: the migration's explicit constraint names had been prefixed a
second time by the naming convention; the migration now marks them final (`op.f`), and a test pins
the exact names.

## The dataset catalog

* `datasets(name)` and `dataset_versions(dataset_id, number, sha256, ...)`. PostgreSQL enforces the
  name pattern, lower-case 64-character hex hashes, positive numbers, non-negative sizes, one row
  per `(dataset, number)` and per `(dataset, sha256)`, and a dataset with versions cannot be
  deleted.
* **Written after the file.** `CatalogedDatasetStore` (a `DatasetStore`) stores a version first and
  then records it. If the catalog write fails the version is still stored (`unsynced` lists it) and
  `db sync` repairs it. The existing `ingest` command is unchanged and does not catalog; use
  `db sync`. (Wiring the store into an API is P4-3.)
* **`db sync` is idempotent and read-only for files.** It catalogs every version whose files pass
  `verify`. It never deletes or changes a row: a row with no file is an **orphan**, a row whose hash
  differs from the file is a **conflict**, a version whose file fails verification is **unverified**;
  all three are reported and make the exit code 1.

## Tests

`tests/test_db_config.py` needs no server (settings, scripts, models, isolation, no password
leak). `tests/test_db_postgres.py` (marker `postgres`) creates an empty database per test and drops
only that one. **If the server is unreachable these tests are skipped and pytest says
`NOT RUN: PostgreSQL not reachable ...`; a skip is never a pass.** They cover up, down and up,
concurrent upgrades and syncs, constraint violations, drift detection, sync (rebuild, idempotence,
orphan, conflict, unverified, no change to the store), the cataloged store and a failing catalog,
the command line, and the real demo store.

## Limits

Single local server, no authentication beyond the password, localhost only. Recovery of a lost
volume: the catalog is rebuilt by `db sync`; later chunks' registry history will not be, so a
`pg_dump` habit will matter then. Verified on this machine with PostgreSQL 17 and the locked
psycopg 3.3.8, SQLAlchemy 2.1.8 and Alembic 1.20.1 only.
