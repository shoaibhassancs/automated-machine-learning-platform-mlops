.PHONY: lint test typecheck check db-up db-stop api openapi worker serve

lint:
	uv run ruff check .

test:
	uv run pytest

typecheck:
	uv run mypy

check: lint typecheck test

# One-time demo-data preparation (Step 2). Downloads from archive.ics.uci.edu only if
# data/raw/ lacks the files, then writes data/demo/. Re-run safely; output is reproducible.
.PHONY: demo-data
demo-data:
	uv run python scripts/prepare_demo_data.py

# Local PostgreSQL for the application database (P4-1): needs .env with POSTGRES_PASSWORD.
# `db-stop` stops the container and keeps its data volume.
db-up:
	docker compose up -d --wait postgres
db-stop:
	docker compose stop postgres

# Control-plane API (P4-3) on 127.0.0.1:8100. Needs `make db-up` and `scripts/db.py upgrade` first;
# it refuses to start if the database is unreachable or behind. `make openapi` rewrites
# docs/openapi-control.json (a test fails if the checked-in file is stale).
api:
	uv run python scripts/serve_control.py
openapi:
	uv run python scripts/export_openapi.py

# Serving app (P4-7) on 127.0.0.1:8200: serves the active deployment. Same start-up checks as `make api`.
serve:
	uv run python scripts/serve_model.py

# Job worker (P4-4): one process, one job at a time. Needs `make db-up` and the migration head.
worker:
	uv run python scripts/worker.py
