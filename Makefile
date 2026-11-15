.PHONY: e2e-compose full-env full-build full-up full-down full-status lint test typecheck check db-up db-stop api openapi worker serve simulate e2e ui-install ui-dev ui-demo ui-check ui-e2e

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

# Traffic simulator (P4-10): normal, shifted, then normal demo traffic against `make api` + `make serve`
# (needs a deployed model); see docs/simulator.md. `make e2e` runs the whole backend story on a THROWAWAY stack it
# creates and removes itself (new database, temp folders); see scripts/e2e_isolated.py.
simulate:
	uv run python scripts/simulate_traffic.py demo abalone
e2e:
	uv run python scripts/e2e_isolated.py abalone

# Frontend (Phase 5, frontend/). `ui-dev` serves http://localhost:5173, the one browser origin the API
# allows by default, and talks to `make api` / `make serve`. `ui-demo` builds a THROWAWAY seeded backend
# (new database, temp folders, SQLite MLflow; removed on Ctrl-C) plus the UI, so the real data is never used.
# `ui-e2e` runs the Playwright suite on two throwaway stacks it creates and removes itself.
ui-install:
	cd frontend && npm ci

ui-dev:
	cd frontend && npm run dev

ui-demo:
	uv run python scripts/ui_stack.py --seed demo --with-ui --api-port 18100 --serve-port 18200

ui-check:
	cd frontend && npm run lint && npm run format:check && npm run typecheck && npm run check:api && npm test && npm run build

ui-e2e:
	cd frontend && npm run e2e

# Full local stack in Docker (Phase 6, compose.full.yaml): a separate Compose project from `make db-up`.
# Local only, 127.0.0.1 only; see docs/compose.md. `full-down` keeps the volumes (and so the data).
FULL = docker compose -f compose.full.yaml --env-file .env.full
full-env:
	@test ! -e .env.full || { echo ".env.full exists; not overwriting"; exit 1; }
	@uv run python -c "import secrets,pathlib; pathlib.Path('.env.full').write_text('AUTOML_FULL_PASSWORD=' + secrets.token_urlsafe(24) + '\\nAUTOML_FULL_E2E_TOKEN=' + secrets.token_hex(16) + '\\n')"
	@chmod 600 .env.full && echo "wrote .env.full (git-ignored)"
full-build:
	$(FULL) build
full-up:
	$(FULL) up -d --wait
full-down:
	$(FULL) down
full-status:
	$(FULL) ps

# The end-to-end story against the running full stack (`make full-up`); refuses unless that stack
# proves its identity and is empty. See docs/compose.md.
e2e-compose:
	uv run python scripts/e2e_compose.py abalone
