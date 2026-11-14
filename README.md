# AutoML Ops

A single-user, local platform that manages the machine learning lifecycle for tabular CSV data: dataset
validation, training with cross-validation, experiment tracking (MLflow), a model registry, REST deployment,
live feedback and drift monitoring with alerts. Solo portfolio project. Scope: [SCOPE.md](SCOPE.md).
Local only, no authentication, not production-ready.

## Status

Built and verified (backend): dataset ingestion, profiling and validation; task configuration; training,
selection, one final test and sealed model packages; MLflow tracking; PostgreSQL database and migrations;
control-plane API; training jobs with a worker; model registry with an optional MLflow mirror; serving app
and deployments; feedback and live performance; drift monitoring, reference samples and alerts; the traffic
simulator and a backend end-to-end script (P4-10).

**Not built yet:** the web frontend (Phase 5), the full Docker Compose stack, CI and the final documentation
(Phase 6). Only PostgreSQL runs in Compose today; the other services run with `uv run`. Known limits:
[docs/simulator.md](docs/simulator.md#limits) and [docs/monitoring.md](docs/monitoring.md#limits).

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.12 and Docker (for PostgreSQL).

```bash
uv sync                                  # environment and dependencies from the lockfile
cp .env.example .env                     # then set POSTGRES_PASSWORD
make db-up                               # PostgreSQL on 127.0.0.1:55432
uv run python scripts/db.py upgrade      # apply migrations (you run this; nothing else does)
make demo-data                           # one-time: downloads the UCI data only if data/raw/ lacks it
make check                               # lint, type-check and the whole test suite
```

## Run it

Four terminals (all bind to 127.0.0.1):

```bash
make api       # control plane, :8100      (datasets, jobs, registry, deployments, monitoring, alerts)
make serve     # serving app, :8200        (POST /predict, /feedback; serves the active deployment)
make worker    # trains jobs one at a time (set AUTOML_JOBS_MLFLOW_DIR for tracked jobs)
make e2e       # the backend story on the Abalone demo data, then read the JSON it prints
```

`make e2e` (`scripts/e2e_demo.py`) uploads the training pool, trains and tracks a final job, registers,
promotes and deploys the model, then replays the monitoring stream as simulated traffic (normal, shifted,
normal) with feedback and monitoring after each phase. **It writes real rows and a model package to the stack's
database and folders.** `make simulate` replays traffic against an already deployed model. See
[docs/simulator.md](docs/simulator.md).

## Documentation

[docs/api.md](docs/api.md) (control plane) · [docs/serving.md](docs/serving.md) ·
[docs/monitoring.md](docs/monitoring.md) · [docs/registry.md](docs/registry.md) ·
[docs/jobs.md](docs/jobs.md) · [docs/tracking.md](docs/tracking.md) · [docs/database.md](docs/database.md) ·
[docs/data.md](docs/data.md) (data sources, licences, the synthetic shifted copies) ·
[docs/simulator.md](docs/simulator.md). OpenAPI files: `docs/openapi-control.json`,
`docs/openapi-serving.json` (`make openapi`; a test fails if they are stale).

## Things to know

- The held-out test part is read once per job, for the selected model only; reuse across jobs is an **open
  decision**, so every reported test result carries that limitation.
- Monitoring uses numpy-only statistics (Evidently was not installed) and runs on demand.
- Simulated traffic is always labelled simulated and is kept apart from real traffic.
