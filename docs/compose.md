# The full local stack in Docker (Phase 6, Steps 33-35)

`compose.full.yaml` runs the whole platform on this machine: PostgreSQL, the control-plane API, the
job worker, the serving app and the UI. It is **local delivery only**. Nothing is hosted, no image is
pushed anywhere, and every published port is bound to `127.0.0.1`. It is not a production
deployment and the project makes no claim that it is one (single user, no authentication).

It is a **separate Compose project** from `compose.yaml` (the development database behind
`make db-up`): its own name (`automl-ops-full`), containers (`automl-full-*`), volumes, database
(`automl_compose`) and ports. Starting or stopping one never touches the other.

```
make demo-data     # once: prepares data/demo (downloads from archive.ics.uci.edu if data/raw is empty)
make full-env      # once: writes the git-ignored .env.full with a random database password and run token
make full-up       # builds the images if they are missing, starts everything, waits until healthy
make e2e-compose   # optional: runs the whole lifecycle story against the running stack (needs it empty)
make full-down     # stops and removes the containers; the volumes (the data) are kept
```

The first `make full-up` builds three images; that needs the base images (`python:3.12-slim`,
`node:22-alpine`, `nginx:stable-alpine`, `postgres:17-alpine`) and downloads the pinned Python and npm
packages. After that it runs offline. `docker compose up` starts **services only**: it prepares no
data and runs no workflow. Use the UI at <http://localhost:25173>, or `make e2e-compose`.

| Service | Image | Port on this machine | Memory limit |
| --- | --- | --- | --- |
| `postgres` | `postgres:17-alpine` (the one already used for development) | none (not published) | 384 MB |
| `migrate` | backend, runs once | none | n/a (exits) |
| `api` | backend (`automl-ops-backend:local`) | 127.0.0.1:28100 | 512 MB |
| `worker` | backend, one job at a time | none | 1 GB |
| `serving` | serving (`automl-ops-serving:local`), no MLflow, no Pandera | 127.0.0.1:28200 | 512 MB |
| `ui` | frontend (`automl-ops-frontend:local`), nginx, static files | 127.0.0.1:25173 | 64 MB |

## Design choices (and why)

- **MLflow has no container of its own.** Tracking runs inside the worker with the SQLite backend on
  the data volume (`/data/mlflow`), the same default as development. The MLflow UI stays the
  existing manual script (`scripts/mlflow_ui.py`). One fewer service for a 4 GB Docker VM.
- **Serving is its own container, reached by name.** The control plane calls it at `serving:28200` on
  the Compose network (setting `AUTOML_SERVING_HOST`, accepted only inside the project's image and only
  as a bare service name; outside Docker it is always 127.0.0.1). The request still carries the
  loopback `Host` header, so the serving app's strict Host check is unchanged. Either container can
  restart on its own. (The first version shared the API container's network; the Phase 6 review showed
  that left serving unreachable after an API restart, so it was replaced. Tested: D-054.)
- **Listening on all interfaces inside the container, and only there.** The scripts still refuse any
  address but loopback unless `AUTOML_IN_CONTAINER=1` is set (the images set it) **and** Docker's
  `/.dockerenv` file exists (`src/automl_ops/bind.py`). The Compose file then publishes each port on
  the host's `127.0.0.1` only.
- **Non-root.** The API, worker and serving containers run as uid 10001 and the UI as `nginx`
  (uid 101). PostgreSQL's official image starts as root and drops to its own user, as upstream
  designs it. Base images are pinned by tag, not by digest.
- **Migrations are a one-shot step.** `migrate` runs `scripts/db.py upgrade` against this stack's own
  empty database before `api` and `worker` start. The development database is never involved; you
  still run `scripts/db.py upgrade` yourself for that one.
- **Pinned, hash-checked installs.** `docker/requirements-backend.txt` and
  `docker/requirements-serving.txt` are generated from `uv.lock` by
  `scripts/export_requirements.py` (a test fails if they are stale). The serving set has 33
  packages; the backend set 98.
- **The UI has the API addresses built in** (`VITE_API_URL`, `VITE_SERVING_URL`, defaults
  127.0.0.1:28100 and :28200). Changing a published port means rebuilding that image.

## Safety: the end-to-end script and this stack

`make e2e-compose` writes data, so it applies the same refusal rules as `make e2e`
(`docs/simulator.md`, D-047), with the stack's identity stated differently. Before any write it
checks that both services report isolation, that they use the database the Compose file declares and
the run token in `.env.full`, and that the control plane is empty. The `automl_compose` database
counts as isolated only inside the project's image; `automl_ops`, `mlflow` and `postgres` never do.
A second run is refused because the stack is no longer empty (use a fresh volume for another run).

## Measured on the development machine (MacBook Air M1, Docker VM about 4 GB)

These are single measurements, not benchmarks.

- Image sizes: backend 1.37 GB, serving 1.01 GB, frontend 93.5 MB (shared base layers).
- Cold start from empty volumes (`make full-up`, images already built): about 22 seconds until all
  services were healthy.
- Memory while idle and after the end-to-end story: api about 196 MB, worker about 262 MB, serving
  about 181 MB, postgres about 72 MB, ui about 3 MB.
- `make e2e-compose` (abalone): all 9 stages passed, about 9.5 seconds in total.

## Limits and what has not been tested

- Restart policy: tested by making each service's own process exit (`kill -TERM 1` inside the
  container, which Docker treats as an exit, not a manual stop): `api`, `serving` and `worker` each
  came back (restart count 1), and the control plane reported serving `in_sync` again. A manual
  `docker kill` is treated as a manual stop and is not restarted. Not tested: an out-of-memory kill,
  a Docker daemon restart, a host reboot.
- The end-to-end story ran on both demo datasets on the stack: Abalone (2,924 training rows) and Adult
  (34,190 training rows, run on a fresh project `automl-ops-adult` with its own volumes). Sampled worker
  memory stayed near 300 MB during Adult (24 samples, so not a true peak; no OOM kill).
- Only Docker Desktop on an Apple-silicon Mac (arm64) was used. Linux hosts, x86-64 and Windows
  were not run. Building on another architecture was not tried.
- The images are for local use; there is no vulnerability scan of the base images in this project.
- The Compose project keeps its data in the volumes `automl-ops-full_full_postgres_data` and
  `automl-ops-full_full_data`. `make full-down` keeps them. To start again from nothing, remove those
  two volumes yourself (`docker volume rm ...`); the project never does it for you.
