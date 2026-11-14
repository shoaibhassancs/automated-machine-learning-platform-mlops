# Control-plane API (P4-3, BUILD_PLAN Step 18)

A FastAPI app over the existing dataset modules. Local, single-user, **no authentication**:
it binds to 127.0.0.1 only. Code: `src/automl_ops/api_control/`; start it with `make api`
(`scripts/serve_control.py`, default `127.0.0.1:8100`). The OpenAPI document is checked in as
`docs/openapi-control.json` (`make openapi` rewrites it; a test fails if it is stale).

It needs the application PostgreSQL (`make db-up`) at the migration head
(`scripts/db.py upgrade`; migrations never run on their own). **It refuses to start otherwise**,
with a message that names the fix. Settings (environment or the git-ignored `.env`):
`AUTOML_API_PORT` (8100), `AUTOML_UI_ORIGIN` (`http://localhost:5173`), `AUTOML_STORE_DIR`
(`data/store`).

## Endpoints

Datasets are addressed by name and version number (`abalone@v1` is `/datasets/abalone/versions/1`).

| Method and path | What it does |
| --- | --- |
| `GET /health` | 200 `ok`, or 503 `degraded` if the database is unreachable or not at the head; also `worker`: `running`, `stopped` or `unknown` (is a job worker holding its lock) |
| `POST /datasets` | Upload a CSV (multipart: `name`, `file`, optional `add_version`). 201 if a version was stored, 200 if identical content already existed |
| `GET /datasets` | Every dataset with its versions |
| `GET /datasets/{name}` | One dataset's versions |
| `GET /datasets/{name}/versions/{n}` | One version's record |
| `GET .../profile` | Column profile (the CLI `profile --json` output) |
| `GET .../validation?target=` | Rule-based findings (`validate --json`) |
| `GET .../preview?target=&task=` | Sample rows, histograms, correlation, class balance (`explore --json`) |
| `POST .../task-check` | Body `{target, task?, excluded[]}`; the task check (`task --json`) |
| `GET /algorithms?task=` | The algorithm catalogue with parameter specs (P4-4) |
| `POST /jobs` | Queue a `train` or `final` job: 202 (P4-4, see `docs/jobs.md`) |
| `GET /jobs`, `GET /jobs/{id}` | List jobs; one job with progress events and, when done, its result |
| `POST /jobs/{id}/cancel` | 200 a queued job cancelled, 202 a running job will stop, 409 too late or finished |

| `POST /models`, `GET /models`, `GET /models/{name}[/versions[/{n}]]` | Register a sealed package; list models and versions; one version with its stored evaluation summary (P4-5, `docs/registry.md`) |
| `POST /models/{name}/versions/{n}/promote`, `.../archive`, `POST /models/{name}/rollback`, `GET /models/{name}/events` | Registry moves and the append-only history |

| `GET /registry/sync`, `GET\|POST /models/{name}/sync`, `GET /models/{name}/mirror/check`, `POST /models/{name}/reconcile` | The opt-in MLflow mirror of the registry: standing, sync, read-only check, explicit audited reconcile (P4-6, `docs/registry.md`) |
| `POST /deployments`, `GET /deployments[/active\|/{id}]`, `POST /deployments/{id}/switch`, `POST /deployments/rollback`, `POST /deployments/{id}/undeploy` | Which registered version is live, its history, and the serving app's state after each change (P4-7, `docs/serving.md`) |
| `GET /deployments/{id}/performance`, `GET /deployments/{id}/feedback` | Live performance from labelled predictions only, with the offline summary shown separately; the label history (P4-8, `docs/serving.md`). Labels are submitted to the serving app's `POST /feedback` |
| `POST\|GET /models/{name}/versions/{n}/reference`, `POST /deployments/{id}/monitor`, `GET /deployments/{id}/monitoring\|/reports`, `GET /reports/{id}`, `GET /alerts[/{id}]`, `POST /alerts/{id}/acknowledge\|/resolve` | Drift reference samples, on-demand monitoring runs and reports, alerts and their audit trail (P4-9, `docs/monitoring.md`) |

`task-check` is **stateless**: it stores nothing (D-029 removed the `task_configs` table, so the old
`PUT .../task` of ARCHITECTURE §5 does not exist; the task configuration will travel in the
job request, P4-4). Problems with a configuration are a normal 200 result with `ok: false`.

The upload response says `created`, `catalogued` (was the PostgreSQL catalog row written; if not,
the file is still stored and `scripts/db.py sync` repairs it), the version record, and how the
upload was converted. The file store stays authoritative for every read; the catalog is only an
index. Not included (not in Step 18): the feature-schema and dataset-verify endpoints.

The JSON of each report is **the same as the command line's `--json`** (one shared shaping layer,
`dataset_reports.py`); a test compares the two for the same inputs. The response models forbid extra
fields, so a field added to a report fails the contract tests instead of drifting silently.

## Errors

Every error, including 404, 405, validation failures, malformed JSON/multipart and oversized
bodies, has one shape: `{"error": {"code": "...", "message": "...", "details": {...}}}`
(`details` only for validation problems). Messages never contain tracebacks, file contents,
credentials or filesystem paths; unexpected errors are a generic 500 and the real error goes to the
server log.

| Status | Codes |
| --- | --- |
| 400 | `host_not_allowed`, `bad_request` |
| 403 | `origin_not_allowed`, `cors_not_allowed` |
| 404 | `not_found` (no such endpoint), `dataset_not_found`, `version_not_found` |
| 405 | `method_not_allowed` (with `Allow`) |
| 409 | `dataset_exists` (a new version needs `add_version`) |
| 413 | `body_too_large`, `too_large` (the CSV itself is over 50 MB) |
| 415 | `unsupported_media_type` |
| 422 | `validation_error`, `invalid_name`, `invalid_task_config`, and the ingest rejections (`empty_file`, `bad_encoding`, `bad_delimiter`, `too_many_columns`, `too_many_rows`, `ragged_row`, ...) |
| 500 | `store_error` (a stored version cannot be read or fails its hash check), `internal_error` |

## Local security rules (`security.py`, D-036)

1. **Host** must be exactly `127.0.0.1`, `localhost` or `[::1]` with the API's port (defeats DNS
   rebinding).
2. **Origin** on POST/PUT/PATCH/DELETE: if present it must equal the configured UI origin as a
   parsed origin (scheme, host, port). `null`, malformed, repeated and other origins are refused.
   No `Origin` (curl, scripts) is allowed. This stops a web page the user visits from POSTing a form
   to the API; it is **not** authentication.
3. **CORS** only for that one origin, no credentials; other preflights get a 403 in the error format.
4. **Content type**: `multipart/form-data` for the upload, `application/json` elsewhere.
5. **Size**, enforced while streaming (including chunked bodies and a lying `Content-Length`):
   upload body = the 50 MB CSV limit **plus 64 KiB** multipart allowance (the file itself is held to
   exactly the ingest limit); JSON bodies 1 MiB. The body is cut off at the limit, so nothing larger
   is spooled to disk or held in memory.

## Pagination, consistency and quality checks (P4-10, Step 26)

- **One contract for the lists that can grow:** `GET /jobs`, `/deployments`, `/deployments/{id}/reports`,
  `/deployments/{id}/feedback` and `/alerts` take `limit` (at most 100) and `before_id` and return
  `next_before_id` (newest first). `GET /jobs/{id}` and `GET /models/{name}/events` page forward with
  `after_seq`.
- **Lists that are deliberately not paged** (bounded in a single-user setup; a new unpaged list fails a test
  until it is added here): `/algorithms` (a fixed catalogue), `/datasets` (reads the file store),
  `/models` (one row per model) and `/models/{name}/versions` (one model's versions).
- **Tests** (`tests/test_api_quality.py`) check on both exported OpenAPI files that every operation has a
  summary and a unique id, that every error response uses the one error shape (only `/health` may also answer
  503 with its health body), that the paged lists share the contract above, and that the serving surface is
  exactly the six documented operations. A runtime sweep sends unknown, malformed and empty input to **every**
  operation of both apps (over 300 requests on the control plane): never a 500, every error an envelope, and
  nothing is written.
- **Coverage gaps, not hidden:** the sweep sends bad input, not every valid combination (those are in the
  per-area tests); the paths with a nonexistent id are covered, a full authorization model is not (there is
  none, by design); timing and memory of the biggest allowed windows were measured only on the demo data
  (`docs/simulator.md`), not at the limits.

## Concurrency and memory

Anything that reads a whole CSV (upload, profile, validation, preview, task-check) runs in a worker
thread behind a semaphore of 2, extra requests wait. Uploads are serialised (the file store is
single-writer). pandas reads every cell as text, so memory is much larger than the file: see the
measurements in `DECISIONS.md` (D-036). The ingest limits (50 MB, 100,000 rows, 100 columns) are
unchanged.

## What this does not cover

No authentication or authorisation (anyone who can reach 127.0.0.1:8100 on this machine can use
it), no rate limiting beyond the semaphore, no TLS, and no protection against a hostile local user
or process. Not production-ready.
