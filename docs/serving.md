# Serving and deployments (P4-7, BUILD_PLAN Steps 21 and 22) and feedback (P4-8, Step 23)

A **deployment** makes one registered model version the live model. A separate **serving app**
loads exactly that version and answers `POST /predict`. Local, single-user, no authentication,
127.0.0.1 only (see `docs/api.md`).

```
make db-up && uv run python scripts/db.py upgrade     # once; both apps refuse an old schema (0005)
make api                                              # control plane  127.0.0.1:8100
make serve                                            # serving app    127.0.0.1:8200 (AUTOML_SERVING_PORT)
```

The two are separate processes and separate applications. The serving app does not import the
control-plane app, MLflow, the worker or the registry mirror (a test checks this in a fresh
interpreter); it reuses the control plane's request rules (Host check, Origin check on POST, JSON
content type, body limit) and its error format. The control plane never loads a model for serving.

## What PostgreSQL decides

`deployments` (migration 0005) holds one row per period in which a version was live; at most one row
is `active` (a unique index). Rows are never deleted or truncated, and an ended row never changes:
the database refuses it even from a hand-written statement. The database also refuses to deploy a
version that is not `staging` or `production`, or a package that is not the version's own. The
registry's rules are untouched: archiving a deployed staging version, or a promotion that archives a
deployed production version, does not undeploy it. The deployment keeps serving until someone
switches or undeploys it, and `registry_status` shows the difference.

| Change | Endpoint | Result |
| --- | --- | --- |
| deploy | `POST /deployments` `{model, version, reason}` | needs no active deployment (else 409 `deployment_active`) |
| switch | `POST /deployments/{id}/switch` `{model, version, reason}` | `{id}` must be the active deployment; the old row ends and a new one starts in one transaction |
| rollback | `POST /deployments/rollback` `{reason}` | makes the version the active deployment replaced live again. Only a *switch* can be rolled back; a rollback is not undone by another rollback (409 `no_previous_deployment`) |
| undeploy | `POST /deployments/{id}/undeploy` `{reason}` | nothing is live afterwards |
| read | `GET /deployments` (newest first, `status`, `before_id`, `limit`), `GET /deployments/{id}`, `GET /deployments/active` | |

Every change deeply verifies the version's sealed package first (seal hashes, read-only files,
library versions, reload with prediction parity: the registry's own check), takes one advisory lock,
and re-checks the version row, so two simultaneous changes have exactly one winner (tested with 5-6
racers). Error codes: `model_not_found`, `version_not_found`, `deployment_not_found`,
`no_active_deployment` (404); `version_not_deployable`, `deployment_active`, `deployment_not_active`,
`already_deployed`, `no_previous_deployment`, `package_unverified`, `deployment_refused` (409);
`validation_error` (422); `deployments_unavailable` (503). Every change needs a reason of 1 to 500
characters; it is recorded.

## The serving app

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | The process is alive (always 200); `ready`, `model_state`, `database`, `desired_deployment_id`, `loaded_deployment_id`, `error` |
| `GET /ready` | 200 only while a verified model is being served, else 503 |
| `GET /model` | The served deployment and version, task, target, classes, the **input columns** (name and kind), `registry_status`, `max_rows` (100) |
| `POST /predict` | `{rows: [{column: value, ...}], simulated?: bool}`, 1 to 100 rows |
| `POST /feedback` | The true value for one `prediction_id` (P4-8, see Feedback below) |
| `POST /reload` | No input. Re-reads the active deployment from PostgreSQL and loads it. It cannot be told what to load; the control plane calls it after every change |

**Loading.** The app serves the one `active` deployment. The package is loaded through the same
`load_package` the rest of the platform uses (seal hashes, read-only files, library versions, the
model file's SHA-256 *before* it is deserialized), then its package id, task and frozen-selection
hash must equal the registry's record, and it must reproduce its saved reference predictions. Only
sealed packages in the platform's own models folder are ever loaded, never anything uploaded.

**Staying current.** A change is picked up when the control plane calls `/reload` (best effort),
within `poll_seconds` (5 s) by polling PostgreSQL, or at start-up. So a stopped serving app, or a
deployment change made while it is down, is not lost: it catches up from PostgreSQL alone. The
control plane reports `serving.state` after each change: `in_sync`, `out_of_sync` (the app answered
but is serving something else or could not load it, with the reason) or `unreachable`. A serving
problem never fails the deployment change, because PostgreSQL already holds it.

**What is served when something is wrong.** Only what PostgreSQL says is active, and only if it
verified:

- the active deployment fails to load or verify: `/predict` answers 503 `model_not_ready` and
  `/health` says why. The previous model is *not* kept as a stand-in. A failing load is retried at
  most every 30 s, or at once by `/reload`. Switch or roll back to recover.
- nothing is deployed: 503 `no_deployment`.
- the database becomes unreachable *after* a model was verified: that model keeps serving (nothing
  can be deployed or undeployed while the database is down); `/health` shows `database:
  unavailable` and prediction logging reports `logged: false`. This can be stale only for a change
  committed just before the outage.
- the database is not at migration head: not ready, with the `db upgrade` hint.

## Predictions

Input is checked against the model's feature schema before the model runs. A row has **exactly** the
model's input columns: a missing and an unknown column are both refused, naming the row and column.
`null` or an empty string is a missing value (the model imputes it); a numeric column takes a JSON
number or a clean number as text; a categorical column takes text; a value that is not a clean number
is refused, never turned into a missing value. An unseen category is not an error. Refusals are 422
`invalid_input` with up to 20 problems. Bad request shapes (no rows, more than 100, extra fields) are
422 `validation_error`.

The response gives each row its own `prediction_id`, the `prediction` (class label, or number for
regression), `probabilities` per class (classification), the deployment, model version and package,
the model time in `latency_ms` for the whole request, and `logged`.

**Logging.** Each row is written to `predictions` with its input, output, deployment, model version,
`latency_ms` (the request's model time) and `is_simulated` (the request's `simulated` flag, for
traffic that is not a real observation). A logged prediction is never changed. A write that fails or
takes over 2 s is reported as `logged: false`; the prediction is still returned. Stored inputs are
the caller's own rows (local only). Nothing here reads the held-out test part or any dataset.

## Verified

Real PostgreSQL and real sealed packages. Served predictions and probabilities equal the saved
pipeline's on the same rows (parity, classification and regression); wrong columns and bad values give
a clear 422; switching changes the model that answers and shows in `/model`; rollback and undeploy
work and are logged; races have one winner; DB rules refuse hand-written bad statements; a
corrupted-after-deployment package is not served and no stale model stands in. A live run with real
processes, a throwaway database and SIGKILL of each process is recorded in `DECISIONS.md` (D-043).

## Measured (this machine, one live run, small models)

| Item | Value |
| --- | --- |
| `/predict`, 1 row, over HTTP | median 14 ms, p95 22 ms (100 requests) |
| `/predict`, 100 rows | 20 ms |
| Deploy / switch (includes the deep package check and the reload call) | 0.07 s / 0.04 s |
| Serving app restart to ready | 1.6 s |
| Poll-only convergence after a change made behind the app's back | 4.2 s (poll interval 5 s) |
| Serving app / control plane memory (RSS) | about 226 MB / 230 MB |
| `predictions` table, 204 rows | 144 KB |

## Limits

One deployment at a time, one serving process, one machine; no canary, shadow or A/B serving and no
multiple models behind one endpoint. No batch endpoints beyond 100 rows, no authentication, no TLS.
The prediction log grows without limit and has no retention or cleanup yet. Not run on Linux, or with models larger than the 13 MB measured elsewhere.
A real PostgreSQL server stop and restart was not exercised (a throwaway database refused new
connections instead). The cross-job test-set reuse decision is **still open**; nothing here counts
or restricts it.

## Feedback and live performance (P4-8, Step 23)

**Labels.** `POST /feedback` on the serving app: `{prediction_id, true_value, deployment_id?, model_version?}`.
It needs only PostgreSQL, so it works after the deployment ended or while no model is loaded. A
classification `true_value` must be one of the model's classes **as text** (the classes are read from the
prediction's own stored probabilities); a regression one a finite number (JSON number or clean number text,
stored in canonical form). Rules, enforced by the application and again by migration 0006:

| Case | Result |
| --- | --- |
| unknown `prediction_id` | 404 `prediction_not_found`; malformed id 422 |
| class not in the model's classes, or not a finite number | 422 `invalid_true_value` (lists the classes) |
| `deployment_id` or `model_version` given and not the prediction's own | 409 `prediction_mismatch`, nothing stored |
| the same label again (for a number: the same value) | 200 `created: false`, nothing written |
| a different label for an already labelled prediction | 409 `feedback_conflict`; the first label stands |
| database unreachable or the write fails | 503 `feedback_unavailable`; nothing partial is stored |

A label records the prediction's deployment and model version, copied from the prediction and checked by
the database; it can never be re-attached elsewhere. `feedback` is append-only (no update, delete or
truncate), so every label stays auditable with its time. Simultaneous submissions of one prediction write
exactly one row (tested with 8 racers, identical and conflicting).

**Live performance** (control plane, read-only):

- `GET /deployments/{id}/performance?scope=real|simulated|all&min_labelled=30`
- `GET /deployments/{id}/feedback?before_id=&limit=` (the labels, newest first)

Performance is computed on demand from **labelled predictions only**, with the same functions as the offline
evaluation (`evaluate.py`: accuracy, macro precision/recall/F1, ROC-AUC, confusion matrix, per-class table;
RMSE, MAE, R²). Unlabelled predictions are counted (`counts.unlabelled`) and excluded from every number.
`status` is `no_predictions`, `no_labels`, `insufficient_labels` (fewer than `min_labelled`: counts and a
note, **no metrics**), `ok`, or `not_computable`. A metric that cannot be defined for the labels at hand
(for example ROC-AUC when only one class was labelled) is null with the reason in `undefined`. Everything is
read in one database snapshot, so counts and scored rows always agree. At most the 50,000 most recent labels
are scored, and the response says so (`truncated`).

The default scope is **real traffic only**; simulated traffic (`simulated: true` on `/predict`) is reported
beside it in `by_traffic` and is included only when asked for. The stored **offline final-test** summary is
returned separately as `offline_reference` (source `offline_final_test`) and is never mixed into, or
subtracted from, the live numbers: it was measured on different rows at a different time, and live labels
can be a biased sample. Nothing here reads a dataset, a sealed package or the held-out test part (a test
spies on file opens and checks the module's imports); it does not retrain or change a deployment.

Limits: labels are matched one prediction at a time (no batch endpoint); no label editing by design (a wrong
label needs an owner decision); no time-window or per-class-over-time views; drift, alerts and the reference
sample are P4-9.
