# Traffic simulator and the backend end-to-end story (P4-10, BUILD_PLAN Steps 25 and 35)

The simulator sends demo traffic to the running services so predictions, labels, drift reports and alerts
can be seen through the normal APIs. It is for a demo and for tests; it does not create real observations.

```
uv run python scripts/simulate_traffic.py run  data/demo/abalone_monitor_stream.csv --rows 200 --feedback 0.3
uv run python scripts/simulate_traffic.py run  data/demo/abalone_monitor_shifted.csv --rows 500
uv run python scripts/simulate_traffic.py demo abalone          # normal, shifted, normal, monitored (make simulate)
uv run python scripts/e2e_isolated.py abalone                  # the whole backend story on a throwaway stack (make e2e)
```

`run` and `demo` need `make api` and `make serve` running and a deployed model, and they **write** simulated
predictions, labels and reports to that stack (that is their purpose; they never create packages or touch
training data). `e2e` needs no running stack: it makes its own. Ports come from `AUTOML_API_PORT` (8100) and `AUTOML_SERVING_PORT` (8200) or
`--api-port` / `--serve-port`. Code: `src/automl_ops/simulator.py`, `src/automl_ops/e2e.py`.

## What it does

- Takes rows from a **monitoring stream file** (or its synthetic shifted copy), a seeded random subset, and
  sends them to `POST /predict` in batches (`--batch-size`, at most 100; `--rate` rows per second) with
  `simulated: true`. With `--feedback F` a seeded share `F` of the rows get their true label through
  `POST /feedback`. The same seed gives the same rows and the same labelled subset.
- Prints a JSON report (rows selected, sent, predicted, labelled, failures, whether it stopped early).
  Exit 0: clean; 1: failures or an early stop; 2: refused.
- `demo` runs three phases of 1,000 rows (the monitoring window): normal, shifted copy, normal again. After each
  it runs one monitoring run on the **simulated scope with the default thresholds** and prints what decided it
  (drifted features and their statistics, prediction statistic, live versus offline metric, the alerts). It does
  not assert that an alert appears. It creates the version's reference sample if it has none.

## Safety rules (each is tested)

- **Only the known demo monitoring files** are accepted: `<name>_monitor_stream.csv` or
  `<name>_monitor_shifted.csv` whose SHA-256 equals the one in `data/demo/manifest.json`. There is no override.
- It **refuses** a file whose SHA-256 (or original-bytes SHA-256) equals any registered dataset version, so
  training data cannot be replayed as traffic. If the control plane cannot be asked, it refuses too.
- Every request says `simulated: true`; the transport only contacts `127.0.0.1`. It writes nothing to the
  database itself and never touches the training pool.
- Real and simulated traffic stay separate everywhere: performance and monitoring default to the real scope, a
  simulated run reads only simulated rows, alerts carry their scope.
- A dead or failing serving app stops the run after three failed batches in a row with a clear reason; counts
  of what did get through are exact. A deployment switched during a run makes later labels for the old
  deployment refuse (409): they are counted as failed and never attached to the wrong deployment.

## The end-to-end story (`scripts/e2e_isolated.py`, `scripts/e2e_demo.py`, `make e2e`)

Over HTTP only: upload the dataset's **training pool**; train a final job with tracking; check it was logged;
register and promote the model; deploy it and check the serving app serves it; then the `demo` traffic with
feedback and monitoring. Each stage is reported passed or failed with what was observed, and the story stops at
the first failure.

**Safety (follow-up to P4-10, `src/automl_ops/isolation.py`, `e2e_stack.py`).** The story writes rows and a model
package, so it only runs on a stack made for it:

- `make e2e` runs `scripts/e2e_isolated.py`. It creates a new database `automl_test_e2e_<random>` (migrated, and
  stamped with a comment naming this run), a temporary folder outside the project holding the dataset store,
  the model packages and a **SQLite MLflow store**, and starts the API, serving app and worker on free local
  ports with an environment built from scratch (no inherited `AUTOML_*`, `MLFLOW_*` or `POSTGRES_*` variables).
  The real `automl_ops` and `mlflow` databases, `data/` and its MLflow archive are never connected to or used;
  the only other connection is to the maintenance database `postgres` to create and drop its own database.
- Both services report an `isolation` object on `GET /health`: booleans for `database`, `store`, `models`,
  `mlflow` and `token`, reason codes, and two one-way identifiers (database, run). It never contains a path,
  host, user, password or token. A service is isolated only if the database is named `automl_test_e2e_*`
  (never `automl_ops`, `mlflow`, `postgres` or a template), the store and models folders are outside `data/`
  (links followed), tracking is SQLite in a folder outside `data/` (never the PostgreSQL `mlflow` database),
  and the per-run token is present. An app built without the report says `not_assessed`.
- `scripts/e2e_demo.py` verifies this **before any write**, using GET requests only: its own settings are
  isolated (otherwise it contacts nothing); both services answer, report isolation, and are on this run's
  database and run; and the datasets, models, jobs and deployments lists are empty. An older stack that does
  not report the field is refused. **There is no force or bypass option.**
- Cleanup stops only the processes it started, then proves ownership (the database comment, the owner marker
  file, the folder location, no symbolic link) before deleting. If any proof fails, nothing is deleted and the
  remaining resources are listed (exit code 3). Sealed packages are read-only by design; they are made writable
  inside the run's own folder only after ownership is proved.

What this does not cover: the worker's own wiring is set by the wrapper, not reported over HTTP; the check is a
guard against mistakes on a single-user machine, not a security boundary.

## Observed with the default thresholds (live run: real processes, real demo CSVs, throwaway database)

Measured 2026-11-14 (before the windowed-performance follow-up; see the end of this section) on the MacBook Air M1: three server processes plus `e2e_demo.py`, a throwaway database,
store, models and MLflow folder (all removed afterwards). Not the real database.

| | Abalone (regression, RMSE) | Adult (classification, macro F1) |
|---|---|---|
| story | all 9 stages passed, 8.6 s | all 9 stages passed, 12.0 s |
| peak memory (API / serving / worker) | 234 / 228 / 301 MB | 295 / 228 / 398 MB |
| normal traffic | no alert; 0 of 8 features drifted | no alert; 1 of 14 flagged (`native-country`), share 0.07 |
| shifted traffic | `prediction_shift` (KS 0.746) and `performance_drop` (RMSE 4.20 vs 2.21 offline, +90 %) opened; **`drifted_feature_share` did not fire**: 3 of 8 features drifted (`Whole_weight`, `Shell_weight`, `Sex`), share 0.375 < 0.5 | `prediction_shift` opened (JS 0.121 against the 0.1 threshold: a thin margin); `performance_drop` did not fire; 4 of 14 features drifted, share 0.286 < 0.5 |
| after normal traffic again | `prediction_shift` auto-resolved; `performance_drop` **stays open** (see below) | `prediction_shift` auto-resolved; nothing open |

What this shows and does not show:

- The shifted copies change only a few columns by design, so the **share-of-drifted-features rule (0.5) does not
  fire on either real demo dataset**; the shift is caught by the prediction rule (and, on Abalone, by
  performance). That is the honest result with the existing thresholds. A lower share threshold (for example
  0.25 to 0.3) would fire on both; that is a change for the owner to approve, not made here.
- **(Measured before the windowed-performance follow-up; since changed.)** `performance_drop` did not clear
  after traffic recovered. Live performance used all labelled
  predictions in scope (up to 50,000, P4-8), not the 1,000-row drift window, so the wrong predictions made
  during the shift keep the metric worse until enough good labels dilute them (Abalone: RMSE 3.64 after the
  recovery phase, still 65 % worse). Performance is now scored over the labels in the same window as drift
  (see `docs/monitoring.md`); the table above was not re-measured on the live demo data.
- A single sparse categorical (`native-country`) can cross the Jensen-Shannon threshold of 0.1 on normal
  traffic through sampling noise; one such feature is far below the share threshold, so no alert resulted.
- Two datasets, one seed, one run each: these are observations, not sensitivity or false-alarm rates.
  The shifts are synthetic (`docs/data.md`).

## Limits

Localhost only, no authentication. The simulator is a script, not a service: nothing runs in the background and
nothing schedules monitoring. Traffic is sampled without replacement from a stream of 1,253 (Abalone) or 14,652
(Adult) rows, so `--rows` cannot exceed the stream. The in-process test transports are not the HTTP stack; the
live run above is the evidence for real processes. Not run on Linux.

**After the windowed-performance follow-up (2026-11-14, one `make e2e` run on the Abalone data, throwaway
stack, default thresholds).** All 9 stages passed in about 12 s. Normal phase: `performance_drop` clear
(RMSE 2.04 against 2.21 offline, 300 labels). Shifted phase: `performance_drop` and `prediction_shift` opened
(RMSE 5.58, +153 %, bootstrap lower bound +147 %). Recovery phase: **`performance_drop` auto-resolved** (RMSE 2.10,
-5 %) together with `prediction_shift`. Adult was not re-run live; one dataset, one seed.
