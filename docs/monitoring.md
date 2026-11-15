# Drift monitoring, reference samples and alerts (P4-9, BUILD_PLAN Step 24)

Monitoring answers: do the inputs and predictions of a **deployed** model differ from what the model was
trained on? It runs **on demand in the control plane**, reads only the `predictions` table (and the
labels of P4-8), and never touches serving: the serving app does not import any of it (a test checks
this), and a failing or slow run cannot slow or break `/predict` (tested under load). Drift means "inputs
differ from the training sample", **not** "the model is wrong". Local, single-user, no authentication.

## Deviation: Evidently is not used (needs owner approval)

The plan names Evidently, to be measured in a spike before any install, with SciPy as the fallback (D-028).
Evidently is not installed or locked, and installing it needs package-index access and a dependency
change, which have not been approved. So `automl_ops/monitor.py` implements the same kinds of tests with
**numpy only** (no dependency or lockfile change) and is the single place to swap in Evidently later. Not
built: Evidently's HTML report on disk (the full report is stored as JSON in PostgreSQL instead).

## Reference samples (`reference.py`, table `reference_samples`)

`POST /models/{name}/versions/{n}/reference` (once per version; `GET` shows its provenance):

- the sealed package must verify (`load_package`); the dataset version in its manifest is read with its
  SHA-256 check; the frozen holdout split is recomputed from the package's seed and settings and must equal
  the split fingerprints in the manifest, else 409 and nothing is stored;
- the sample is up to **1,000 rows of the training part only** (all of them if fewer), chosen with the
  package's seed, so the same package always gives the same sample. Held-out test rows are never selected
  or stored. (The CSV is read whole, as training does, and the target of every row places it in the split,
  as in training; no test-row feature value is used);
- it is stored with the model's predictions on those rows, the dataset reference and SHA-256, the SHA-256 of
  the training row positions, the package id, the seed and a SHA-256 of the sample itself;
- it is **separate from the dataset and the sealed package** (stored in PostgreSQL; no package file is
  written), **immutable** (database triggers) and never replaced: asking again returns the stored one
  (200, `created: false`). Nothing is retrained.

## Runs and reports

`POST /deployments/{id}/monitor` `{scope?: real|simulated, thresholds?: {...}}`:

- the window is the most recent `window` (default 1,000) logged predictions of that deployment **in that
  scope**; real and simulated traffic are never mixed (default real);
- per feature: numeric -> two-sample Kolmogorov-Smirnov, drift when p < 0.05; categorical -> Jensen-Shannon
  distance, drift when >= 0.1; the **dataset-drift share** (drifted / evaluated features) and **prediction
  drift** (JS on predicted classes, KS on regression predictions); missing-value shares are shown;
- performance: the selection metric on the labelled predictions **inside the same window as the drift
  verdict** (labels read for exactly those prediction ids, in the same snapshot), against the stored offline
  value (relative worsening > 10 %), evaluated only with at least 30 labels in the window. A seeded bootstrap
  (200 resamples, one-sided 90 % lower bound, seed derived from the scored prediction ids) guards against
  small-sample noise: verdict `triggered` (worse than 10 % and the lower bound is above zero), `clear` (not
  worse than 10 %) or `inconclusive` (worse by the point estimate but within sampling noise: no verdict).
  F1/accuracy/R² are higher-is-better; RMSE/MAE lower-is-better. The window basis, the interval and the label
  count are stored in the report's JSON. `GET /deployments/{id}/performance` is unchanged (all labelled history);
- the thresholds used are stored with every report; they are provisional defaults, overridable per run.

| Status | Meaning |
| --- | --- |
| `ok` | a verdict was reached; rules were evaluated and alerts updated |
| `insufficient_data` | window or reference below `min_rows` (30): counts and the reason, **no drift result, no alert change** |
| `failed` | the run broke: the error is recorded; never shown as a success; no alert change |

Reports are immutable (database triggers); a run writes its report and all alert changes in one transaction
at the very end, so an interrupted run leaves nothing behind. A second concurrent run for the same
deployment is refused (409 `monitoring_in_progress`) by a PostgreSQL advisory lock. No reference sample:
409 `no_reference`. Reads: `GET /deployments/{id}/monitoring` (reference present, latest report per scope,
unresolved alerts), `GET /deployments/{id}/reports` (history, summaries omitted), `GET /reports/{id}`.

## Alerts

Rules: `drifted_feature_share` (share >= 0.5, warning), `prediction_shift` (warning), `performance_drop`
(critical). "Minimum sample size" is a guard (`insufficient_data`), not an alert, so thin data cannot raise
false alarms. At most **one unresolved alert** exists per (deployment, rule, scope) (unique index): a
condition that keeps holding updates it (`occurrences`, `reoccurred` event). A later `ok` run that evaluates
the rule as clear resolves it (`auto_resolved`); a rule that was not evaluated changes nothing; failed and
insufficient runs change nothing. People can `POST /alerts/{id}/acknowledge` (open only) and
`POST /alerts/{id}/resolve` (open or acknowledged) with a reason. A resolved alert never changes; a
recurrence opens a new alert. Every state change is an append-only event (`GET /alerts/{id}` shows them) and
the database refuses forbidden moves from hand-written statements. `GET /alerts` filters by status,
deployment, rule and scope.

## Verified

Real PostgreSQL, real sealed packages, a real dataset store. Reference: provenance, determinism, no test row
in the sample, no package file changed, immutability, errors (missing dataset, changed dataset, split
mismatch, unverifiable package). Monitoring: unshifted traffic raises no alert and injected shift does
(classification and regression); repeats update one alert; recovery auto-resolves and a recurrence opens a
new one; real and simulated stay apart; failed and interrupted runs leave no false success or partial state;
6 racing runs never duplicate alerts; serving keeps answering during monitoring and during monitor failures;
a run opens no model, dataset or package file. The P4-9 tests used small synthetic toy data (150 rows).
P4-10 added the traffic simulator and a live run on the real demo datasets with these same thresholds:
see `docs/simulator.md` for what the shifted copies did and did not trip (the feature-share rule did not
fire on either dataset). That run also showed `performance_drop` staying open after recovery, because live
performance then used all labelled history; it is now windowed like drift (`tests/test_performance_window.py`;
the simulator recovery test expects it to resolve). The live numbers in `docs/simulator.md` were measured
before that change.

## Limits

On demand only: no scheduler or background monitoring. No Evidently and no HTML report (above). Reports and
alerts grow without limit. One reference sample per version (a different one would need an owner decision).
KS on large windows flags tiny shifts as significant (the statistic is shown beside the p-value). Windows
are the most recent N predictions, not time windows. Not run on Linux. The cross-job test-set reuse decision
is still open; nothing here reads or counts it.
