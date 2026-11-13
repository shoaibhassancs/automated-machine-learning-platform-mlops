# Scope

Draft 2 (revised after review) · Nov 13, 2026 · Quick reference derived from `project-notes/PROJECT_GUIDE.md`.

## About this document
A working reference, not a fixed contract. Tools, packages, versions and implementation details in
the planning documents are provisional recommendations; open decisions are made when their step
arrives. Scope or boundary changes are presented to the owner first, and the phase gates stay.

## Objective
A single-user web platform that automates and manages the ML lifecycle for tabular CSV data:
dataset validation, multi-model training, experiment tracking, evaluation, a model registry,
REST deployment, and monitoring (performance, drift, alerts). It demonstrates MLOps
engineering, not a new prediction application.

## Intended users
Hiring reviewers and engineers evaluating an AI/ML portfolio; the owner as a learner.

## In scope (Version 1)
- Upload CSV, preview, profile, Pandera validation, missing-value report, immutable dataset
  versions, target and task (classification/regression) configuration
- Training several scikit-learn algorithms with editable parameters, background jobs, and
  per-algorithm progress
- MLflow experiment tracking: runs, parameters, metrics, artifacts, history, comparison,
  reproducibility check
- Evaluation: classification and regression metrics, confusion matrix, ROC/PR and residual
  charts, comparison, best-model selection by a chosen metric (cross-validation based);
  regular runs record cross-validation results only, and a sealed test set is scored once per
  job, for the selected model, after its configuration is frozen (reuse of the test part across
  jobs is an open decision; until decided, reported test results carry that limitation)
- Model registry: versions, status (development/staging/production), promote, roll back,
  event log (PostgreSQL is authoritative for status; MLflow aliases mirror it; mismatches are
  reported and reconciled explicitly)
- Deployment of a registered version through a separate FastAPI serving app; prediction form;
  deployed-version management
- Monitoring: prediction logging, ground-truth feedback, live performance, Evidently data and
  prediction drift, rule-based alerts, dashboard; labelled synthetic traffic simulator
- PostgreSQL with migrations, FastAPI, React (TypeScript), Docker Compose, GitHub Actions CI,
  README with screenshots and a short architecture doc

## Out of scope (Version 1)
- Kubernetes, cloud or public hosting, dynamic container orchestration
- Accounts, roles, authentication (local, single-user only)
- Celery/Redis, Airflow/Prefect, feature stores, other orchestration tools
- AutoML libraries, hyperparameter search, deep learning, non-tabular data
- Automatic retraining, scheduled monitoring, A/B/shadow/canary serving
- Batch-prediction file upload; email/chat alerts
- Publishing images or deploying anywhere from CI
- Claims of production readiness, public-internet security, or benchmark-level model quality

## Key boundaries
- Local, localhost-bound, no auth: do not expose publicly or upload sensitive data.
- Models are pickle-based joblib files and are loaded only from the platform's own store.
- "Automated" means one-click pipelines with defaults; promotion and deployment are human decisions.
- Drift means "inputs differ from training data", not "the model is wrong".
- Data, models, MLflow runs and secrets are never committed.
- Demo datasets are small and openly licensed (checked at Step 2); drift demos use labelled synthetic shifts.
- Monitoring data is separate: the traffic simulator and its shifted copy use a disjoint monitoring
  stream, never rows used for training, cross-validation, selection or the final test.

## Done for Version 1
Every item above works end to end from the UI; tests and CI pass; the demo is reproducible in
three documented stages: (1) a one-time demo-data preparation script, (2) `docker compose up`,
which starts the services only (it does not prepare data or run the workflow), and (3) an
end-to-end script that runs the workflow against the running stack (upload → train → compare →
register → promote → deploy → predict → drift alert), or the same flow driven by hand in the UI;
documentation states what is and is not verified.
