# Frontend (Phase 5, BUILD_PLAN Steps 27-32, D-048)

A React 19 + TypeScript (strict) + Vite single-page app in `frontend/`. Local, single-user, no
authentication, localhost only. Start it with `make ui-dev` (see `frontend/README.md`), or try it on a
throwaway seeded backend with `make ui-demo`.

## Design approach

One shell, one page pattern: **title, one-line "what is this", primary action, content**. The sidebar follows the
lifecycle in order (Overview, 1 Datasets, 2 Training with Compare runs, 3 Models, 4 Deployment, 5 Monitoring), and the
Overview page is a map of those five stages with the project's current state under each, plus a single "what to do
next". Plain CSS with design tokens (one accent, restrained neutrals), light and dark by `prefers-color-scheme`.
No UI kit, no external fonts or icons: nothing is fetched from outside. Technical words are explained in place by
`Term` (a button that opens a short note; the definitions live in `src/lib/glossary.ts`). Every chart sits in a
`ChartFrame`: title, "what am I looking at", the chart, a text summary and the same numbers as a table.

## Screens and the backend they use

| Screen (route) | Backend |
| --- | --- |
| Overview `/` | `/datasets`, `/jobs`, `/models`, `/deployments/active`, `/alerts` |
| Datasets `/datasets` | `GET/POST /datasets` (multipart upload; a rejected CSV shows the API's message and code) |
| Dataset version `/datasets/:name/v/:n` (+ `/profile`, `/validation`, `/task`) | `.../preview`, `.../profile`, `.../validation`, `POST .../task-check` |
| Training `/training`, `/training/new`, `/training/:id` | `GET/POST /jobs`, `GET /jobs/{id}` (polled every 1.5 s while active), `POST /jobs/{id}/cancel`, `/algorithms`, `POST /models` (register) |
| Experiments `/experiments`, `/experiments/compare` | `GET /jobs?status=succeeded` + each job's stored result (cross-validation scores only) |
| Models `/models`, `/models/:name`, `/models/:name/v/:n` | `/models`, `/versions`, `/events`, `promote`, `archive`, `rollback`, `reference` (GET and POST) |
| Deployment `/deployment`, `/deployment/try` | `/deployments`, `/active`, `switch`, `rollback`, `undeploy`; serving `GET /model`, `POST /predict`, `POST /feedback` |
| Monitoring `/monitoring` | `/deployments/{id}/activity` (new), `/performance`, `/monitoring`, `POST /monitor`, `/reports/{id}`, `/alerts` |
| Reports `/monitoring/reports`, `/:id` | `/deployments/{id}/reports`, `/reports/{id}` |
| Alerts `/monitoring/alerts`, `/:id` | `/alerts`, `/alerts/{id}`, `acknowledge`, `resolve` (a reason is required) |
| Predictions and feedback `/monitoring/predictions` | `/deployments/{id}/predictions` (new), serving `POST /feedback` |

The two new read-only endpoints (`predictions`, `activity`) are described in `docs/api.md`.

## What the screens say about the evidence

- Training scores are cross-validation on the training rows; the winner is explained by the stored selection rule.
- The **final test result** appears only for the selected model, labelled as such, with the API's own limitation text
  (reuse of the held-out part across jobs is accepted for V1 (D-057); never "independent" or "leakage-proof").
- Live performance is shown beside the offline score with the note that they are different measurements, and with
  label counts; below 30 labels it says so instead of showing a number.
- Drift is described as "a reason to look, not proof". Simulated traffic is badged everywhere and never mixed with
  real traffic. Input values of predictions are not displayed.
- Destructive or consequential actions (promote, archive, deploy, switch, rollback, stop serving, resolve, run a final
  test) open a confirmation that names the consequence; registry and alert actions require a reason.

## States, responsive layout, accessibility

Every data view has a skeleton (loading), an empty state with a next action, an error panel (the API's message and
code, or "the backend is not reachable", with retry), and a success toast (`aria-live`). Layouts were reviewed at
390, 900 and 1280 px (the sidebar becomes a menu below 900 px; tables scroll inside their container; the page itself
never scrolls sideways, which a test checks). Keyboard: a skip link, focus moves to the new page's title after
navigation, visible focus rings, native `<dialog>` for confirmations, scrollable tables are focusable regions.
`prefers-reduced-motion` is respected. Status is always text plus an icon, never colour alone.

## Tests

- `npm test` (Vitest, 61 tests): formatting; chart data equals API data (cross-validation bars, ROC/PR, residuals,
  confusion matrix, histograms, activity, prediction bars, comparison matrix); metric direction (F1 against RMSE/MAE);
  registry transitions; recorded responses checked against the exported OpenAPI schemas (a backend change that breaks
  a fixture fails); the typed client's error mapping; component tests with a mock server (empty, list, bad CSV, job
  results and progress, the new-job form, the prediction form generated from the model, alert actions, dialogs).
- `npm run check:api`: generated types are current. `npm run lint` (ESLint incl. jsx-a11y), `format:check`,
  `typecheck`, `build`.
- `npm run e2e` (Playwright, Chromium): two throwaway backends created by `scripts/ui_stack.py` (new database
  `automl_test_e2e_*`, temporary folders, SQLite MLflow, removed afterwards). One is empty and the whole lifecycle is
  built through the browser (upload, inspect, configure, train, compare, register, promote, deploy, predict, feedback,
  switch, roll back, monitor); the other is seeded with the demo story (model, traffic, drift alerts) for the
  monitoring and alert flows, keyboard tests, axe-core accessibility checks (light and dark) and layout checks at three
  widths. Serious or critical axe violations fail the run.
- Python: tests for the two new endpoints (`tests/test_predictions_api.py`) and for the UI-origin option of the
  throwaway stack.

## Limits

- Chromium only; Safari and Firefox are not tested. No screen-reader test (VoiceOver is NOT RUN). Automated axe
  cannot find every problem (reading order, meaning of text); a manual visual pass was made at three widths.
- Fixtures are recorded from a real stack at one point in time; regenerate them after a backend change.
- `npm audit` reports 7 moderate findings, all in the development-only lint toolchain (`minimatch` through
  `typescript-eslint`); the production dependencies report none. Not upgraded without approval.
- Containers, Compose and CI for the frontend are Phase 6.
