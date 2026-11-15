# AutoML Ops frontend

React 19 + TypeScript (strict) + Vite. A local, single-user UI over the two backend apps: the control
plane (`:8100`) and the serving app (`:8200`). No authentication, localhost only. Overview in
[docs/frontend.md](../docs/frontend.md).

## Run it

```
make db-up                      # PostgreSQL (once)
make api & make serve & make worker   # three terminals: control plane, serving app, training worker
make ui-install                 # npm ci (exact versions from package-lock.json)
make ui-dev                     # http://localhost:5173
```

The backend allows exactly one browser origin (`AUTOML_UI_ORIGIN`, default `http://localhost:5173`), so
the dev server uses that port with `strictPort`. If you serve the UI elsewhere, set `AUTOML_UI_ORIGIN`
to match. API addresses come from `VITE_API_URL` and `VITE_SERVING_URL` (defaults
`http://127.0.0.1:8100` and `http://127.0.0.1:8200`).

**Try it without touching your data:** `make ui-demo` creates a throwaway backend (new database
`automl_test_e2e_*`, temporary folders, SQLite MLflow), seeds it with the Abalone demo story (needs
`make demo-data`), starts the UI at http://localhost:5173 and removes everything on Ctrl-C. The header
shows "Throwaway demo stack" when the backend reports isolation; otherwise it says that actions are saved.

## Scripts (`cd frontend`)

| Command                                       | What it does                                                                           |
| --------------------------------------------- | -------------------------------------------------------------------------------------- |
| `npm run dev` / `build` / `preview`           | dev server, production build (type-checks first), preview                              |
| `npm run lint` / `format:check` / `typecheck` | ESLint (incl. jsx-a11y), Prettier, `tsc --noEmit`                                      |
| `npm run gen:api`                             | regenerate `src/api/*.d.ts` from `../docs/openapi-*.json` (`make openapi` first)       |
| `npm run check:api`                           | fails if the generated types are stale                                                 |
| `npm test`                                    | Vitest: formatting, chart data, contract check of recorded responses, components (MSW) |
| `npm run e2e`                                 | Playwright (Chromium) on two throwaway stacks it creates and removes; axe checks       |
| `make ui-check`                               | lint, format, types, API types, unit tests and build in one go                         |

## Layout

`src/api/` typed client (`openapi-fetch`), generated types, TanStack Query hooks, shapes of the open JSON
objects. `src/components/` shared shell, tables, states, dialogs, charts frame, glossary `Term`.
`src/features/<area>/` one folder per lifecycle stage. `src/styles/` design tokens and components (plain CSS,
light and dark). `tests/` unit and component tests with recorded fixtures. `e2e/` browser tests.

`npm run e2e` needs Chromium once: `npx playwright install chromium`.
