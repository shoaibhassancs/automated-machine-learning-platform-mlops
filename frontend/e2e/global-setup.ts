import { type ChildProcess, spawn } from "node:child_process";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { FLOW, SEEDED, type Stack } from "./stacks";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
export const PIDFILE = join(tmpdir(), "automl-ui-e2e.json");

async function waitFor(check: () => Promise<boolean> | boolean, what: string, ms: number) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (await check()) return;
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`timed out waiting for ${what}`);
}

export default async function globalSetup() {
  const dir = mkdtempSync(join(tmpdir(), "automl-ui-e2e-"));
  const started: { name: string; pid: number; kind: "stack" | "ui" }[] = [];
  const children: ChildProcess[] = [];
  const ready: Record<string, string> = {};

  for (const [name, s] of [
    ["flow", FLOW],
    ["seeded", SEEDED],
  ] as [string, Stack][]) {
    const file = join(dir, `${name}.json`);
    ready[name] = file;
    // The venv's interpreter directly (not `uv run`), so SIGTERM reaches the process that cleans up.
    const child = spawn(
      join(root, ".venv", "bin", "python"),
      [
        "scripts/ui_stack.py",
        "--seed",
        s.seed,
        "--api-port",
        String(s.apiPort),
        "--serve-port",
        String(s.servingPort),
        "--ui-origin",
        s.ui,
        "--ready-file",
        file,
      ],
      { cwd: root, stdio: ["ignore", "pipe", "pipe"] },
    );
    child.stderr?.on("data", (d: Buffer) => process.stderr.write(`[stack:${name}] ${d}`));
    children.push(child);
    started.push({ name, pid: child.pid!, kind: "stack" });
  }
  // Remember the processes before waiting, so teardown can clean up even if setup fails.
  writeFileSync(PIDFILE, JSON.stringify({ dir, started }));

  await Promise.all(
    Object.entries(ready).map(([name, file]) =>
      waitFor(() => existsSync(file), `the ${name} stack`, 240_000),
    ),
  );

  for (const [name, s] of [
    ["flow", FLOW],
    ["seeded", SEEDED],
  ] as [string, Stack][]) {
    const ui = spawn(
      process.execPath,
      ["node_modules/vite/bin/vite.js", "--port", String(s.uiPort), "--strictPort"],
      {
        cwd: join(root, "frontend"),
        env: { ...process.env, VITE_API_URL: s.api, VITE_SERVING_URL: s.serving },
        stdio: "ignore",
      },
    );
    started.push({ name, pid: ui.pid!, kind: "ui" });
    children.push(ui);
    writeFileSync(PIDFILE, JSON.stringify({ dir, started }));
    await waitFor(
      async () => {
        try {
          return (await fetch(s.ui)).ok;
        } catch {
          return false;
        }
      },
      `the ${name} dev server`,
      60_000,
    );
  }
}
