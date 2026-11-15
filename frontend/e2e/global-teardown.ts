import { existsSync, readFileSync, rmSync } from "node:fs";
import { PIDFILE } from "./global-setup";

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** Stop the dev servers, then ask each stack to remove its throwaway database and folders (SIGTERM
 *  makes scripts/ui_stack.py prove ownership and clean up) and wait for it to finish. */
export default async function globalTeardown() {
  if (!existsSync(PIDFILE)) return;
  const saved = JSON.parse(readFileSync(PIDFILE, "utf8")) as {
    dir: string;
    started: { pid: number; kind: string }[];
  };
  const started = saved.started;
  for (const p of started.filter((x) => x.kind === "ui")) {
    try {
      process.kill(p.pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
  for (const p of started.filter((x) => x.kind === "stack")) {
    try {
      process.kill(p.pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
  const end = Date.now() + 90_000;
  while (Date.now() < end && started.some((p) => alive(p.pid))) {
    await new Promise((r) => setTimeout(r, 500));
  }
  // the folder only holds the two "ready" files this run wrote
  rmSync(saved.dir, { recursive: true, force: true });
  rmSync(PIDFILE, { force: true });
}
