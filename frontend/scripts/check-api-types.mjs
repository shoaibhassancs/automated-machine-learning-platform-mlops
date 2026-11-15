// Fails if the generated API types are stale: regenerates them into a temp folder and compares.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "automl-api-types-"));
let stale = false;
try {
  for (const name of ["control", "serving"]) {
    const out = join(dir, `${name}.d.ts`);
    execFileSync("npx", ["openapi-typescript", `../docs/openapi-${name}.json`, "-o", out], {
      stdio: "ignore",
    });
    if (readFileSync(out, "utf8") !== readFileSync(`src/api/${name}.d.ts`, "utf8")) {
      console.error(`src/api/${name}.d.ts is stale: run \`npm run gen:api\``);
      stale = true;
    }
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}
if (stale) process.exit(1);
console.log("API types are current");
