import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { fx } from "./fixtures";

/** A small JSON-Schema checker (the parts FastAPI emits) so the recorded responses are held to the
 *  exported OpenAPI files: a backend change that breaks a fixture shape fails here. */
type Schema = Record<string, unknown>;
const spec = {
  control: JSON.parse(readFileSync("../docs/openapi-control.json", "utf8")),
  serving: JSON.parse(readFileSync("../docs/openapi-serving.json", "utf8")),
};

function check(
  api: "control" | "serving",
  schema: Schema,
  value: unknown,
  path: string,
  errors: string[],
) {
  if (schema.$ref) {
    const name = String(schema.$ref).split("/").pop()!;
    return check(api, spec[api].components.schemas[name], value, `${path}<${name}>`, errors);
  }
  if (Array.isArray(schema.anyOf)) {
    const trials = (schema.anyOf as Schema[]).map((s) => {
      const e: string[] = [];
      check(api, s, value, path, e);
      return e;
    });
    if (!trials.some((e) => e.length === 0))
      errors.push(`${path}: matches none of anyOf (${trials[0]![0]})`);
    return;
  }
  if (schema.enum && !(schema.enum as unknown[]).includes(value))
    errors.push(`${path}: ${String(value)} not in enum`);
  const type = schema.type as string | undefined;
  if (type === "object" || schema.properties) {
    if (typeof value !== "object" || value === null || Array.isArray(value))
      return void errors.push(`${path}: not an object`);
    const props = (schema.properties ?? {}) as Record<string, Schema>;
    for (const key of (schema.required ?? []) as string[])
      if (!(key in value)) errors.push(`${path}.${key}: missing`);
    for (const [key, v] of Object.entries(value)) {
      if (props[key]) check(api, props[key], v, `${path}.${key}`, errors);
      else if (schema.additionalProperties === false)
        errors.push(`${path}.${key}: not in the schema`);
    }
  } else if (type === "array") {
    if (!Array.isArray(value)) return void errors.push(`${path}: not an array`);
    if (schema.items)
      value
        .slice(0, 3)
        .forEach((v, i) => check(api, schema.items as Schema, v, `${path}[${i}]`, errors));
  } else if (type === "string" && typeof value !== "string") errors.push(`${path}: not a string`);
  else if (type === "integer" && !Number.isInteger(value)) errors.push(`${path}: not an integer`);
  else if (type === "number" && typeof value !== "number") errors.push(`${path}: not a number`);
  else if (type === "boolean" && typeof value !== "boolean") errors.push(`${path}: not a boolean`);
}

const cases: [string, "control" | "serving", string, unknown][] = [
  ["health", "control", "Health", fx.health],
  ["datasets", "control", "DatasetList", fx.datasets],
  ["dataset version", "control", "VersionModel", fx.version],
  ["validation", "control", "ValidationReportModel", fx.validation],
  ["algorithms", "control", "AlgorithmList", fx.algorithms],
  ["jobs", "control", "JobList", fx.jobs],
  ["job", "control", "JobDetail", fx.job],
  ["models", "control", "RegisteredModelList", fx.models],
  ["model versions", "control", "ModelVersionList", fx.versions],
  ["model version", "control", "ModelVersionDetail", fx.version1],
  ["model events", "control", "RegistryEventList", fx.events],
  ["active deployment", "control", "DeploymentDetail", fx.active],
  ["deployments", "control", "DeploymentList", fx.deployments],
  ["monitoring status", "control", "MonitoringStatus", fx.mon],
  ["reports", "control", "ReportList", fx.reports],
  ["report", "control", "ReportModel", fx.report],
  ["alerts", "control", "AlertList", fx.alerts],
  ["alert", "control", "AlertModel", fx.alert],
  ["live performance", "control", "LivePerformance", fx.perf],
  ["predictions", "control", "PredictionList", fx.preds],
  ["activity", "control", "ActivityReport", fx.act],
  ["served model", "serving", "ModelInfo", fx.smodel],
];

describe("recorded API responses satisfy the exported OpenAPI schemas", () => {
  for (const [label, api, name, value] of cases) {
    it(label, () => {
      const errors: string[] = [];
      check(api, { $ref: `#/components/schemas/${name}` }, value, name, errors);
      expect(errors).toEqual([]);
    });
  }
  it("a broken fixture is caught", () => {
    const errors: string[] = [];
    check(
      "control",
      { $ref: "#/components/schemas/PredictionList" },
      { ...fx.preds, extra: 1, scope: "everything" },
      "x",
      errors,
    );
    expect(errors.length).toBeGreaterThanOrEqual(2);
  });
});
