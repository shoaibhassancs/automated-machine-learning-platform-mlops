import { describe, expect, it } from "vitest";
import { GLOSSARY } from "@/lib/glossary";
import {
  bytes,
  clock,
  duration,
  metricLabel,
  num,
  pct,
  plural,
  signedPct,
  when,
} from "@/lib/format";
import { suggestName } from "@/features/datasets/DatasetList";
import { rangeSince } from "@/features/monitoring/common";
import { allowedActions } from "@/features/models/transitions";
import { metricDirection } from "@/features/training/evalData";

describe("number formatting", () => {
  it("formats numbers, percentages and missing values", () => {
    expect(num(1234567)).toBe("1,234,567");
    expect(num(0.123456, 3)).toBe("0.123");
    expect(num(null)).toBe("n/a");
    expect(pct(0.256)).toBe("25.6%");
    expect(signedPct(0.808)).toBe("+80.8%");
    expect(signedPct(-0.05)).toBe("\u22125%");
    expect(plural(1, "job")).toBe("1 job");
    expect(plural(2, "job")).toBe("2 jobs");
  });
  it("formats durations, sizes, dates and metric names", () => {
    expect(duration(0.25)).toBe("250 ms");
    expect(duration(12.34)).toBe("12.3 s");
    expect(duration(125)).toBe("2 min 5 s");
    expect(bytes(2048)).toBe("2.0 KB");
    expect(when(null)).toBe("n/a");
    expect(when("not a date")).toBe("not a date");
    expect(clock("2026-11-14T09:05:00Z")).toMatch(/\d\d:\d\d/);
    expect(metricLabel("f1_macro")).toBe("F1 (macro)");
    expect(metricLabel("custom")).toBe("custom");
  });
});

describe("metric direction (F1 against RMSE and MAE)", () => {
  it("treats errors as lower-is-better and scores as higher-is-better", () => {
    for (const m of ["rmse", "mae"]) expect(metricDirection(m)).toBe("lower");
    for (const m of ["f1_macro", "accuracy", "roc_auc", "r2", "precision_macro"]) {
      expect(metricDirection(m)).toBe("higher");
    }
  });
});

describe("dataset names and time ranges", () => {
  it("suggests an API-valid name from a file name", () => {
    expect(suggestName("Abalone Train Pool.csv")).toBe("abalone-train-pool");
    expect(suggestName("__Data (1).CSV")).toMatch(/^[a-z0-9][a-z0-9_-]*$/);
  });
  it("turns a range into a start time", () => {
    const now = Date.parse("2026-11-14T12:00:00Z");
    expect(rangeSince("all", now)).toBeUndefined();
    expect(rangeSince("1h", now)).toBe("2026-11-14T11:00:00.000Z");
    expect(rangeSince("7d", now)).toBe("2026-11-07T12:00:00.000Z");
  });
});

describe("registry rules shown in the UI match docs/registry.md", () => {
  it("offers only the allowed transitions", () => {
    expect(allowedActions("development")).toEqual({ promote: ["staging"], archive: true });
    expect(allowedActions("staging")).toEqual({ promote: ["production"], archive: true });
    expect(allowedActions("archived")).toEqual({ promote: ["staging"], archive: false });
    expect(allowedActions("production")).toEqual({ promote: [], archive: false });
  });
});

describe("glossary", () => {
  it("has a plain-language definition for every term", () => {
    for (const [key, text] of Object.entries(GLOSSARY)) {
      expect(text.length, key).toBeGreaterThan(30);
      expect(text.endsWith("."), key).toBe(true);
    }
  });
  it("does not over-claim", () => {
    const all = Object.values(GLOSSARY).join(" ").toLowerCase();
    expect(all).not.toContain("leakage-proof");
    expect(all).not.toContain("guarantee");
  });
});
