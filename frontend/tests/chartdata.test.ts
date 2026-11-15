import { describe, expect, it } from "vitest";
import type { JobResult, ReportSummary } from "@/api/shapes";
import { histogramData } from "@/features/datasets/histogramData";
import { compareMatrix } from "@/features/experiments/Compare";
import { activitySeries } from "@/features/monitoring/MonitoringOverview";
import { predictionBars } from "@/features/monitoring/ReportDetail";
import {
  confusionShares,
  curveSeries,
  residualPoints,
  scorePoints,
} from "@/features/training/evalData";
import type { Evaluation } from "@/api/shapes";
import { fx } from "./fixtures";

const result = fx.job.result as unknown as JobResult;

describe("chart values equal the API values", () => {
  it("cross-validation bars are the ranking's means and spreads, with the winner marked", () => {
    const sel = result.selection!;
    const points = scorePoints(sel);
    expect(points.map((p) => [p.key, p.mean, p.std])).toEqual(
      sel.ranking.map((r) => [r.algorithm, r.metric_mean, r.metric_std]),
    );
    expect(points.filter((p) => p.selected).map((p) => p.key)).toEqual([sel.selected!.algorithm]);
  });

  it("residual points keep every complete point and drop incomplete ones", () => {
    const raw = result.test_result!.evaluation.residual_points!;
    const out = residualPoints(raw);
    expect(out).toHaveLength(raw.length);
    expect(out[0]).toEqual({
      actual: raw[0]!.actual,
      predicted: raw[0]!.predicted,
      residual: raw[0]!.residual,
    });
    expect(residualPoints([{ row: 1, actual: null, predicted: 2, residual: 1 }])).toEqual([]);
  });

  it("ROC and PR series carry the curve points unchanged, one series per class", () => {
    const e = fx.classification as unknown as Evaluation;
    const roc = curveSeries(e.roc_curves!, "roc");
    expect(roc.map((s) => s.name)).toEqual(["neg", "pos"]);
    expect(roc[0]!.points).toEqual([
      { x: 0, y: 0 },
      { x: 0.1, y: 0.6 },
      { x: 0.4, y: 0.9 },
      { x: 1, y: 1 },
    ]);
    const pr = curveSeries(e.pr_curves!, "pr");
    expect(pr[1]!.points[1]).toEqual({ x: 0.6, y: 0.85 });
    expect(
      curveSeries([{ class: "a", fpr: [0, null, 1], tpr: [0, 0.5, 1] }], "roc")[0]!.points,
    ).toHaveLength(2);
  });

  it("confusion shares are row-normalised", () => {
    const shares = confusionShares((fx.classification as unknown as Evaluation).confusion_matrix!);
    expect(shares[0]).toEqual([0.8, 0.2]);
    expect(shares.map((r) => r.reduce((a, b) => a + b, 0))).toEqual([1, 1]);
    expect(confusionShares([[0, 0]])).toEqual([[0, 0]]);
  });

  it("histogram bars are the API counts with matching edges", () => {
    for (const h of fx.preview.histograms) {
      const bars = histogramData(h);
      expect(bars.map((b) => b.count)).toEqual(h.counts);
      expect(bars.reduce((s, b) => s + b.count, 0)).toBe(h.count - h.missing);
      expect(bars[0]!.from).toBe(h.edges[0]);
      expect(bars.at(-1)!.to).toBe(h.edges.at(-1));
    }
  });

  it("activity series splits labelled and unlabelled and keeps the totals", () => {
    const series = activitySeries(fx.act.buckets, "simulated", fx.act.bucket);
    expect(series.reduce((s, b) => s + b.total, 0)).toBe(fx.act.totals.simulated.predictions);
    expect(series.reduce((s, b) => s + b.labelled, 0)).toBe(fx.act.totals.simulated.labelled);
    expect(series.every((b) => b.labelled + b.unlabelled === b.total)).toBe(true);
    expect(activitySeries(fx.act.buckets, "real", "hour").every((b) => b.total === 0)).toBe(true);
  });

  it("prediction bars cover every class of either side", () => {
    const bars = predictionBars({
      method: "js",
      status: "ok",
      statistic: 0.2,
      p_value: null,
      threshold: 0.1,
      drifted: true,
      reference: { neg: 0.7, pos: 0.3 },
      current: { pos: 0.9, other: 0.1 },
    } as ReportSummary["prediction"] & object);
    expect(bars).toEqual([
      { name: "neg", reference: 0.7, current: 0 },
      { name: "pos", reference: 0.3, current: 0.9 },
      { name: "other", reference: 0, current: 0.1 },
    ]);
  });

  it("the comparison matrix lines algorithms up across jobs and leaves gaps", () => {
    const a = { id: 1, result };
    const trimmed = structuredClone(result);
    trimmed.selection!.ranking = trimmed.selection!.ranking.slice(0, 1);
    const rows = compareMatrix([a, { id: 2, result: trimmed }, { id: 3, result: null }]);
    const ridge = rows.find((r) => r.algorithm === "ridge")!;
    expect(ridge.scores[1]!.mean).toBe(result.selection!.ranking[0]!.metric_mean);
    expect(ridge.scores[2]).toBeDefined();
    expect(rows.find((r) => r.algorithm === "random_forest")!.scores[2]).toBeUndefined();
  });
});
