import type { AlgorithmResult, Curve, Evaluation, ResidualPoint, Selection } from "@/api/shapes";
import { LOWER_IS_BETTER } from "@/lib/format";

/** Pure data-preparation for the charts, kept apart from the rendering so tests can check that the
 *  plotted numbers equal the API numbers. */

export interface ScorePoint {
  key: string;
  label: string;
  mean: number;
  std: number;
  selected: boolean;
}

export function scorePoints(selection: Selection): ScorePoint[] {
  return selection.ranking.map((r) => ({
    key: r.algorithm,
    label: r.label,
    mean: r.metric_mean,
    std: r.metric_std,
    selected: selection.selected?.algorithm === r.algorithm,
  }));
}

export function metricDirection(metric: string): "lower" | "higher" {
  return LOWER_IS_BETTER.has(metric) ? "lower" : "higher";
}

export interface LinePoint {
  x: number;
  y: number;
}

/** One series per class: x/y pairs with missing points dropped. */
export function curveSeries(
  curves: Curve[],
  kind: "roc" | "pr",
): { name: string; points: LinePoint[] }[] {
  return curves.map((c, i) => {
    const xs = (kind === "roc" ? c.fpr : c.recall) ?? [];
    const ys = (kind === "roc" ? c.tpr : c.precision) ?? [];
    const points: LinePoint[] = [];
    xs.forEach((x, j) => {
      const y = ys[j];
      if (x !== null && x !== undefined && y !== null && y !== undefined) points.push({ x, y });
    });
    return { name: c.class ?? `curve ${i + 1}`, points };
  });
}

export function residualPoints(points: ResidualPoint[]): {
  actual: number;
  predicted: number;
  residual: number;
}[] {
  const out: { actual: number; predicted: number; residual: number }[] = [];
  for (const p of points) {
    if (p.actual !== null && p.predicted !== null && p.residual !== null) {
      out.push({ actual: p.actual, predicted: p.predicted, residual: p.residual });
    }
  }
  return out;
}

/** Share of each true class predicted as each class (row-normalised confusion matrix). */
export function confusionShares(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const total = row.reduce((a, b) => a + b, 0);
    return row.map((v) => (total ? v / total : 0));
  });
}

export function headlineMetric(result: AlgorithmResult, metric: string): number | null {
  return result.summary[metric]?.mean ?? null;
}

export function hasCurves(e: Evaluation): boolean {
  return (e.roc_curves?.length ?? 0) > 0;
}
