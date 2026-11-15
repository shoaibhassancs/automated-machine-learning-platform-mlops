/** Shapes of the JSON objects the API returns as open objects (their schemas say only "object").
 *  They mirror what `train.py`, `evaluate.py`, `selection.py` and `monitor.py` produce (see
 *  docs/training.md, docs/selection-and-packaging.md, docs/monitoring.md). */

export interface MetricSummary {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
}

export interface ClassRow {
  class: string;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  support: number;
}

export interface Curve {
  class?: string;
  x?: (number | null)[];
  y?: (number | null)[];
  fpr?: (number | null)[];
  tpr?: (number | null)[];
  precision?: (number | null)[];
  recall?: (number | null)[];
  auc?: number | null;
  ap?: number | null;
  average_precision?: number | null;
  [key: string]: unknown;
}

export interface ResidualPoint {
  row: number;
  actual: number | null;
  predicted: number | null;
  residual: number | null;
}

export interface Evaluation {
  rows: number;
  metrics: Record<string, number | null>;
  undefined: Record<string, string>;
  classes?: string[];
  confusion_matrix?: number[][];
  per_class?: ClassRow[];
  roc_curves?: Curve[];
  pr_curves?: Curve[];
  curve_notes?: string[];
  residual_points?: ResidualPoint[];
  residual_points_note?: string | null;
}

export interface AlgorithmResult {
  algorithm: string;
  label: string;
  params: Record<string, unknown>;
  status: string;
  error: string | null;
  seconds: number;
  summary: Record<string, MetricSummary>;
  fold_metrics: Record<string, number | null>[];
  fold_seconds: number[];
  oof: Evaluation | null;
}

export interface TrainingResult {
  task: "classification" | "regression";
  target: string;
  classes: string[] | null;
  folds: { used: number; requested: number };
  split: { rows: number; train_rows: number; test_rows: number; stratified: boolean };
  columns: { numeric: string[]; categorical: string[]; dropped: unknown[] };
  results: AlgorithmResult[];
  seconds: number;
  warnings: string[];
  settings: { seed: number; folds: number; test_fraction: number };
}

export interface RankingRow {
  rank: number;
  algorithm: string;
  label: string;
  params: Record<string, unknown>;
  metric_mean: number;
  metric_std: number;
  pooled_oof: number | null;
  decided_by: string;
  tied_with: string[];
  secondary: { metric: string; mean: number };
}

export interface Selection {
  task: string;
  metric: string;
  direction: "higher" | "lower";
  ranking: RankingRow[];
  selected: { algorithm: string; label: string } | null;
  explanation: string;
  margin: { runner_up: string; difference: number; within_fold_std: boolean } | null;
  warnings: string[];
  excluded: { algorithm: string; reason: string }[];
  rankings_agree: boolean;
}

export interface TestResult {
  algorithm: string;
  evaluation: Evaluation;
  limitation: string;
  package_id: string;
  selection_metric: string;
  test_split: { rows: number };
}

export interface JobResult {
  dataset_ref: string;
  training: TrainingResult;
  selection: Selection | null;
  selection_note: string | null;
  test_result: TestResult | null;
  package_id: string | null;
}

export interface JobProgress {
  event?: string;
  algorithm?: string | null;
  algorithm_index?: number | null;
  algorithm_count?: number | null;
  fold?: number | null;
  folds?: number | null;
}

export interface StoredEvaluation {
  algorithm: string;
  evaluation: Evaluation;
  limitation: string;
  selection_metric: string;
  test_split: { rows: number };
}

export interface ColumnProfile {
  name: string;
  position: number;
  kind: string;
  count: number;
  missing: number;
  missing_pct: number;
  unique: number;
  top_values: { value: string; count: number }[];
  numeric: { min: number; max: number; mean: number; median: number; std: number } | null;
}

export interface Histogram {
  column: string;
  count: number;
  missing: number;
  min: number;
  max: number;
  edges: number[];
  counts: number[];
}

export interface Preview {
  dataset_ref: string;
  rows: number;
  columns: string[];
  sample_rows: Record<string, string | null>[];
  histograms: Histogram[];
  correlation: {
    columns: string[];
    matrix: (number | null)[][];
    top_pairs: { a: string; b: string; r: number }[];
  };
  class_balance: { column: string; counts: { value: string; count: number }[] } | null;
}

export interface Finding {
  severity: "error" | "warning" | "info";
  code: string;
  column: string | null;
  message: string;
  [key: string]: unknown;
}

export interface FeatureDrift {
  name: string;
  kind: string;
  method: "ks" | "js";
  status: string;
  drifted: boolean;
  statistic: number | null;
  p_value: number | null;
  threshold: number;
  current_missing_share: number;
  reference_missing_share: number;
}

export interface PerformanceComparison {
  metric: string | null;
  evaluated: boolean;
  triggered: boolean;
  verdict?: "triggered" | "clear" | "inconclusive";
  live?: number;
  offline?: number;
  relative_worsening?: number;
  threshold: number;
  window_labelled?: number;
  reason?: string;
  interval?: { lower: number | null; confidence: number; resamples: number };
}

export interface RuleResult {
  rule: string;
  severity: string;
  evaluated: boolean;
  triggered: boolean;
  details: Record<string, unknown>;
}

export interface ReportSummary {
  reason?: string;
  rules: RuleResult[];
  window: { rows: number; labelled?: number; max_rows: number };
  traffic: { scope: string; simulated: boolean; note: string };
  features?: FeatureDrift[];
  dataset_drift?: {
    share: number | null;
    drifted: boolean;
    threshold: number;
    drifted_features: number;
    evaluated_features: number;
  };
  prediction?: {
    method: "ks" | "js";
    status: string;
    statistic: number | null;
    p_value: number | null;
    threshold: number;
    drifted: boolean;
    reference: Record<string, number>;
    current: Record<string, number>;
  };
  performance?: PerformanceComparison;
  reference?: { sample_rows: number; dataset_ref: string; version: number };
}
