export function jobKindLabel(kind: string): string {
  return kind === "final" ? "Final run (with test)" : "Cross-validation only";
}

const PHASES: Record<string, string> = {
  queued: "Waiting for the worker",
  loading: "Loading the data",
  training: "Cross-validating algorithms",
  selecting: "Applying the selection rule",
  finalizing: "Running the final test",
  tracking: "Logging to MLflow",
  done: "Done",
};
export function phaseLabel(phase: string): string {
  return PHASES[phase] ?? phase;
}

export const PHASE_ORDER = [
  "queued",
  "loading",
  "training",
  "selecting",
  "finalizing",
  "tracking",
  "done",
];

export const CLASSIFICATION_METRICS = [
  "f1_macro",
  "accuracy",
  "precision_macro",
  "recall_macro",
  "roc_auc",
];
export const REGRESSION_METRICS = ["rmse", "mae", "r2"];
