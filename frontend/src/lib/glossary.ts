/** Plain-language definitions, written once and shown through <Term>. Kept accurate and modest. */
export const GLOSSARY = {
  "cross-validation":
    "The training rows are split into folds. Each fold takes a turn as the unseen part while the model learns from the rest. Averaging over folds gives a fairer estimate than one split.",
  "out-of-fold":
    "A prediction made for a row while that row was in the held-out fold, so the model had not seen it. Pooled over all folds, these give honest charts.",
  "final test":
    "One last evaluation on rows held back from all training and selection. It is run once, for the chosen model only. Because several jobs can reuse the same held-out part, it can still influence choices across jobs.",
  "selection metric":
    "The score used to rank algorithms (for example F1 or RMSE). Whether higher or lower is better depends on the metric.",
  RMSE: "Root mean squared error: the typical size of a prediction error, in the units of the target. Lower is better. Large errors count extra.",
  MAE: "Mean absolute error: the average size of a prediction error, in the units of the target. Lower is better.",
  "R²": "How much of the variation in the target the model explains. 1 is perfect, 0 is no better than always guessing the average.",
  F1: "A balance of precision and recall between 0 and 1. Higher is better. Macro averages the classes equally.",
  accuracy: "The share of predictions that were correct.",
  "ROC curve":
    "Shows how well the model separates a class from the rest across all cut-offs. A curve closer to the top-left corner is better; the diagonal is guessing.",
  "PR curve": "Precision against recall across all cut-offs. Useful when one class is rare.",
  "confusion matrix":
    "Rows are the true classes, columns are what the model predicted. The diagonal is correct answers.",
  residual:
    "Actual value minus predicted value. A good model has residuals scattered evenly around zero.",
  registry:
    "The list of saved model versions and where each one stands (development, staging, production, archived).",
  staging: "A version that passed training and is being checked before it is used for real.",
  production:
    "The version the team treats as the live one. Only one version of a model is in production.",
  deployment: "Making one registered version the model that the serving app answers with.",
  rollback: "Going back to the previous version without retraining.",
  "serving app":
    "The small separate service that answers prediction requests with the deployed model.",
  feedback:
    "The true outcome for an earlier prediction, sent back later so live accuracy can be measured.",
  "live performance":
    "How the deployed model is doing on predictions whose true outcome is now known. It is measured on different rows from the offline test, so compare with care.",
  drift:
    "The data arriving now looks statistically different from the data the model was trained on. It is a reason to look, not proof that the model is wrong.",
  "reference sample":
    "A fixed sample of the training rows kept for each model version, used as the baseline that live traffic is compared with.",
  "KS test":
    "Kolmogorov–Smirnov test: compares two sets of numbers and gives a p-value. A small p-value means the two sets are unlikely to come from the same distribution.",
  "Jensen-Shannon distance":
    "A distance between two category frequency tables, from 0 (identical) to 1 (completely different).",
  "p-value":
    "How surprising the difference would be if nothing had changed. Small values (below the threshold) are flagged as drift.",
  bootstrap:
    "Re-computing a score on many random resamples of the labelled rows to see how much it could vary by chance. It stops small samples from raising false alarms.",
  "simulated traffic":
    "Demo traffic replayed from a separate monitoring file. It is always marked and kept apart from real requests.",
  alert:
    "A rule fired, for example drift above its threshold. An alert stays open until the condition clears or someone resolves it.",
  window: "The most recent predictions (up to 1,000) that one monitoring report looks at.",
  label: "The true value for a prediction.",
  target: "The column the model learns to predict.",
  "dataset version":
    "An immutable copy of an uploaded CSV. Uploading changed data creates a new version; old ones never change.",
  seed: "A number that fixes the random choices, so the same settings give the same split and results.",
} as const;

export type GlossaryKey = keyof typeof GLOSSARY;
