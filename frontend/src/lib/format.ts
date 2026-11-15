export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (Number.isInteger(value) && Math.abs(value) < 1e9) return value.toLocaleString("en-US");
  return value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toLocaleString("en-US", { maximumFractionDigits: digits })}%`;
}

export function signedPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "n/a";
  const text = pct(Math.abs(value));
  return value > 0 ? `+${text}` : value < 0 ? `−${text}` : text;
}

const dateTime = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});
const timeOnly = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit" });

export function when(iso: string | null | undefined): string {
  if (!iso) return "n/a";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : dateTime.format(d);
}

export function clock(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : timeOnly.format(d);
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "n/a";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 90) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  return `${m} min ${Math.round(seconds - m * 60)} s`;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function metricLabel(key: string): string {
  const names: Record<string, string> = {
    accuracy: "Accuracy",
    precision_macro: "Precision (macro)",
    recall_macro: "Recall (macro)",
    f1_macro: "F1 (macro)",
    roc_auc: "ROC AUC",
    rmse: "RMSE",
    mae: "MAE",
    r2: "R²",
  };
  return names[key] ?? key;
}

export const LOWER_IS_BETTER = new Set(["rmse", "mae"]);

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${num(n)} ${n === 1 ? one : many}`;
}
