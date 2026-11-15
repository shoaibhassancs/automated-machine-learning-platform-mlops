import { useQuery } from "@tanstack/react-query";
import { control, unwrap } from "@/api/client";
import type { ReportSummary } from "@/api/shapes";
import { Segmented, SimulatedBadge } from "@/components/ui";

export type Scope = "real" | "simulated";
export type Range = "1h" | "24h" | "7d" | "all";

export const RULES: Record<string, { name: string; explain: string }> = {
  drifted_feature_share: {
    name: "Share of drifted features",
    explain:
      "How many input columns now look different from the training data. The alert fires when half or more have drifted.",
  },
  prediction_shift: {
    name: "Prediction shift",
    explain:
      "The model's answers are distributed differently from how they were on the training sample.",
  },
  performance_drop: {
    name: "Performance drop",
    explain:
      "On rows whose true value is known, the live score is more than 10% worse than the offline test score, and not just by chance.",
  },
};

export function ruleName(rule: string): string {
  return RULES[rule]?.name ?? rule;
}

export function ScopeSwitch({ value, onChange }: { value: Scope; onChange: (s: Scope) => void }) {
  return (
    <Segmented
      label="Traffic to show"
      value={value}
      onChange={onChange}
      options={[
        { value: "real", label: "Real requests" },
        { value: "simulated", label: "Simulated demo traffic" },
      ]}
    />
  );
}

export function RangeSwitch({ value, onChange }: { value: Range; onChange: (r: Range) => void }) {
  return (
    <Segmented
      label="Time range"
      value={value}
      onChange={onChange}
      options={[
        { value: "1h", label: "Last hour" },
        { value: "24h", label: "24 hours" },
        { value: "7d", label: "7 days" },
        { value: "all", label: "All time" },
      ]}
    />
  );
}

export function rangeSince(range: Range, now = Date.now()): string | undefined {
  const hours = { "1h": 1, "24h": 24, "7d": 168, all: 0 }[range];
  return hours ? new Date(now - hours * 3_600_000).toISOString() : undefined;
}

export function ScopeBadge({ scope }: { scope: string }) {
  return scope === "simulated" ? <SimulatedBadge /> : <span className="badge">Real</span>;
}

/** The full summary of the latest report for a scope (the list endpoints omit summaries). */
export function useReport(id: number | null | undefined) {
  return useQuery({
    queryKey: ["report", id],
    enabled: !!id,
    queryFn: () =>
      unwrap(control.GET("/reports/{report_id}", { params: { path: { report_id: id! } } })),
  });
}

export function summaryOf(r: { summary: unknown } | undefined): ReportSummary | null {
  return (r?.summary ?? null) as ReportSummary | null;
}
