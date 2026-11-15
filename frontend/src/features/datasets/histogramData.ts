import type { Histogram as H } from "@/api/shapes";
import { num } from "@/lib/format";

export function histogramData(h: H): { label: string; from: number; to: number; count: number }[] {
  return h.counts.map((count, i) => {
    const from = h.edges[i] ?? 0;
    const to = h.edges[i + 1] ?? from;
    return { label: `${num(from, 2)} to ${num(to, 2)}`, from, to, count };
  });
}
