import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Histogram as H } from "@/api/shapes";
import { num } from "@/lib/format";
import { histogramData } from "./histogramData";

export default function Histogram({ h }: { h: H }) {
  const data = histogramData(h);
  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={data} margin={{ top: 4, right: 4, left: -18, bottom: 0 }} accessibilityLayer>
        <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
        <XAxis
          dataKey="from"
          tickFormatter={(v: number) => num(v, 2)}
          tickCount={4}
          interval="preserveStartEnd"
        />
        <YAxis allowDecimals={false} />
        <Tooltip
          formatter={(v) => [num(Number(v)), "rows"]}
          labelFormatter={(_l, p) => p[0]?.payload.label ?? ""}
        />
        <Bar dataKey="count" fill="var(--chart-1)" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
