import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import type { Evaluation, Selection } from "@/api/shapes";
import { ChartFrame } from "@/components/ChartFrame";
import { DataTable } from "@/components/DataTable";
import { Term } from "@/components/Term";
import { metricLabel, num, pct } from "@/lib/format";
import {
  confusionShares,
  curveSeries,
  metricDirection,
  residualPoints,
  scorePoints,
} from "./evalData";

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];

export function ScoreChart({ selection }: { selection: Selection }) {
  const data = scorePoints(selection);
  const dir = metricDirection(selection.metric);
  const best = data.find((d) => d.selected);
  return (
    <ChartFrame
      title={`Cross-validation ${metricLabel(selection.metric)} by algorithm`}
      help={
        <>
          Each bar is the average <Term k="cross-validation">cross-validation</Term> score; the
          whiskers show how much it varied between folds. {dir === "lower" ? "Lower" : "Higher"} is
          better. The selected algorithm is highlighted.
        </>
      }
      summary={`${data.length} algorithms compared. ${
        best ? `${best.label} was selected with ${num(best.mean)} (spread ${num(best.std)}).` : ""
      }`}
      table={
        <DataTable
          caption="Cross-validation scores"
          hideCaption
          rows={data}
          rowKey={(d) => d.key}
          columns={[
            { key: "a", header: "Algorithm", cell: (d) => d.label },
            {
              key: "m",
              header: `Mean ${metricLabel(selection.metric)}`,
              numeric: true,
              cell: (d) => num(d.mean, 4),
            },
            { key: "s", header: "Spread between folds", numeric: true, cell: (d) => num(d.std, 4) },
          ]}
        />
      }
    >
      <ResponsiveContainer width="100%" height={Math.max(180, data.length * 52 + 40)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
          accessibilityLayer
        >
          <CartesianGrid stroke="var(--chart-grid)" horizontal={false} />
          <XAxis type="number" domain={[0, "auto"]} />
          <YAxis type="category" dataKey="label" width={150} />
          <Tooltip formatter={(v) => num(Number(v), 4)} />
          <Bar dataKey="mean" name={metricLabel(selection.metric)} radius={[0, 4, 4, 0]}>
            {data.map((d) => (
              <Cell key={d.key} fill={d.selected ? "var(--chart-1)" : "var(--border-strong)"} />
            ))}
            <ErrorBar dataKey="std" width={6} stroke="var(--text-muted)" direction="x" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function ConfusionMatrix({ e }: { e: Evaluation }) {
  if (!e.confusion_matrix || !e.classes) return null;
  const shares = confusionShares(e.confusion_matrix);
  return (
    <ChartFrame
      title="Confusion matrix"
      help={
        <>
          Rows are the true class, columns what the model predicted. The diagonal is correct. Colour
          shows the share of each true class. See <Term k="confusion matrix">what this means</Term>.
        </>
      }
      summary={`${e.classes.length} classes. Correct predictions: ${num(
        e.confusion_matrix.reduce((s, row, i) => s + (row[i] ?? 0), 0),
      )} of ${num(e.rows)}.`}
    >
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Confusion matrix">
        <table className="confusion">
          <thead>
            <tr>
              <th scope="col">True \ Predicted</th>
              {e.classes.map((c) => (
                <th key={c} scope="col">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {e.confusion_matrix.map((row, i) => (
              <tr key={e.classes![i]}>
                <th scope="row">{e.classes![i]}</th>
                {row.map((v, j) => {
                  const share = shares[i]?.[j] ?? 0;
                  const diagonal = i === j;
                  return (
                    <td
                      key={j}
                      style={{
                        background: `color-mix(in srgb, ${diagonal ? "var(--chart-3)" : "var(--danger)"} ${Math.round(share * 70)}%, var(--surface-2))`,
                        color: "var(--text)",
                      }}
                      title={`${pct(share, 0)} of true ${e.classes![i]}`}
                    >
                      {num(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartFrame>
  );
}

function Curves({ e, kind }: { e: Evaluation; kind: "roc" | "pr" }) {
  const series = curveSeries((kind === "roc" ? e.roc_curves : e.pr_curves) ?? [], kind);
  if (series.length === 0) return null;
  const title = kind === "roc" ? "ROC curve" : "Precision–recall curve";
  const xName = kind === "roc" ? "False positive rate" : "Recall";
  const yName = kind === "roc" ? "True positive rate" : "Precision";
  const total = series.reduce((n, s) => n + s.points.length, 0);
  return (
    <ChartFrame
      title={title}
      help={
        kind === "roc" ? (
          <>
            One line per class against the rest. Closer to the top-left is better; the dashed
            diagonal is guessing. <Term k="ROC curve">More</Term>.
          </>
        ) : (
          <>
            Precision against recall for each class. Closer to the top-right is better.{" "}
            <Term k="PR curve">More</Term>.
          </>
        )
      }
      summary={`${series.length} curve${series.length === 1 ? "" : "s"} with ${num(total)} points in total.`}
      table={
        <DataTable
          caption={`${title} points`}
          hideCaption
          rows={series.flatMap((s) =>
            s.points.slice(0, 40).map((p, i) => ({ id: `${s.name}-${i}`, name: s.name, ...p })),
          )}
          rowKey={(r) => r.id}
          columns={[
            { key: "n", header: "Class", cell: (r) => r.name },
            { key: "x", header: xName, numeric: true, cell: (r) => num(r.x, 3) },
            { key: "y", header: yName, numeric: true, cell: (r) => num(r.y, 3) },
          ]}
        />
      }
    >
      <ResponsiveContainer width="100%" height={280}>
        <LineChart margin={{ top: 8, right: 16, left: 0, bottom: 20 }} accessibilityLayer>
          <CartesianGrid stroke="var(--chart-grid)" />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            name={xName}
            label={{ value: xName, position: "insideBottom", offset: -10 }}
          />
          <YAxis type="number" dataKey="y" domain={[0, 1]} name={yName} />
          <Tooltip formatter={(v) => num(Number(v), 3)} />
          {kind === "roc" && (
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke="var(--text-muted)"
              strokeDasharray="5 5"
            />
          )}
          {series.length > 1 && <Legend verticalAlign="top" />}
          {series.map((s, i) => (
            <Line
              key={s.name}
              data={s.points}
              dataKey="y"
              name={s.name}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
export const RocChart = (p: { e: Evaluation }) => <Curves e={p.e} kind="roc" />;
export const PrChart = (p: { e: Evaluation }) => <Curves e={p.e} kind="pr" />;

export function ResidualCharts({ e }: { e: Evaluation }) {
  const points = residualPoints(e.residual_points ?? []);
  if (points.length === 0) return null;
  const lo = Math.min(...points.map((p) => Math.min(p.actual, p.predicted)));
  const hi = Math.max(...points.map((p) => Math.max(p.actual, p.predicted)));
  return (
    <div className="grid-2">
      <ChartFrame
        title="Predicted against actual"
        help="Each dot is one row. Dots on the dashed diagonal were predicted exactly; the further away, the larger the error."
        summary={`${num(points.length)} rows plotted${e.residual_points_note ? `. ${e.residual_points_note}` : ""}.`}
        table={
          <DataTable
            caption="Predicted and actual values (first rows)"
            hideCaption
            rows={points.slice(0, 30).map((p, i) => ({ i, ...p }))}
            rowKey={(r) => r.i}
            columns={[
              { key: "a", header: "Actual", numeric: true, cell: (r) => num(r.actual, 3) },
              { key: "p", header: "Predicted", numeric: true, cell: (r) => num(r.predicted, 3) },
              { key: "r", header: "Residual", numeric: true, cell: (r) => num(r.residual, 3) },
            ]}
          />
        }
      >
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 20 }} accessibilityLayer>
            <CartesianGrid stroke="var(--chart-grid)" />
            <XAxis
              type="number"
              dataKey="actual"
              name="Actual"
              domain={[lo, hi]}
              tickFormatter={(v: number) => num(v, 1)}
              label={{ value: "Actual", position: "insideBottom", offset: -10 }}
            />
            <YAxis
              type="number"
              dataKey="predicted"
              name="Predicted"
              domain={[lo, hi]}
              tickFormatter={(v: number) => num(v, 1)}
            />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v) => num(Number(v), 2)} />
            <ReferenceLine
              segment={[
                { x: lo, y: lo },
                { x: hi, y: hi },
              ]}
              stroke="var(--text-muted)"
              strokeDasharray="5 5"
            />
            <Scatter
              data={points}
              fill="var(--chart-1)"
              fillOpacity={0.45}
              isAnimationActive={false}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartFrame>
      <ChartFrame
        title="Residuals"
        help={
          <>
            Actual minus predicted (<Term k="residual">residual</Term>) for each predicted value. A
            healthy model scatters evenly around the zero line, with no pattern.
          </>
        }
        summary="Residuals plotted against the predicted value."
      >
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 20 }} accessibilityLayer>
            <CartesianGrid stroke="var(--chart-grid)" />
            <XAxis
              type="number"
              dataKey="predicted"
              name="Predicted"
              domain={["auto", "auto"]}
              label={{ value: "Predicted", position: "insideBottom", offset: -10 }}
            />
            <YAxis type="number" dataKey="residual" name="Residual" />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v) => num(Number(v), 2)} />
            <ReferenceLine y={0} stroke="var(--text-muted)" strokeDasharray="5 5" />
            <Scatter
              data={points}
              fill="var(--chart-2)"
              fillOpacity={0.45}
              isAnimationActive={false}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartFrame>
    </div>
  );
}

export function PerClassTable({ e }: { e: Evaluation }) {
  if (!e.per_class?.length) return null;
  return (
    <DataTable
      caption="Per-class results"
      rows={e.per_class}
      rowKey={(r) => r.class}
      columns={[
        { key: "c", header: "Class", cell: (r) => r.class },
        { key: "p", header: "Precision", numeric: true, cell: (r) => num(r.precision, 3) },
        { key: "r", header: "Recall", numeric: true, cell: (r) => num(r.recall, 3) },
        { key: "f", header: "F1", numeric: true, cell: (r) => num(r.f1, 3) },
        { key: "s", header: "Rows", numeric: true, cell: (r) => num(r.support) },
      ]}
    />
  );
}

/** All the charts one evaluation supports, for either task. */
export function EvaluationCharts({ e, task }: { e: Evaluation; task: string }) {
  return task === "classification" ? (
    <div className="stack-lg">
      <ConfusionMatrix e={e} />
      <div className="grid-2">
        <RocChart e={e} />
        <PrChart e={e} />
      </div>
      <PerClassTable e={e} />
    </div>
  ) : (
    <ResidualCharts e={e} />
  );
}
