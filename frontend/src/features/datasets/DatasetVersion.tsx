import { useMutation, useQuery } from "@tanstack/react-query";
import { lazy, Suspense, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { control, unwrap, type Schemas } from "@/api/client";
import { keys } from "@/api/queries";
import type { ColumnProfile, Finding, Histogram as H, Preview } from "@/api/shapes";
import { ChartFrame } from "@/components/ChartFrame";
import { DataTable } from "@/components/DataTable";
import { Async, ErrorPanel, Loading } from "@/components/States";
import { Term } from "@/components/Term";
import {
  Badge,
  BarCell,
  Callout,
  KV,
  Metric,
  NavTabs,
  PageHeader,
  Panel,
  Section,
  StatusBadge,
} from "@/components/ui";
import { bytes, num, when } from "@/lib/format";
import { histogramData } from "./histogramData";

const Histogram = lazy(() => import("./Histogram"));

type Version = Schemas["VersionModel"];

function useVersionPart<T>(
  name: string,
  number: number,
  part: "preview" | "profile" | "validation",
) {
  return useQuery<T, Error>({
    queryKey: [part, name, number],
    queryFn: () =>
      unwrap(
        control.GET(
          `/datasets/{name}/versions/{number}/${part}` as "/datasets/{name}/versions/{number}/preview",
          {
            params: { path: { name, number } },
          },
        ),
      ) as Promise<T>,
  });
}

function PreviewTab({ name, number, version }: { name: string; number: number; version: Version }) {
  const query = useVersionPart<Preview>(name, number, "preview");
  return (
    <div className="stack-lg">
      <Panel>
        <div className="metrics">
          <Metric label="Rows" value={num(version.rows)} />
          <Metric label="Columns" value={num(version.columns.length)} />
          <Metric label="Size" value={bytes(version.size_bytes)} />
          <Metric
            label="Uploaded"
            value={<span style={{ fontSize: "1rem" }}>{when(version.created_at)}</span>}
            sub={version.source_filename}
          />
        </div>
        <p className="muted" style={{ marginTop: 14 }}>
          <Term k="dataset version">Dataset version</Term> checksum{" "}
          <code>{version.sha256.slice(0, 16)}…</code>. This version is read-only: uploading changed
          data creates a new version.
        </p>
      </Panel>
      <Async query={query} label="Loading preview">
        {(p) => (
          <>
            <Section
              title="Sample rows"
              lead={`The first ${p.sample_rows.length} rows, exactly as stored (values are shown as text).`}
            >
              <Panel flush>
                <DataTable
                  caption="Sample rows"
                  hideCaption
                  rows={p.sample_rows.map((r, i) => ({ r, i }))}
                  rowKey={(x) => x.i}
                  columns={p.columns.map((c) => ({
                    key: c,
                    header: c,
                    cell: (x: { r: Record<string, string | null> }) =>
                      x.r[c] ?? <span className="muted">(empty)</span>,
                  }))}
                />
              </Panel>
            </Section>
            {p.histograms.length > 0 && (
              <Section
                title="Distributions"
                lead="How the values of each numeric column are spread. Tall bars show where most rows sit."
              >
                <div className="grid-2">
                  {p.histograms.map((h) => (
                    <HistogramPanel key={h.column} h={h} />
                  ))}
                </div>
              </Section>
            )}
            {p.correlation.top_pairs.length > 0 && (
              <Section
                title="Strongly related columns"
                lead="Pairs of numeric columns that move together (correlation near 1 or -1). Very strong pairs carry the same information twice."
              >
                <Panel flush>
                  <DataTable
                    caption="Most correlated column pairs"
                    hideCaption
                    rows={p.correlation.top_pairs.slice(0, 6)}
                    rowKey={(x) => `${x.a}|${x.b}`}
                    columns={[
                      { key: "a", header: "Column", cell: (x) => x.a },
                      { key: "b", header: "Related to", cell: (x) => x.b },
                      { key: "r", header: "Correlation", numeric: true, cell: (x) => num(x.r, 2) },
                    ]}
                  />
                </Panel>
              </Section>
            )}
            {p.class_balance && (
              <Section
                title="Class balance"
                lead={`How often each value of ${p.class_balance.column} occurs.`}
              >
                <Panel>
                  <DataTable
                    caption="Class balance"
                    hideCaption
                    rows={p.class_balance.counts}
                    rowKey={(x) => x.value}
                    columns={[
                      { key: "v", header: "Value", cell: (x) => x.value },
                      { key: "c", header: "Rows", numeric: true, cell: (x) => num(x.count) },
                    ]}
                  />
                </Panel>
              </Section>
            )}
          </>
        )}
      </Async>
    </div>
  );
}

function HistogramPanel({ h }: { h: H }) {
  const data = histogramData(h);
  const top = data.reduce((best, d) => (d.count > best.count ? d : best), data[0]!);
  return (
    <Panel>
      <ChartFrame
        title={h.column}
        summary={`${num(h.count)} values from ${num(h.min, 3)} to ${num(h.max, 3)}; most are between ${top.label}.`}
        table={
          <DataTable
            caption={`${h.column} histogram`}
            hideCaption
            rows={data}
            rowKey={(d) => d.label}
            columns={[
              { key: "r", header: "Range", cell: (d) => d.label },
              { key: "c", header: "Rows", numeric: true, cell: (d) => num(d.count) },
            ]}
          />
        }
      >
        <Suspense fallback={<Loading rows={2} />}>
          <Histogram h={h} />
        </Suspense>
      </ChartFrame>
    </Panel>
  );
}

function ProfileTab({ name, number }: { name: string; number: number }) {
  const query = useVersionPart<{
    rows: number;
    columns: number;
    duplicate_rows: number;
    rows_with_missing: number;
    missing_cells: number;
    column_profiles: ColumnProfile[];
  }>(name, number, "profile");
  return (
    <Async query={query} label="Loading profile">
      {(p) => (
        <div className="stack-lg">
          <Panel>
            <div className="metrics">
              <Metric label="Rows" value={num(p.rows)} />
              <Metric label="Columns" value={num(p.columns)} />
              <Metric
                label="Rows with a missing value"
                value={num(p.rows_with_missing)}
                sub={`${num(p.missing_cells)} empty cells in total`}
              />
              <Metric label="Duplicate rows" value={num(p.duplicate_rows)} />
            </div>
          </Panel>
          <Section
            title="Columns"
            lead="One row per column: its detected type, how much is missing, how many distinct values it has, and a summary of its values. Only an empty cell counts as missing."
          >
            <Panel flush>
              <DataTable
                caption="Column profile"
                hideCaption
                rows={p.column_profiles}
                rowKey={(c) => c.name}
                columns={[
                  { key: "n", header: "Column", cell: (c) => <strong>{c.name}</strong> },
                  { key: "k", header: "Type", cell: (c) => <Badge>{c.kind}</Badge> },
                  {
                    key: "m",
                    header: "Missing",
                    cell: (c) => (
                      <BarCell
                        value={c.missing_pct / 100}
                        text={`${num(c.missing_pct, 1)}%`}
                        hot={c.missing_pct >= 40}
                      />
                    ),
                  },
                  { key: "u", header: "Distinct", numeric: true, cell: (c) => num(c.unique) },
                  {
                    key: "s",
                    header: "Summary",
                    cell: (c) =>
                      c.numeric ? (
                        <span className="num">
                          {num(c.numeric.min, 3)} to {num(c.numeric.max, 3)}, median{" "}
                          {num(c.numeric.median, 3)}
                        </span>
                      ) : (
                        <span>
                          {c.top_values
                            .slice(0, 3)
                            .map((v) => `${v.value} (${num(v.count)})`)
                            .join(", ")}
                        </span>
                      ),
                  },
                ]}
              />
            </Panel>
          </Section>
        </div>
      )}
    </Async>
  );
}

function FindingsTable({ findings }: { findings: Finding[] }) {
  if (findings.length === 0) return <p className="muted">No findings.</p>;
  return (
    <DataTable
      caption="Findings"
      hideCaption
      rows={findings.map((f, i) => ({ ...f, i }))}
      rowKey={(f) => f.i}
      columns={[
        {
          key: "s",
          header: "Severity",
          cell: (f) => (
            <StatusBadge
              status={
                f.severity === "error" ? "error" : f.severity === "warning" ? "warning" : "info"
              }
              label={f.severity}
            />
          ),
        },
        {
          key: "c",
          header: "Column",
          cell: (f) => f.column ?? <span className="muted">whole dataset</span>,
        },
        { key: "m", header: "What was found", cell: (f) => f.message },
      ]}
    />
  );
}

function ValidationTab({ name, number }: { name: string; number: number }) {
  const query = useVersionPart<{
    ok: boolean;
    errors: number;
    warnings: number;
    infos: number;
    findings: Finding[];
  }>(name, number, "validation");
  return (
    <Async query={query} label="Loading validation">
      {(v) => (
        <div className="stack-lg">
          {v.ok ? (
            <Callout tone="ok" title="No blocking problems">
              <p>
                The data passed the checks that must hold before training. Warnings are worth
                reading but do not stop you.
              </p>
            </Callout>
          ) : (
            <Callout tone="bad" title="This dataset has errors">
              <p>Fix these in the source file and upload a new version before training.</p>
            </Callout>
          )}
          <Panel>
            <div className="metrics">
              <Metric label="Errors" value={num(v.errors)} />
              <Metric label="Warnings" value={num(v.warnings)} />
              <Metric label="Notes" value={num(v.infos)} />
            </div>
          </Panel>
          <Panel flush>
            <FindingsTable findings={v.findings} />
          </Panel>
        </div>
      )}
    </Async>
  );
}

function TaskTab({ name, number, version }: { name: string; number: number; version: Version }) {
  const [target, setTarget] = useState("");
  const [task, setTask] = useState<"" | "classification" | "regression">("");
  const [excluded, setExcluded] = useState<string[]>([]);
  const check = useMutation({
    mutationFn: () =>
      unwrap(
        control.POST("/datasets/{name}/versions/{number}/task-check", {
          params: { path: { name, number } },
          body: { target, ...(task ? { task } : {}), excluded },
        }),
      ),
  });
  const columns = version.columns;
  const result = check.data;
  const params = new URLSearchParams({ dataset: name, version: String(number), target });
  if (result) params.set("task", result.config.task);
  if (excluded.length) params.set("excluded", excluded.join(","));

  return (
    <div className="stack-lg">
      <Panel>
        <form
          className="stack"
          onSubmit={(e) => {
            e.preventDefault();
            if (target) check.mutate();
          }}
        >
          <h2>Set up the prediction task</h2>
          <p className="muted">
            Choose the <Term k="target">target</Term> column the model should predict. The platform
            suggests whether that is classification (a category) or regression (a number) and checks
            the data is suitable. Nothing is stored here: the setup is carried into the training
            form.
          </p>
          <div className="form-grid">
            <div className="field">
              <label htmlFor="target">Target column</label>
              <select
                id="target"
                value={target}
                onChange={(e) => {
                  setTarget(e.target.value);
                  check.reset();
                }}
              >
                <option value="">Choose a column…</option>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="task">Task type</label>
              <select
                id="task"
                value={task}
                onChange={(e) => {
                  setTask(e.target.value as typeof task);
                  check.reset();
                }}
              >
                <option value="">Suggest one for me</option>
                <option value="classification">Classification (predict a category)</option>
                <option value="regression">Regression (predict a number)</option>
              </select>
            </div>
          </div>
          <fieldset>
            <legend>Columns to leave out of the model (optional)</legend>
            <div className="row">
              {columns
                .filter((c) => c !== target)
                .map((c) => (
                  <label key={c} className="check">
                    <input
                      type="checkbox"
                      checked={excluded.includes(c)}
                      onChange={(e) => {
                        setExcluded((cur) =>
                          e.target.checked ? [...cur, c] : cur.filter((x) => x !== c),
                        );
                        check.reset();
                      }}
                    />
                    {c}
                  </label>
                ))}
            </div>
          </fieldset>
          <div>
            <button type="submit" className="btn primary" disabled={!target || check.isPending}>
              {check.isPending ? "Checking…" : "Check this setup"}
            </button>
          </div>
        </form>
      </Panel>
      {check.isError && <ErrorPanel error={check.error} title="The setup could not be checked" />}
      {result && (
        <Section title="Result">
          <div className="stack">
            {result.ok ? (
              <Callout tone="ok" title="Ready to train">
                <p>
                  Task: <strong>{result.config.task}</strong>, target{" "}
                  <strong>{result.config.target}</strong>, {num(result.feature_columns.length)}{" "}
                  input columns.
                </p>
              </Callout>
            ) : (
              <Callout tone="bad" title="This setup has problems">
                <p>
                  Resolve the errors below (change the target, task or excluded columns) and check
                  again.
                </p>
              </Callout>
            )}
            {result.suggestion && (
              <p className="muted">
                Suggested task: <strong>{result.suggestion.task}</strong>.{" "}
                {result.suggestion.reason}
              </p>
            )}
            {result.findings.length > 0 && (
              <Panel flush>
                <FindingsTable findings={result.findings as Finding[]} />
              </Panel>
            )}
            <KV items={[["Input columns", result.feature_columns.join(", ")]]} />
            {result.ok && (
              <div>
                <Link to={`/training/new?${params.toString()}`} className="btn primary">
                  Train on this dataset →
                </Link>
              </div>
            )}
          </div>
        </Section>
      )}
    </div>
  );
}

export default function DatasetVersion() {
  const { name = "", number: raw = "1", "*": tab = "" } = useParams();
  const number = Number(raw);
  const query = useQuery({
    queryKey: keys.version(name, number),
    queryFn: () =>
      unwrap(
        control.GET("/datasets/{name}/versions/{number}", { params: { path: { name, number } } }),
      ),
  });
  const base = `/datasets/${name}/v/${number}`;
  return (
    <div className="stack-lg">
      <PageHeader
        crumbs={[{ to: "/datasets", label: "Datasets" }, { label: `${name} v${number}` }]}
        title={`${name} · version ${number}`}
        lead="Look at the data before trusting it: what is in it, how complete it is, whether it passes the checks, and what you want to predict."
      />
      <Async query={query} label="Loading dataset">
        {(version) => (
          <div>
            <NavTabs
              label="Dataset sections"
              tabs={[
                { to: base, label: "Preview" },
                { to: `${base}/profile`, label: "Profile" },
                { to: `${base}/validation`, label: "Validation" },
                { to: `${base}/task`, label: "Task setup" },
              ]}
            />
            {tab === "profile" ? (
              <ProfileTab name={name} number={number} />
            ) : tab === "validation" ? (
              <ValidationTab name={name} number={number} />
            ) : tab === "task" ? (
              <TaskTab name={name} number={number} version={version} />
            ) : (
              <PreviewTab name={name} number={number} version={version} />
            )}
            <p className="muted" style={{ marginTop: 24, fontSize: "0.85rem" }}>
              Missing values: only an empty cell counts as missing; text like "NA" is kept as
              written and reported.
            </p>
          </div>
        )}
      </Async>
    </div>
  );
}
