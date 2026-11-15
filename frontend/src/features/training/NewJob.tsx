import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { control, unwrap, type Schemas } from "@/api/client";
import { keys, useDatasets } from "@/api/queries";
import { Async, ErrorPanel } from "@/components/States";
import { Term } from "@/components/Term";
import { Callout, PageHeader, Panel, Section } from "@/components/ui";
import { metricLabel, num } from "@/lib/format";
import { CLASSIFICATION_METRICS, REGRESSION_METRICS } from "./labels";

type Algorithm = Schemas["AlgorithmModel"];
type Param = Algorithm["params"][number];

function ParamInput({
  param,
  value,
  onChange,
}: {
  param: Param;
  value: string;
  onChange: (v: string) => void;
}) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{param.name}</label>
      {param.choices && param.choices.length > 0 ? (
        <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
          {param.choices.map((c) => (
            <option key={String(c)} value={String(c)}>
              {String(c)}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type="number"
          step="any"
          value={value}
          min={param.minimum ?? undefined}
          max={param.maximum ?? undefined}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
      <span className="hint">{param.description}</span>
    </div>
  );
}

export default function NewJob() {
  const [search] = useSearchParams();
  const datasets = useDatasets();
  const algorithms = useQuery({
    queryKey: keys.algorithms,
    queryFn: () => unwrap(control.GET("/algorithms")),
  });
  const navigate = useNavigate();
  const client = useQueryClient();

  const [ref, setRef] = useState(
    search.get("dataset") && search.get("version")
      ? `${search.get("dataset")}@v${search.get("version")}`
      : "",
  );
  const [target, setTarget] = useState(search.get("target") ?? "");
  const [taskChoice, setTaskChoice] = useState<"" | "classification" | "regression">(
    (search.get("task") as "classification" | "regression" | null) ?? "",
  );
  const [excluded, setExcluded] = useState<string[]>(
    search.get("excluded")?.split(",").filter(Boolean) ?? [],
  );
  const [selected, setSelected] = useState<string[] | null>(null);
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [folds, setFolds] = useState("5");
  const [testFraction, setTestFraction] = useState("0.2");
  const [seed, setSeed] = useState("42");
  const [metric, setMetric] = useState("");
  const [kind, setKind] = useState<"train" | "final">("train");
  const [confirm, setConfirm] = useState(false);
  const [track, setTrack] = useState(true);

  const versions = useMemo(
    () => datasets.data?.datasets.flatMap((d) => d.versions) ?? [],
    [datasets.data],
  );
  const [name, number] = ref.split("@");
  const version = versions.find((v) => v.ref === ref);
  const versionNumber = Number(number?.replace("v", "") ?? NaN);

  // task-check is stateless: it tells us the task, the input columns and any problems.
  const check = useQuery({
    queryKey: ["task-check", ref, target, taskChoice, excluded.join("|")],
    enabled: !!version && !!target,
    queryFn: () =>
      unwrap(
        control.POST("/datasets/{name}/versions/{number}/task-check", {
          params: { path: { name: name!, number: versionNumber } },
          body: { target, ...(taskChoice ? { task: taskChoice } : {}), excluded },
        }),
      ),
  });
  const task = check.data?.config.task;

  // Choices that depend on the task are reset when the task changes (adjusted during render).
  const [seenTask, setSeenTask] = useState(task);
  if (seenTask !== task) {
    setSeenTask(task);
    setSelected(null);
    setOverrides({});
    setMetric("");
  }

  const submit = useMutation({
    mutationFn: () => {
      const choice = (selected ?? []).map((key) => {
        const algo = algorithms.data?.algorithms.find((a) => a.key === key);
        const params: Record<string, unknown> = {};
        for (const p of algo?.params ?? []) {
          const raw = overrides[`${key}.${p.name}`];
          if (raw !== undefined && raw !== String(p.default)) {
            params[p.name] = p.choices && p.choices.length ? raw : Number(raw);
          }
        }
        return { key, params };
      });
      return unwrap(
        control.POST("/jobs", {
          body: {
            kind,
            dataset: name!,
            version: versionNumber,
            target,
            ...(taskChoice ? { task: taskChoice } : {}),
            excluded,
            ...(selected && selected.length ? { algorithms: choice } : {}),
            folds: Number(folds),
            test_fraction: Number(testFraction),
            seed: Number(seed),
            ...(metric ? { selection_metric: metric } : {}),
            confirm_final_test: kind === "final" && confirm,
            track,
          },
        }),
      );
    },
    onSuccess: async (job) => {
      await client.invalidateQueries({ queryKey: ["jobs"] });
      navigate(`/training/${job.id}`);
    },
  });

  const available = (algorithms.data?.algorithms ?? []).filter((a) => !task || a.task === task);
  const chosen = selected ?? available.map((a) => a.key);
  const metrics =
    task === "classification"
      ? CLASSIFICATION_METRICS
      : task === "regression"
        ? REGRESSION_METRICS
        : [];
  const blocked =
    !version || !target || !check.data?.ok || chosen.length === 0 || (kind === "final" && !confirm);

  return (
    <div className="stack-lg">
      <PageHeader
        crumbs={[{ to: "/training", label: "Training" }, { label: "New job" }]}
        title="New training job"
        lead="Choose the data and what to predict, the algorithms to try, and how carefully to validate. The job runs in the background; you can leave this page."
      />
      <form
        className="stack-lg"
        onSubmit={(e) => {
          e.preventDefault();
          if (!blocked) submit.mutate();
        }}
      >
        <Panel>
          <h2>1. Data and task</h2>
          <div className="form-grid" style={{ marginTop: 12 }}>
            <div className="field">
              <label htmlFor="ds">Dataset version</label>
              <select
                id="ds"
                value={ref}
                onChange={(e) => {
                  setRef(e.target.value);
                  setTarget("");
                  setExcluded([]);
                }}
              >
                <option value="">Choose a dataset…</option>
                {versions.map((v) => (
                  <option key={v.ref} value={v.ref}>
                    {v.ref} ({num(v.rows)} rows)
                  </option>
                ))}
              </select>
              {datasets.isSuccess && versions.length === 0 && (
                <span className="hint">
                  No datasets yet. Upload one on the Datasets page first.
                </span>
              )}
            </div>
            <div className="field">
              <label htmlFor="tg">Target column (what to predict)</label>
              <select
                id="tg"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                disabled={!version}
              >
                <option value="">Choose a column…</option>
                {version?.columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="tk">Task</label>
              <select
                id="tk"
                value={taskChoice}
                onChange={(e) => setTaskChoice(e.target.value as typeof taskChoice)}
              >
                <option value="">Detect automatically</option>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>
          </div>
          {version && target && (
            <fieldset style={{ marginTop: 16 }}>
              <legend>Leave out of the model (optional)</legend>
              <div className="row">
                {version.columns
                  .filter((c) => c !== target)
                  .map((c) => (
                    <label key={c} className="check">
                      <input
                        type="checkbox"
                        checked={excluded.includes(c)}
                        onChange={(e) =>
                          setExcluded((cur) =>
                            e.target.checked ? [...cur, c] : cur.filter((x) => x !== c),
                          )
                        }
                      />
                      {c}
                    </label>
                  ))}
              </div>
            </fieldset>
          )}
          <div style={{ marginTop: 16 }} aria-live="polite">
            {check.isFetching && <span className="muted">Checking the setup…</span>}
            {check.isError && (
              <ErrorPanel error={check.error} title="The setup could not be checked" />
            )}
            {check.data &&
              (check.data.ok ? (
                <Callout tone="ok" title={`Ready: ${check.data.config.task}`}>
                  <p>
                    {num(check.data.feature_columns.length)} input columns.{" "}
                    {check.data.suggestion?.reason}
                  </p>
                </Callout>
              ) : (
                <Callout tone="bad" title="This setup has problems">
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {check.data.findings
                      .filter((f) => f.severity === "error")
                      .map((f, i) => (
                        <li key={i}>{f.message}</li>
                      ))}
                  </ul>
                </Callout>
              ))}
          </div>
        </Panel>

        <Panel>
          <h2>2. Algorithms</h2>
          <p className="muted" style={{ margin: "4px 0 12px" }}>
            Each selected algorithm is cross-validated and compared. Leave parameters at their
            defaults unless you have a reason.
          </p>
          <Async query={algorithms} label="Loading algorithms">
            {() =>
              !task ? (
                <p className="muted">
                  Choose a dataset and a target to see the algorithms for its task.
                </p>
              ) : (
                <div className="stack">
                  {available.map((a) => (
                    <div key={a.key} className="stack" style={{ gap: 8 }}>
                      <label className="check">
                        <input
                          type="checkbox"
                          checked={chosen.includes(a.key)}
                          onChange={(e) =>
                            setSelected(
                              e.target.checked
                                ? [...chosen, a.key]
                                : chosen.filter((k) => k !== a.key),
                            )
                          }
                        />
                        <strong>{a.label}</strong>
                      </label>
                      {chosen.includes(a.key) && a.params.length > 0 && (
                        <details style={{ marginLeft: 28 }}>
                          <summary>Parameters</summary>
                          <div className="form-grid" style={{ marginTop: 10 }}>
                            {a.params.map((p) => (
                              <ParamInput
                                key={p.name}
                                param={p}
                                value={overrides[`${a.key}.${p.name}`] ?? String(p.default)}
                                onChange={(v) =>
                                  setOverrides((cur) => ({ ...cur, [`${a.key}.${p.name}`]: v }))
                                }
                              />
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              )
            }
          </Async>
        </Panel>

        <Panel>
          <h2>3. Validation settings</h2>
          <div className="form-grid" style={{ marginTop: 12 }}>
            <div className="field">
              <label htmlFor="folds">Folds</label>
              <input
                id="folds"
                type="number"
                min={2}
                max={20}
                value={folds}
                onChange={(e) => setFolds(e.target.value)}
              />
              <span className="hint">
                <Term k="cross-validation">Cross-validation</Term> splits: more folds are slower but
                steadier.
              </span>
            </div>
            <div className="field">
              <label htmlFor="tf">Held-out test share</label>
              <input
                id="tf"
                type="number"
                step="0.05"
                min={0.05}
                max={0.5}
                value={testFraction}
                onChange={(e) => setTestFraction(e.target.value)}
              />
              <span className="hint">
                Rows kept back for the final test (never used for training or selection).
              </span>
            </div>
            <div className="field">
              <label htmlFor="seed">
                <Term k="seed">Random seed</Term>
              </label>
              <input
                id="seed"
                type="number"
                min={0}
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="metric">Selection metric</label>
              <select
                id="metric"
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                disabled={!task}
              >
                <option value="">Default for the task</option>
                {metrics.map((m) => (
                  <option key={m} value={m}>
                    {metricLabel(m)}
                  </option>
                ))}
              </select>
              <span className="hint">
                The <Term k="selection metric">score</Term> used to pick the winner.
              </span>
            </div>
          </div>
        </Panel>

        <Panel>
          <h2>4. What should this job do?</h2>
          <fieldset style={{ marginTop: 12 }}>
            <legend className="sr-only">Job type</legend>
            <div className="stack">
              <label className="check">
                <input
                  type="radio"
                  name="kind"
                  checked={kind === "train"}
                  onChange={() => setKind("train")}
                />
                <span>
                  <strong>Cross-validate and compare</strong>
                  <br />
                  <span className="muted">Safe to repeat. Never reads the held-out test rows.</span>
                </span>
              </label>
              <label className="check">
                <input
                  type="radio"
                  name="kind"
                  checked={kind === "final"}
                  onChange={() => setKind("final")}
                />
                <span>
                  <strong>Cross-validate, select, and run the final test</strong>
                  <br />
                  <span className="muted">
                    Also reads the held-out rows once, for the selected model only, and saves a
                    model package you can register.
                  </span>
                </span>
              </label>
            </div>
          </fieldset>
          {kind === "final" && (
            <div style={{ marginTop: 14 }}>
              <Callout tone="warn" title="The final test is read once">
                <p>
                  The held-out rows are used one time for the chosen model. Reusing the same
                  held-out part across several jobs can still influence which models you pick, so
                  treat this result as a final check, not as independent proof. See{" "}
                  <Term k="final test">final test</Term>.
                </p>
                <label className="check" style={{ marginTop: 8 }}>
                  <input
                    type="checkbox"
                    checked={confirm}
                    onChange={(e) => setConfirm(e.target.checked)}
                  />
                  <span>I understand and want to run the final test.</span>
                </label>
              </Callout>
            </div>
          )}
          <label className="check" style={{ marginTop: 16 }}>
            <input type="checkbox" checked={track} onChange={(e) => setTrack(e.target.checked)} />
            <span>
              Log this job to MLflow (experiment tracking)
              <br />
              <span className="muted">
                Needs the worker to have tracking configured; otherwise the job is refused with a
                clear message.
              </span>
            </span>
          </label>
        </Panel>

        {submit.isError && <ErrorPanel error={submit.error} title="The job was not started" />}
        <div className="row">
          <button type="submit" className="btn primary" disabled={blocked || submit.isPending}>
            {submit.isPending ? "Starting…" : "Start training"}
          </button>
          {blocked && <span className="muted">Complete the setup above to continue.</span>}
        </div>
      </form>
      <Section title="What happens next">
        <p className="muted">
          The worker picks up the job, cross-validates each algorithm fold by fold, applies the
          selection rule and, for a final job, runs the one-time test. You will see live progress on
          the next page.
        </p>
      </Section>
    </div>
  );
}
