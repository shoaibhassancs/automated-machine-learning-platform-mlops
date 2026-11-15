import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ApiError, control, unwrap } from "@/api/client";
import type { Evaluation, JobProgress, JobResult } from "@/api/shapes";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable } from "@/components/DataTable";
import { Async } from "@/components/States";
import { Term } from "@/components/Term";
import { useToast } from "@/components/Toast";
import {
  Badge,
  Callout,
  KV,
  Metric,
  PageHeader,
  Panel,
  Section,
  StatusBadge,
} from "@/components/ui";
import { duration, metricLabel, num, when } from "@/lib/format";
import { EvaluationCharts, ScoreChart } from "./EvalCharts";
import { jobKindLabel, PHASE_ORDER, phaseLabel } from "./labels";

function progressFraction(p: JobProgress | null): number {
  if (!p || !p.algorithm_count) return 0;
  const done = Math.max(0, (p.algorithm_index ?? 1) - 1);
  const inside =
    p.fold && p.folds ? (p.fold / p.folds) * 0.9 : p.event === "algorithm_finished" ? 1 : 0;
  return Math.min(1, (done + inside) / p.algorithm_count);
}

function MetricsRow({ e, keysToShow }: { e: Evaluation; keysToShow?: string[] }) {
  const entries = Object.entries(e.metrics).filter(([k]) => !keysToShow || keysToShow.includes(k));
  return (
    <div className="metrics">
      {entries.map(([k, v]) => (
        <Metric
          key={k}
          label={metricLabel(k)}
          value={num(v, 3)}
          sub={e.undefined[k] ?? undefined}
        />
      ))}
      <Metric label="Rows scored" value={num(e.rows)} />
    </div>
  );
}

function Results({ jobId, result, kind }: { jobId: number; result: JobResult; kind: string }) {
  const training = result.training;
  const selection = result.selection;
  const [shown, setShown] = useState<string | null>(null);
  const ok = training.results.filter((r) => r.status === "ok" && r.oof);
  const current =
    ok.find((r) => r.algorithm === (shown ?? selection?.selected?.algorithm)) ?? ok[0];
  const [registering, setRegistering] = useState(false);

  return (
    <div className="stack-lg">
      {selection ? (
        <Section
          title="Which algorithm won"
          lead={
            <>
              Algorithms are ranked by their average{" "}
              <Term k="cross-validation">cross-validation</Term> score on the training rows only.
              The rule below decided the winner; the test rows played no part.
            </>
          }
        >
          <div className="stack">
            <Callout
              tone="ok"
              title={
                selection.selected
                  ? `Selected: ${selection.selected.label}`
                  : "No algorithm was selected"
              }
            >
              <p>{selection.explanation}</p>
              {selection.margin && (
                <p>
                  Margin over {selection.margin.runner_up}: {num(selection.margin.difference, 4)}
                  {selection.margin.within_fold_std
                    ? ". That is smaller than the variation between folds, so the two are close."
                    : "."}
                </p>
              )}
              {!selection.rankings_agree && (
                <p>
                  The two ways of scoring (fold average and pooled predictions) disagreed on the
                  order.
                </p>
              )}
            </Callout>
            <ScoreChart selection={selection} />
            <Panel flush>
              <DataTable
                caption="Ranking"
                rows={selection.ranking}
                rowKey={(r) => r.algorithm}
                columns={[
                  { key: "r", header: "Rank", numeric: true, cell: (r) => r.rank },
                  { key: "a", header: "Algorithm", cell: (r) => <strong>{r.label}</strong> },
                  {
                    key: "m",
                    header: metricLabel(selection.metric),
                    numeric: true,
                    cell: (r) => (
                      <span className="nowrap">{`${num(r.metric_mean, 4)} ± ${num(r.metric_std, 4)}`}</span>
                    ),
                  },
                  {
                    key: "o",
                    header: "Pooled out-of-fold",
                    numeric: true,
                    cell: (r) => num(r.pooled_oof, 4),
                  },
                  {
                    key: "s",
                    header: metricLabel(r0(selection)),
                    numeric: true,
                    cell: (r) => num(r.secondary.mean, 4),
                  },
                  {
                    key: "p",
                    header: "Parameters",
                    cell: (r) =>
                      Object.keys(r.params).length ? (
                        <span className="mono">{JSON.stringify(r.params)}</span>
                      ) : (
                        <span className="muted">defaults</span>
                      ),
                  },
                ]}
              />
            </Panel>
          </div>
        </Section>
      ) : (
        <Callout tone="warn" title="No selection was made">
          <p>{result.selection_note ?? "Every algorithm failed, so there is nothing to select."}</p>
        </Callout>
      )}

      {training.results.some((r) => r.status !== "ok") && (
        <Callout tone="warn" title="Some algorithms did not finish">
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {training.results
              .filter((r) => r.status !== "ok")
              .map((r) => (
                <li key={r.algorithm}>
                  {r.label}: {r.error ?? r.status}
                </li>
              ))}
          </ul>
        </Callout>
      )}

      {current?.oof && (
        <Section
          title="How each algorithm did on unseen folds"
          lead={
            <>
              These charts use <Term k="out-of-fold">out-of-fold</Term> predictions: every training
              row is predicted by a model that never saw it. Choose an algorithm to inspect.
            </>
          }
        >
          <div className="stack">
            <div className="field" style={{ maxWidth: 320 }}>
              <label htmlFor="algo">Algorithm</label>
              <select
                id="algo"
                value={current.algorithm}
                onChange={(e) => setShown(e.target.value)}
              >
                {ok.map((r) => (
                  <option key={r.algorithm} value={r.algorithm}>
                    {r.label}
                    {selection?.selected?.algorithm === r.algorithm ? " (selected)" : ""}
                  </option>
                ))}
              </select>
            </div>
            <Panel>
              <MetricsRow e={current.oof} />
            </Panel>
            <EvaluationCharts e={current.oof} task={training.task} />
          </div>
        </Section>
      )}

      {result.test_result && (
        <Section
          title="Final test result"
          lead="One evaluation on the held-out rows, for the selected model only, made after the selection was frozen."
        >
          <div className="stack">
            <Callout tone="warn" title="Read this before quoting the number">
              <p>{result.test_result.limitation}</p>
            </Callout>
            <Panel>
              <div className="row" style={{ marginBottom: 12 }}>
                <Badge tone="info">Final test result</Badge>
                <span className="muted">
                  {result.test_result.algorithm} on {num(result.test_result.test_split.rows)}{" "}
                  held-out rows
                </span>
              </div>
              <MetricsRow e={result.test_result.evaluation} />
            </Panel>
            <EvaluationCharts e={result.test_result.evaluation} task={training.task} />
            {result.package_id && (
              <Panel>
                <div className="row-between">
                  <div>
                    <h3>Register this model</h3>
                    <p className="muted">
                      The sealed package <code>{result.package_id}</code> can become a version in
                      the registry.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="btn primary"
                    onClick={() => setRegistering(true)}
                  >
                    Register as a model
                  </button>
                </div>
              </Panel>
            )}
          </div>
          <RegisterDialog open={registering} jobId={jobId} onClose={() => setRegistering(false)} />
        </Section>
      )}
      {kind === "train" && (
        <Callout title="This was a cross-validation-only job">
          <p>
            It never read the held-out test rows, so there is no final test result and no model
            package. Start a final job to produce one.
          </p>
        </Callout>
      )}
    </div>
  );
}

function r0(selection: { ranking: { secondary: { metric: string } }[] }): string {
  return selection.ranking[0]?.secondary.metric ?? "secondary";
}

function RegisterDialog({
  open,
  jobId,
  onClose,
}: {
  open: boolean;
  jobId: number;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const navigate = useNavigate();
  const client = useQueryClient();
  const toast = useToast();
  const register = useMutation({
    mutationFn: (reason: string) =>
      unwrap(
        control.POST("/models", { body: { name, job_id: jobId, ...(reason ? { reason } : {}) } }),
      ),
    onSuccess: async (version) => {
      await client.invalidateQueries({ queryKey: ["models"] });
      toast(`Registered ${version.model} version ${version.version}.`);
      onClose();
      navigate(`/models/${version.model}`);
    },
  });
  const valid = /^[a-z0-9][a-z0-9_-]{0,63}$/.test(name);
  return (
    <ConfirmDialog
      open={open}
      title="Register as a model"
      confirmLabel="Register"
      busy={register.isPending}
      error={register.error ? (register.error as Error).message : null}
      onCancel={onClose}
      onConfirm={(reason) => valid && register.mutate(reason)}
    >
      <div className="field">
        <label htmlFor="model-name">Model name</label>
        <input
          id="model-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          aria-invalid={name && !valid ? true : undefined}
        />
        <span className="hint">
          Lower-case letters, digits, '-' or '_'. Using an existing name adds a new version.
        </span>
      </div>
      <p className="muted">The new version starts in development.</p>
    </ConfirmDialog>
  );
}

export default function JobDetail() {
  const { id = "" } = useParams();
  const jobId = Number(id);
  const client = useQueryClient();
  const toast = useToast();
  const query = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => unwrap(control.GET("/jobs/{job_id}", { params: { path: { job_id: jobId } } })),
    refetchInterval: (q) =>
      q.state.data && ["queued", "running"].includes(q.state.data.status) ? 1500 : false,
  });
  const [cancelling, setCancelling] = useState(false);
  const cancel = useMutation({
    mutationFn: () =>
      unwrap(control.POST("/jobs/{job_id}/cancel", { params: { path: { job_id: jobId } } })),
    onSuccess: async () => {
      toast("Cancel requested.");
      setCancelling(false);
      await client.invalidateQueries({ queryKey: ["job", jobId] });
    },
  });

  return (
    <div className="stack-lg">
      <Async query={query} label="Loading job">
        {(job) => {
          const active = job.status === "queued" || job.status === "running";
          const cancellable =
            active && ["queued", "loading", "training", "selecting"].includes(job.phase);
          const progress = (job.progress ?? null) as JobProgress | null;
          const result = (job.result ?? null) as JobResult | null;
          const phaseIndex = PHASE_ORDER.indexOf(job.phase);
          return (
            <>
              <PageHeader
                crumbs={[{ to: "/training", label: "Training" }, { label: `Job #${job.id}` }]}
                title={`Training job #${job.id}`}
                lead={
                  <>
                    {jobKindLabel(job.kind)} on{" "}
                    <Link to={`/datasets/${job.dataset.replace("@v", "/v/")}`}>{job.dataset}</Link>,
                    target <strong>{job.config.target}</strong> ({job.config.task}).
                  </>
                }
                actions={
                  <>
                    <StatusBadge status={job.status} />
                    {cancellable && (
                      <button
                        type="button"
                        className="btn danger small"
                        onClick={() => setCancelling(true)}
                      >
                        Cancel job
                      </button>
                    )}
                  </>
                }
              />

              {active && (
                <Panel>
                  <div className="stack">
                    <div className="row-between">
                      <h2>{phaseLabel(job.phase)}</h2>
                      <span className="spinner" aria-hidden="true" />
                    </div>
                    <ol
                      className="row"
                      style={{ listStyle: "none", margin: 0, padding: 0 }}
                      aria-label="Job phases"
                    >
                      {PHASE_ORDER.filter(
                        (p) =>
                          p !== "done" &&
                          (p !== "finalizing" || job.kind === "final") &&
                          (p !== "tracking" || job.track),
                      ).map((p) => (
                        <li key={p}>
                          <Badge
                            tone={
                              p === job.phase
                                ? "info"
                                : PHASE_ORDER.indexOf(p) < phaseIndex
                                  ? "ok"
                                  : "neutral"
                            }
                          >
                            {phaseLabel(p)}
                          </Badge>
                        </li>
                      ))}
                    </ol>
                    {job.phase === "training" && progress?.algorithm_count ? (
                      <div>
                        <div
                          className="progress"
                          role="progressbar"
                          aria-label="Training progress"
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={Math.round(progressFraction(progress) * 100)}
                        >
                          <div style={{ width: `${progressFraction(progress) * 100}%` }} />
                        </div>
                        <p className="muted" aria-live="polite" style={{ marginTop: 6 }}>
                          Algorithm {progress.algorithm_index} of {progress.algorithm_count}
                          {progress.algorithm ? ` (${progress.algorithm})` : ""}
                          {progress.fold ? `, fold ${progress.fold} of ${progress.folds}` : ""}
                        </p>
                      </div>
                    ) : null}
                    {job.stale && (
                      <Callout tone="warn">
                        The worker has not reported for a while. It may have stopped (is `make
                        worker` running?).
                      </Callout>
                    )}
                  </div>
                </Panel>
              )}

              {job.status === "failed" && (
                <Callout tone="bad" title="The job failed">
                  <p>
                    {(job.error as { message?: string } | null)?.message ??
                      "No error message was recorded."}
                  </p>
                </Callout>
              )}
              {job.status === "cancelled" && (
                <Callout tone="warn" title="The job was cancelled">
                  <p>No result was kept.</p>
                </Callout>
              )}
              {job.tracking_error && (
                <Callout tone="warn" title="MLflow logging did not complete">
                  <p>{String(job.tracking_error)}</p>
                </Callout>
              )}

              {result && <Results jobId={job.id} result={result} kind={job.kind} />}

              <Section title="Job details">
                <Panel>
                  <KV
                    items={[
                      ["Created", when(job.created_at)],
                      [
                        "Duration",
                        job.started_at && job.finished_at
                          ? duration(
                              (Date.parse(job.finished_at) - Date.parse(job.started_at)) / 1000,
                            )
                          : "n/a",
                      ],
                      [
                        "Algorithms",
                        job.config.algorithms.length
                          ? job.config.algorithms.map((a: { key: string }) => a.key).join(", ")
                          : "all for the task",
                      ],
                      [
                        "Settings",
                        `${job.config.settings.folds} folds, ${job.config.settings.test_fraction * 100}% held out, seed ${job.config.settings.seed}`,
                      ],
                      ["Selection metric", job.config.selection_metric ?? "task default"],
                      [
                        "MLflow tracking",
                        job.track ? <StatusBadge key="t" status={job.tracking_state} /> : "off",
                      ],
                    ]}
                  />
                </Panel>
              </Section>
              <ConfirmDialog
                open={cancelling}
                title="Cancel this job?"
                confirmLabel="Cancel job"
                danger
                busy={cancel.isPending}
                error={cancel.error instanceof ApiError ? cancel.error.message : null}
                onCancel={() => setCancelling(false)}
                onConfirm={() => cancel.mutate()}
              >
                <p>
                  A running job stops at its next fold boundary. A cancelled job keeps no result.
                </p>
              </ConfirmDialog>
            </>
          );
        }}
      </Async>
    </div>
  );
}
