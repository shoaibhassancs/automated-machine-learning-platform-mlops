import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, control, unwrap } from "@/api/client";
import type { StoredEvaluation } from "@/api/shapes";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Async } from "@/components/States";
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
import { metricLabel, num, when } from "@/lib/format";
import { EvaluationCharts } from "../training/EvalCharts";
import { allowedActions, STATUS_HELP, type Status } from "./transitions";

export default function VersionDetail() {
  const { name = "", number: raw = "1" } = useParams();
  const number = Number(raw);
  const client = useQueryClient();
  const toast = useToast();
  const query = useQuery({
    queryKey: ["version", name, number],
    queryFn: () =>
      unwrap(
        control.GET("/models/{name}/versions/{number}", { params: { path: { name, number } } }),
      ),
  });
  const reference = useQuery({
    queryKey: ["reference", name, number],
    retry: false,
    queryFn: async () => {
      try {
        return await unwrap(
          control.GET("/models/{name}/versions/{number}/reference", {
            params: { path: { name, number } },
          }),
        );
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
  });
  const [action, setAction] = useState<null | "staging" | "production" | "archive">(null);

  const change = useMutation({
    mutationFn: (reason: string) =>
      action === "archive"
        ? unwrap(
            control.POST("/models/{name}/versions/{number}/archive", {
              params: { path: { name, number } },
              body: { reason },
            }),
          )
        : unwrap(
            control.POST("/models/{name}/versions/{number}/promote", {
              params: { path: { name, number } },
              body: { to: action as "staging" | "production", reason },
            }),
          ),
    onSuccess: async () => {
      toast(action === "archive" ? "Version archived." : `Promoted to ${action}.`);
      setAction(null);
      await client.invalidateQueries();
    },
  });
  const makeReference = useMutation({
    mutationFn: () =>
      unwrap(
        control.POST("/models/{name}/versions/{number}/reference", {
          params: { path: { name, number } },
        }),
      ),
    onSuccess: async () => {
      toast("Drift reference sample created.");
      await client.invalidateQueries({ queryKey: ["reference", name, number] });
    },
  });

  return (
    <div className="stack-lg">
      <Async query={query} label="Loading version">
        {(v) => {
          const evaluation = v.evaluation as unknown as StoredEvaluation;
          const acts = allowedActions(v.status as Status);
          return (
            <>
              <PageHeader
                crumbs={[
                  { to: "/models", label: "Models" },
                  { to: `/models/${name}`, label: name },
                  { label: `v${number}` },
                ]}
                title={`${name} · version ${number}`}
                lead={STATUS_HELP[v.status as Status]}
                actions={
                  <>
                    <StatusBadge status={v.status} />
                    {acts.promote.map((to) => (
                      <button
                        key={to}
                        type="button"
                        className="btn primary small"
                        onClick={() => setAction(to)}
                      >
                        Promote to {to}
                      </button>
                    ))}
                    {acts.archive && (
                      <button
                        type="button"
                        className="btn small"
                        onClick={() => setAction("archive")}
                      >
                        Archive
                      </button>
                    )}
                    <Link to={`/deployment?model=${name}&version=${number}`} className="btn small">
                      Deploy…
                    </Link>
                  </>
                }
              />
              <Panel>
                <KV
                  items={[
                    ["Algorithm", v.algorithm],
                    ["Task", `${v.task} (target: ${v.target})`],
                    ["Trained on", v.dataset ?? "n/a"],
                    [
                      "From job",
                      v.job_id ? (
                        <Link key="j" to={`/training/${v.job_id}`}>
                          #{v.job_id}
                        </Link>
                      ) : (
                        "n/a"
                      ),
                    ],
                    ["Registered", when(v.registered_at)],
                    ["Last status change", when(v.status_changed_at)],
                    ["Package", <code key="p">{v.package_id}</code>],
                  ]}
                />
              </Panel>
              <Section
                title="Offline test result"
                lead="Measured once on held-out rows before this version was registered."
              >
                <div className="stack">
                  <Callout tone="warn" title="Limitation">
                    <p>{evaluation.limitation}</p>
                  </Callout>
                  <Panel>
                    <div className="row" style={{ marginBottom: 12 }}>
                      <Badge tone="info">Final test result</Badge>
                      <span className="muted">
                        Selection metric: {metricLabel(evaluation.selection_metric)}
                      </span>
                    </div>
                    <div className="metrics">
                      {Object.entries(evaluation.evaluation.metrics).map(([k, val]) => (
                        <Metric key={k} label={metricLabel(k)} value={num(val, 3)} />
                      ))}
                      <Metric label="Rows scored" value={num(evaluation.evaluation.rows)} />
                    </div>
                  </Panel>
                  <EvaluationCharts e={evaluation.evaluation} task={v.task} />
                </div>
              </Section>
              <Section
                title="Drift reference"
                lead="A fixed sample of this version's training rows. Monitoring compares live traffic with it. It is created once and never replaced."
              >
                <Panel>
                  <Async query={reference} label="Loading reference" rows={1}>
                    {(r) =>
                      r ? (
                        <KV
                          items={[
                            ["Sample rows", num(r.sample_rows)],
                            ["Taken from", r.dataset_ref],
                            ["Created", when(r.created_at)],
                          ]}
                        />
                      ) : (
                        <div className="row-between">
                          <p className="muted">
                            No reference sample yet. Create one before monitoring this version.
                          </p>
                          <button
                            type="button"
                            className="btn"
                            onClick={() => makeReference.mutate()}
                            disabled={makeReference.isPending}
                          >
                            {makeReference.isPending ? "Creating…" : "Create reference sample"}
                          </button>
                        </div>
                      )
                    }
                  </Async>
                  {makeReference.error && (
                    <p role="alert" className="error" style={{ color: "var(--danger)" }}>
                      {makeReference.error.message}
                    </p>
                  )}
                </Panel>
              </Section>
              <ConfirmDialog
                open={action !== null}
                title={action === "archive" ? "Archive this version?" : `Promote to ${action}?`}
                confirmLabel={action === "archive" ? "Archive" : `Promote to ${action}`}
                danger={action === "archive"}
                askReason
                busy={change.isPending}
                error={change.error instanceof ApiError ? change.error.message : null}
                onCancel={() => setAction(null)}
                onConfirm={(r) => change.mutate(r)}
              >
                <p>
                  {action === "production"
                    ? "This version becomes the production version. The current production version, if any, is archived in the same step."
                    : action === "archive"
                      ? "An archived version is kept for the record and can be promoted back to staging."
                      : "The version moves to staging for checking. It is not live until it is deployed."}
                </p>
              </ConfirmDialog>
            </>
          );
        }}
      </Async>
    </div>
  );
}
