import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ApiError, control, unwrap } from "@/api/client";
import { keys, useActiveDeployment, useModels } from "@/api/queries";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable } from "@/components/DataTable";
import { Async, ErrorPanel } from "@/components/States";
import { Term } from "@/components/Term";
import { useToast } from "@/components/Toast";
import { Callout, KV, PageHeader, Panel, Section, StatusBadge } from "@/components/ui";
import { when } from "@/lib/format";

function VersionPicker({
  model,
  version,
  onChange,
}: {
  model: string;
  version: string;
  onChange: (model: string, version: string) => void;
}) {
  const models = useModels();
  const versions = useQuery({
    queryKey: ["versions", model],
    enabled: !!model,
    queryFn: () =>
      unwrap(control.GET("/models/{name}/versions", { params: { path: { name: model } } })),
  });
  return (
    <div className="form-grid">
      <div className="field">
        <label htmlFor="dm">Model</label>
        <select id="dm" value={model} onChange={(e) => onChange(e.target.value, "")}>
          <option value="">Choose a model…</option>
          {models.data?.models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor="dv">Version</label>
        <select
          id="dv"
          value={version}
          onChange={(e) => onChange(model, e.target.value)}
          disabled={!model}
        >
          <option value="">Choose a version…</option>
          {versions.data?.versions
            .filter((v) => v.status !== "archived")
            .map((v) => (
              <option key={v.version} value={String(v.version)}>
                v{v.version} ({v.status}, {v.algorithm})
              </option>
            ))}
        </select>
      </div>
    </div>
  );
}

export default function Deployment() {
  const [search] = useSearchParams();
  const active = useActiveDeployment();
  const client = useQueryClient();
  const toast = useToast();
  const history = useQuery({
    queryKey: keys.deployments,
    queryFn: () => unwrap(control.GET("/deployments", { params: { query: { limit: 50 } } })),
  });
  const [model, setModel] = useState(search.get("model") ?? "");
  const [version, setVersion] = useState(search.get("version") ?? "");
  const [dialog, setDialog] = useState<null | "deploy" | "switch" | "rollback" | "undeploy">(null);

  const run = useMutation({
    mutationFn: async (reason: string) => {
      const body = { model, version: Number(version), reason };
      const id = active.data?.id;
      if (dialog === "deploy") return unwrap(control.POST("/deployments", { body }));
      if (dialog === "switch" && id)
        return unwrap(
          control.POST("/deployments/{deployment_id}/switch", {
            params: { path: { deployment_id: id } },
            body,
          }),
        );
      if (dialog === "rollback")
        return unwrap(control.POST("/deployments/rollback", { body: { reason } }));
      if (dialog === "undeploy" && id)
        return unwrap(
          control.POST("/deployments/{deployment_id}/undeploy", {
            params: { path: { deployment_id: id } },
            body: { reason },
          }),
        );
      throw new Error("Nothing to do");
    },
    onSuccess: async () => {
      toast(
        dialog === "undeploy"
          ? "Stopped serving: no model is live now."
          : dialog === "rollback"
            ? "Rolled back to the previous deployment."
            : "Deployment updated.",
      );
      setDialog(null);
      await client.invalidateQueries();
    },
  });

  const picked = !!model && !!version;
  const d = active.data;
  const needsPick = dialog === "deploy" || dialog === "switch";

  return (
    <div className="stack-lg">
      <PageHeader
        title="Deployment"
        lead={
          <>
            <Term k="deployment">Deploying</Term> makes one registered version the model the{" "}
            <Term k="serving app">serving app</Term> answers with. You can switch versions or roll
            back without retraining.
          </>
        }
        actions={
          d ? (
            <Link to="/deployment/try" className="btn primary">
              Try a prediction
            </Link>
          ) : undefined
        }
      />
      <Async query={active} label="Loading deployment">
        {(dep) =>
          dep ? (
            <Panel>
              <div className="stack">
                <div className="row-between">
                  <h2>
                    Live now: {dep.model} · v{dep.version}
                  </h2>
                  <span className="row">
                    <StatusBadge status={dep.registry_status} />
                    {dep.serving && (
                      <StatusBadge
                        status={dep.serving.state}
                        label={`serving ${dep.serving.state.replace(/_/g, " ")}`}
                      />
                    )}
                  </span>
                </div>
                {dep.serving && dep.serving.state !== "in_sync" && (
                  <Callout tone="warn" title="The serving app is not showing this deployment yet">
                    <p>
                      {dep.serving.error ??
                        "It re-reads the active deployment every few seconds. If this stays, check that make serve is running."}
                    </p>
                  </Callout>
                )}
                <KV
                  items={[
                    ["Algorithm", dep.algorithm],
                    ["Task", dep.task],
                    ["Deployed", when(dep.deployed_at)],
                    ["Why", dep.reason],
                  ]}
                />
                <div className="row">
                  <button type="button" className="btn" onClick={() => setDialog("switch")}>
                    Switch to another version
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => setDialog("rollback")}
                    disabled={!dep.replaced_id}
                  >
                    Roll back
                  </button>
                  <button
                    type="button"
                    className="btn danger"
                    onClick={() => setDialog("undeploy")}
                  >
                    Stop serving
                  </button>
                </div>
              </div>
            </Panel>
          ) : (
            <Panel>
              <div className="stack">
                <h2>Nothing is deployed</h2>
                <p className="muted">
                  Choose a registered version to serve. Staging or production versions are the usual
                  choice.
                </p>
                <VersionPicker
                  model={model}
                  version={version}
                  onChange={(m, v) => {
                    setModel(m);
                    setVersion(v);
                  }}
                />
                <div>
                  <button
                    type="button"
                    className="btn primary"
                    disabled={!picked}
                    onClick={() => setDialog("deploy")}
                  >
                    Deploy this version
                  </button>
                </div>
              </div>
            </Panel>
          )
        }
      </Async>
      <Section title="Deployment history" lead="Every deploy, switch and rollback, newest first.">
        <Async query={history} label="Loading history">
          {(h) =>
            h.deployments.length === 0 ? (
              <p className="muted">No deployments yet.</p>
            ) : (
              <Panel flush>
                <DataTable
                  caption="Deployment history"
                  hideCaption
                  rows={h.deployments}
                  rowKey={(x) => x.id}
                  columns={[
                    { key: "m", header: "Model", cell: (x) => `${x.model} v${x.version}` },
                    { key: "a", header: "Action", cell: (x) => x.action },
                    { key: "s", header: "Status", cell: (x) => <StatusBadge status={x.status} /> },
                    { key: "t", header: "When", cell: (x) => when(x.deployed_at) },
                    { key: "r", header: "Reason", cell: (x) => x.reason },
                  ]}
                />
              </Panel>
            )
          }
        </Async>
      </Section>
      <ConfirmDialog
        open={dialog !== null}
        title={
          dialog === "deploy"
            ? "Deploy this version?"
            : dialog === "switch"
              ? "Switch the live model?"
              : dialog === "rollback"
                ? "Roll back the deployment?"
                : "Stop serving?"
        }
        confirmLabel={
          dialog === "undeploy" ? "Stop serving" : dialog === "rollback" ? "Roll back" : "Confirm"
        }
        danger={dialog === "undeploy"}
        askReason
        busy={run.isPending}
        error={run.error instanceof ApiError ? run.error.message : null}
        onCancel={() => setDialog(null)}
        onConfirm={(r) => (needsPick && !picked ? undefined : run.mutate(r))}
      >
        {needsPick && (
          <VersionPicker
            model={model}
            version={version}
            onChange={(m, v) => {
              setModel(m);
              setVersion(v);
            }}
          />
        )}
        <p>
          {dialog === "undeploy"
            ? "Prediction requests will be refused until you deploy something again."
            : dialog === "rollback"
              ? "The previous deployment becomes live again."
              : "Requests are answered with the chosen version from the next few seconds. Earlier predictions stay in the log."}
        </p>
      </ConfirmDialog>
      {active.isError && <ErrorPanel error={active.error} />}
    </div>
  );
}
