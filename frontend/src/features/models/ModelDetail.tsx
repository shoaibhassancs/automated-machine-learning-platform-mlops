import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, control, unwrap } from "@/api/client";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable } from "@/components/DataTable";
import { Async } from "@/components/States";
import { Term } from "@/components/Term";
import { useToast } from "@/components/Toast";
import { Callout, KV, PageHeader, Panel, Section, StatusBadge } from "@/components/ui";
import { when } from "@/lib/format";
import { STATUS_HELP, type Status } from "./transitions";

export default function ModelDetail() {
  const { name = "" } = useParams();
  const client = useQueryClient();
  const toast = useToast();
  const model = useQuery({
    queryKey: ["model", name],
    queryFn: () => unwrap(control.GET("/models/{name}", { params: { path: { name } } })),
  });
  const versions = useQuery({
    queryKey: ["versions", name],
    queryFn: () => unwrap(control.GET("/models/{name}/versions", { params: { path: { name } } })),
  });
  const events = useQuery({
    queryKey: ["events", name],
    queryFn: () => unwrap(control.GET("/models/{name}/events", { params: { path: { name } } })),
  });
  const [rolling, setRolling] = useState(false);
  const rollback = useMutation({
    mutationFn: (reason: string) =>
      unwrap(
        control.POST("/models/{name}/rollback", { params: { path: { name } }, body: { reason } }),
      ),
    onSuccess: async () => {
      toast("Rolled back: the previous version is in production again.");
      setRolling(false);
      await client.invalidateQueries();
    },
  });

  return (
    <div className="stack-lg">
      <PageHeader
        crumbs={[{ to: "/models", label: "Models" }, { label: name }]}
        title={name}
        lead="All versions of this model and everything that happened to them, newest first."
        actions={
          model.data?.production_version ? (
            <button type="button" className="btn small" onClick={() => setRolling(true)}>
              Roll back production
            </button>
          ) : undefined
        }
      />
      <Section title="Versions">
        <Async query={versions} label="Loading versions">
          {(data) => (
            <Panel flush>
              <DataTable
                caption="Versions"
                hideCaption
                rows={[...data.versions].sort((a, b) => b.version - a.version)}
                rowKey={(v) => v.version}
                columns={[
                  {
                    key: "v",
                    header: "Version",
                    cell: (v) => (
                      <Link className="row-link" to={`/models/${name}/v/${v.version}`}>
                        v{v.version}
                      </Link>
                    ),
                  },
                  { key: "s", header: "Status", cell: (v) => <StatusBadge status={v.status} /> },
                  { key: "a", header: "Algorithm", cell: (v) => v.algorithm },
                  { key: "t", header: "Task", cell: (v) => v.task },
                  { key: "d", header: "Trained on", cell: (v) => v.dataset ?? "n/a" },
                  { key: "r", header: "Registered", cell: (v) => when(v.registered_at) },
                ]}
              />
            </Panel>
          )}
        </Async>
        <p className="muted" style={{ marginTop: 10 }}>
          <Term k="staging">Staging</Term> and <Term k="production">production</Term> are statuses
          in this registry. A version cannot skip staging, and only one version is in production at
          a time.
        </p>
      </Section>
      <Section
        title="History"
        lead="Every registration and status change, with the reason that was given."
      >
        <Async query={events} label="Loading history">
          {(data) => (
            <Panel>
              <ol className="timeline">
                {[...data.events].reverse().map((e) => (
                  <li key={e.seq}>
                    <strong>
                      v{e.version}: {e.action.replace(/_/g, " ")}
                    </strong>
                    {e.to_status && (
                      <>
                        {" "}
                        {e.from_status ? `${e.from_status} → ` : "→ "}
                        <StatusBadge status={e.to_status} />
                      </>
                    )}
                    <div className="muted">
                      {when(e.created_at)}
                      {e.reason ? ` · ${e.reason}` : ""}
                    </div>
                  </li>
                ))}
              </ol>
            </Panel>
          )}
        </Async>
      </Section>
      {model.data?.mirror && model.data.mirror.state !== "disabled" && (
        <Section title="MLflow mirror">
          <Panel>
            <KV
              items={[
                ["State", <StatusBadge key="s" status={model.data.mirror.state} />],
                ["Pending events", model.data.mirror.pending_events],
              ]}
            />
          </Panel>
        </Section>
      )}
      <Callout title="Statuses explained">
        <p>
          {(Object.keys(STATUS_HELP) as Status[]).map((s) => (
            <span key={s} style={{ display: "block" }}>
              <strong>{s}</strong>: {STATUS_HELP[s]}
            </span>
          ))}
        </p>
      </Callout>
      <ConfirmDialog
        open={rolling}
        title="Roll back production?"
        confirmLabel="Roll back"
        danger
        askReason
        busy={rollback.isPending}
        error={rollback.error instanceof ApiError ? rollback.error.message : null}
        onCancel={() => setRolling(false)}
        onConfirm={(r) => rollback.mutate(r)}
      >
        <p>
          The current production version is archived and the version it replaced becomes production
          again. Nothing is retrained. This does not change what the serving app answers until you
          deploy.
        </p>
      </ConfirmDialog>
    </div>
  );
}
