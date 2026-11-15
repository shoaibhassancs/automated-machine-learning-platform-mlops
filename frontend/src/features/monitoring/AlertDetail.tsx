import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, control, unwrap } from "@/api/client";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Async } from "@/components/States";
import { useToast } from "@/components/Toast";
import { Callout, KV, PageHeader, Panel, Section, StatusBadge } from "@/components/ui";
import { when } from "@/lib/format";
import { RULES, ruleName, ScopeBadge } from "./common";

const ACTION_TEXT: Record<string, string> = {
  opened: "Opened",
  reoccurred: "Still firing in a later report",
  acknowledged: "Acknowledged by a person",
  resolved: "Resolved by a person",
  auto_resolved: "Resolved automatically: a later run was clear",
};

export default function AlertDetail() {
  const { id = "" } = useParams();
  const alertId = Number(id);
  const client = useQueryClient();
  const toast = useToast();
  const query = useQuery({
    queryKey: ["alert", alertId],
    queryFn: () =>
      unwrap(control.GET("/alerts/{alert_id}", { params: { path: { alert_id: alertId } } })),
  });
  const [dialog, setDialog] = useState<null | "acknowledge" | "resolve">(null);
  const act = useMutation({
    mutationFn: (reason: string) =>
      dialog === "acknowledge"
        ? unwrap(
            control.POST("/alerts/{alert_id}/acknowledge", {
              params: { path: { alert_id: alertId } },
              body: { reason },
            }),
          )
        : unwrap(
            control.POST("/alerts/{alert_id}/resolve", {
              params: { path: { alert_id: alertId } },
              body: { reason },
            }),
          ),
    onSuccess: async () => {
      toast(dialog === "acknowledge" ? "Alert acknowledged." : "Alert resolved.");
      setDialog(null);
      await client.invalidateQueries();
    },
  });

  return (
    <div className="stack-lg">
      <Async query={query} label="Loading alert">
        {(a) => (
          <>
            <PageHeader
              crumbs={[
                { to: "/monitoring", label: "Monitoring" },
                { to: "/monitoring/alerts", label: "Alerts" },
                { label: `#${a.id}` },
              ]}
              title={ruleName(a.rule)}
              lead={RULES[a.rule]?.explain}
              actions={
                <>
                  <ScopeBadge scope={a.scope} />
                  <StatusBadge status={a.severity} />
                  <StatusBadge status={a.status} />
                  {a.status === "open" && (
                    <button
                      type="button"
                      className="btn small"
                      onClick={() => setDialog("acknowledge")}
                    >
                      Acknowledge
                    </button>
                  )}
                  {a.status !== "resolved" && (
                    <button
                      type="button"
                      className="btn small"
                      onClick={() => setDialog("resolve")}
                    >
                      Resolve
                    </button>
                  )}
                </>
              }
            />
            {a.status === "resolved" && (
              <Callout tone="ok" title="This alert is resolved">
                <p>A resolved alert never changes. If the condition returns, a new alert opens.</p>
              </Callout>
            )}
            <Panel>
              <KV
                items={[
                  ["Seen in", `${a.occurrences} report${a.occurrences === 1 ? "" : "s"}`],
                  [
                    "First report",
                    <Link key="f" to={`/monitoring/reports/${a.first_report_id}`}>
                      #{a.first_report_id}
                    </Link>,
                  ],
                  [
                    "Latest report",
                    <Link key="l" to={`/monitoring/reports/${a.last_report_id}`}>
                      #{a.last_report_id}
                    </Link>,
                  ],
                  ["Opened", when(a.created_at)],
                  ["Resolved", a.resolved_at ? when(a.resolved_at) : "not yet"],
                ]}
              />
            </Panel>
            <Section title="History" lead="Every change to this alert, oldest first.">
              <Panel>
                <ol className="timeline">
                  {(a.events ?? []).map((e) => (
                    <li key={e.id}>
                      <strong>{ACTION_TEXT[e.action] ?? e.action}</strong>
                      <div className="muted">
                        {when(e.created_at)}
                        {e.report_id ? (
                          <>
                            {" "}
                            ·{" "}
                            <Link to={`/monitoring/reports/${e.report_id}`}>
                              report #{e.report_id}
                            </Link>
                          </>
                        ) : null}
                        {e.reason ? ` · ${e.reason}` : ""}
                      </div>
                    </li>
                  ))}
                </ol>
              </Panel>
            </Section>
            <ConfirmDialog
              open={dialog !== null}
              title={
                dialog === "acknowledge" ? "Acknowledge this alert?" : "Resolve this alert by hand?"
              }
              confirmLabel={dialog === "acknowledge" ? "Acknowledge" : "Resolve"}
              askReason
              busy={act.isPending}
              error={act.error instanceof ApiError ? act.error.message : null}
              onCancel={() => setDialog(null)}
              onConfirm={(r) => act.mutate(r)}
            >
              <p>
                {dialog === "acknowledge"
                  ? "It stays open, marked as seen. It still resolves itself when the condition clears."
                  : "A resolved alert is closed for good. If the condition is still true, a later report opens a new alert."}
              </p>
            </ConfirmDialog>
          </>
        )}
      </Async>
    </div>
  );
}
