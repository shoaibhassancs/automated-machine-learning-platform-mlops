import { useState } from "react";
import { Link } from "react-router-dom";
import { useAlerts } from "@/api/queries";
import { DataTable } from "@/components/DataTable";
import { Async, Empty } from "@/components/States";
import { Term } from "@/components/Term";
import { PageHeader, Panel, Segmented, StatusBadge } from "@/components/ui";
import { when } from "@/lib/format";
import { ruleName, ScopeBadge } from "./common";

export default function Alerts() {
  const [status, setStatus] = useState<"" | "open" | "acknowledged" | "resolved">("");
  const query = useAlerts(status ? { status } : {}, 100);
  return (
    <div className="stack-lg">
      <PageHeader
        crumbs={[{ to: "/monitoring", label: "Monitoring" }, { label: "Alerts" }]}
        title="Alerts"
        lead={
          <>
            An <Term k="alert">alert</Term> opens when a monitoring rule fires. It stays open while
            the condition holds and resolves itself when a later run is clear. You can also
            acknowledge or resolve one by hand, with a reason.
          </>
        }
      />
      <Segmented
        label="Alert status"
        value={status}
        onChange={setStatus}
        options={[
          { value: "", label: "All" },
          { value: "open", label: "Open" },
          { value: "acknowledged", label: "Acknowledged" },
          { value: "resolved", label: "Resolved" },
        ]}
      />
      <Async query={query} label="Loading alerts">
        {(data) =>
          data.alerts.length === 0 ? (
            <Panel>
              <Empty title="No alerts here">Nothing matches this filter.</Empty>
            </Panel>
          ) : (
            <Panel flush>
              <DataTable
                caption="Alerts, newest first"
                hideCaption
                rows={data.alerts}
                rowKey={(a) => a.id}
                columns={[
                  {
                    key: "r",
                    header: "Alert",
                    cell: (a) => (
                      <Link className="row-link" to={`/monitoring/alerts/${a.id}`}>
                        {ruleName(a.rule)}
                      </Link>
                    ),
                  },
                  { key: "t", header: "Traffic", cell: (a) => <ScopeBadge scope={a.scope} /> },
                  {
                    key: "sv",
                    header: "Severity",
                    cell: (a) => <StatusBadge status={a.severity} />,
                  },
                  { key: "st", header: "Status", cell: (a) => <StatusBadge status={a.status} /> },
                  { key: "o", header: "Seen", numeric: true, cell: (a) => `${a.occurrences}×` },
                  { key: "c", header: "First seen", cell: (a) => when(a.created_at) },
                  { key: "u", header: "Last update", cell: (a) => when(a.updated_at) },
                ]}
              />
            </Panel>
          )
        }
      </Async>
    </div>
  );
}
