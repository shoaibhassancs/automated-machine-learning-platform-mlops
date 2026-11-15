import { useInfiniteQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { control, unwrap } from "@/api/client";
import { useActiveDeployment } from "@/api/queries";
import { DataTable } from "@/components/DataTable";
import { Async, Empty, Loading } from "@/components/States";
import { PageHeader, Panel, Segmented, StatusBadge } from "@/components/ui";
import { num, when } from "@/lib/format";
import { ScopeBadge } from "./common";

export default function Reports() {
  const active = useActiveDeployment();
  const [scope, setScope] = useState<"all" | "real" | "simulated">("all");
  const id = active.data?.id;
  const reports = useInfiniteQuery({
    queryKey: ["reports", id, scope],
    enabled: !!id,
    initialPageParam: undefined as number | undefined,
    queryFn: ({ pageParam }) =>
      unwrap(
        control.GET("/deployments/{deployment_id}/reports", {
          params: {
            path: { deployment_id: id! },
            query: {
              limit: 25,
              ...(scope !== "all" ? { scope } : {}),
              ...(pageParam ? { before_id: pageParam } : {}),
            },
          },
        }),
      ),
    getNextPageParam: (last) => last.next_before_id ?? undefined,
  });
  const rows = reports.data?.pages.flatMap((p) => p.reports) ?? [];
  return (
    <div className="stack-lg">
      <PageHeader
        crumbs={[{ to: "/monitoring", label: "Monitoring" }, { label: "Reports" }]}
        title="Monitoring reports"
        lead="Each run of monitoring leaves one immutable report. Open one to see which features drifted and why an alert opened."
      />
      <Async query={active} label="Loading deployment">
        {(dep) =>
          !dep ? (
            <Panel>
              <Empty title="Nothing is deployed">Reports belong to a deployment.</Empty>
            </Panel>
          ) : (
            <div className="stack">
              <Segmented
                label="Traffic"
                value={scope}
                onChange={setScope}
                options={[
                  { value: "all", label: "All" },
                  { value: "real", label: "Real" },
                  { value: "simulated", label: "Simulated" },
                ]}
              />
              {reports.isPending ? (
                <Loading />
              ) : rows.length === 0 ? (
                <Panel>
                  <Empty title="No reports yet">Run monitoring from the Monitoring page.</Empty>
                </Panel>
              ) : (
                <Panel flush>
                  <DataTable
                    caption="Monitoring reports, newest first"
                    hideCaption
                    rows={rows}
                    rowKey={(r) => r.id}
                    columns={[
                      {
                        key: "i",
                        header: "Report",
                        cell: (r) => (
                          <Link className="row-link" to={`/monitoring/reports/${r.id}`}>
                            #{r.id}
                          </Link>
                        ),
                      },
                      { key: "s", header: "Traffic", cell: (r) => <ScopeBadge scope={r.scope} /> },
                      {
                        key: "st",
                        header: "Result",
                        cell: (r) => <StatusBadge status={r.status} />,
                      },
                      {
                        key: "w",
                        header: "Predictions looked at",
                        numeric: true,
                        cell: (r) => num(r.window_rows),
                      },
                      { key: "t", header: "Run at", cell: (r) => when(r.finished_at) },
                    ]}
                  />
                </Panel>
              )}
              {reports.hasNextPage && (
                <div>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => void reports.fetchNextPage()}
                    disabled={reports.isFetchingNextPage}
                  >
                    {reports.isFetchingNextPage ? "Loading…" : "Load more"}
                  </button>
                </div>
              )}
            </div>
          )
        }
      </Async>
    </div>
  );
}
