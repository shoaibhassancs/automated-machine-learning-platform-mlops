import { useState } from "react";
import { Link } from "react-router-dom";
import { useJobs } from "@/api/queries";
import { DataTable } from "@/components/DataTable";
import { Async, Empty } from "@/components/States";
import { PageHeader, Panel, Segmented, StatusBadge } from "@/components/ui";
import { duration, when } from "@/lib/format";
import { jobKindLabel, phaseLabel } from "./labels";

export default function JobList() {
  const [filter, setFilter] = useState<"all" | "active" | "succeeded" | "failed">("all");
  const query = useJobs({}, 100);
  return (
    <div className="stack-lg">
      <PageHeader
        title="Training"
        lead="A job cross-validates several algorithms on the training rows, then selects one by a stated rule. A final job also runs the one-time test on rows held back, and produces a model package."
        actions={
          <Link to="/training/new" className="btn primary">
            New training job
          </Link>
        }
      />
      <Async query={query} label="Loading jobs">
        {(data) => {
          const shown = data.jobs.filter((j) =>
            filter === "all"
              ? true
              : filter === "active"
                ? j.status === "queued" || j.status === "running"
                : filter === "succeeded"
                  ? j.status === "succeeded"
                  : j.status === "failed" || j.status === "cancelled",
          );
          return (
            <div className="stack">
              {data.jobs.length > 0 && (
                <Segmented
                  label="Filter jobs"
                  value={filter}
                  onChange={setFilter}
                  options={[
                    { value: "all", label: "All" },
                    { value: "active", label: "In progress" },
                    { value: "succeeded", label: "Succeeded" },
                    { value: "failed", label: "Failed or cancelled" },
                  ]}
                />
              )}
              <Panel flush>
                {shown.length === 0 ? (
                  <Empty
                    title={data.jobs.length ? "No jobs match this filter" : "No training jobs yet"}
                    action={
                      !data.jobs.length && (
                        <Link to="/training/new" className="btn primary">
                          Start your first job
                        </Link>
                      )
                    }
                  >
                    {data.jobs.length
                      ? "Choose another filter above."
                      : "Pick a dataset and the algorithms to try. Training runs in the background worker."}
                  </Empty>
                ) : (
                  <DataTable
                    caption="Training jobs, newest first"
                    hideCaption
                    rows={shown}
                    rowKey={(j) => j.id}
                    columns={[
                      {
                        key: "id",
                        header: "Job",
                        cell: (j) => (
                          <Link className="row-link" to={`/training/${j.id}`}>
                            #{j.id}
                          </Link>
                        ),
                      },
                      { key: "d", header: "Dataset", cell: (j) => j.dataset },
                      { key: "k", header: "Type", cell: (j) => jobKindLabel(j.kind) },
                      {
                        key: "s",
                        header: "Status",
                        cell: (j) => (
                          <span className="row">
                            <StatusBadge status={j.status} />
                            {(j.status === "running" || j.status === "queued") && (
                              <span className="muted">{phaseLabel(j.phase)}</span>
                            )}
                          </span>
                        ),
                      },
                      { key: "c", header: "Created", cell: (j) => when(j.created_at) },
                      {
                        key: "t",
                        header: "Duration",
                        numeric: true,
                        cell: (j) =>
                          j.started_at && j.finished_at
                            ? duration(
                                (Date.parse(j.finished_at) - Date.parse(j.started_at)) / 1000,
                              )
                            : "n/a",
                      },
                    ]}
                  />
                )}
              </Panel>
            </div>
          );
        }}
      </Async>
    </div>
  );
}
