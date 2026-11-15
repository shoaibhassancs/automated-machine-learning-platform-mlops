import { useInfiniteQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { control, postJson, SERVING_URL, unwrap } from "@/api/client";
import { useActiveDeployment } from "@/api/queries";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable } from "@/components/DataTable";
import { Async, Empty, Loading } from "@/components/States";
import { Term } from "@/components/Term";
import { useToast } from "@/components/Toast";
import { PageHeader, Panel } from "@/components/ui";
import { num, pct, when } from "@/lib/format";
import { RangeSwitch, rangeSince, ScopeSwitch, type Range, type Scope } from "./common";

export default function Predictions() {
  const active = useActiveDeployment();
  const client = useQueryClient();
  const toast = useToast();
  const [scope, setScope] = useState<Scope>("real");
  const [range, setRange] = useState<Range>("all");
  const [labelling, setLabelling] = useState<string | null>(null);
  const [value, setValue] = useState("");
  const id = active.data?.id;

  const list = useInfiniteQuery({
    queryKey: ["predictions", id, scope, range],
    enabled: !!id,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) =>
      unwrap(
        control.GET("/deployments/{deployment_id}/predictions", {
          params: {
            path: { deployment_id: id! },
            query: {
              scope,
              limit: 25,
              ...(rangeSince(range) ? { since: rangeSince(range)! } : {}),
              ...(pageParam ? { before: pageParam } : {}),
            },
          },
        }),
      ),
    getNextPageParam: (last) => last.next_before ?? undefined,
  });
  const rows = list.data?.pages.flatMap((p) => p.predictions) ?? [];

  const label = useMutation({
    mutationFn: () =>
      postJson(SERVING_URL, "/feedback", { prediction_id: labelling, true_value: value.trim() }),
    onSuccess: async () => {
      toast("True value recorded.");
      setLabelling(null);
      setValue("");
      await client.invalidateQueries();
    },
  });

  return (
    <div className="stack-lg">
      <PageHeader
        crumbs={[{ to: "/monitoring", label: "Monitoring" }, { label: "Predictions and feedback" }]}
        title="Predictions and feedback"
        lead={
          <>
            Every answer the live model gave. When the real outcome becomes known, record it (
            <Term k="feedback">feedback</Term>) so live performance can be measured. Input values
            are not shown here.
          </>
        }
      />
      <Async query={active} label="Loading deployment">
        {(dep) =>
          !dep ? (
            <Panel>
              <Empty title="Nothing is deployed">Predictions belong to a deployment.</Empty>
            </Panel>
          ) : (
            <div className="stack">
              <div className="row-between">
                <ScopeSwitch value={scope} onChange={setScope} />
                <RangeSwitch value={range} onChange={setRange} />
              </div>
              {list.isPending ? (
                <Loading />
              ) : rows.length === 0 ? (
                <Panel>
                  <Empty
                    title="No predictions in this view"
                    action={
                      scope === "real" ? (
                        <Link to="/deployment/try" className="btn primary">
                          Try a prediction
                        </Link>
                      ) : undefined
                    }
                  >
                    {scope === "real"
                      ? "Send a prediction yourself, or look at simulated traffic."
                      : "No simulated traffic in this range."}
                  </Empty>
                </Panel>
              ) : (
                <Panel flush>
                  <DataTable
                    caption="Predictions, newest first"
                    hideCaption
                    rows={rows}
                    rowKey={(r) => r.id}
                    columns={[
                      { key: "t", header: "When", cell: (r) => when(r.created_at) },
                      { key: "v", header: "Version", cell: (r) => `v${r.model_version}` },
                      {
                        key: "p",
                        header: "Prediction",
                        cell: (r) =>
                          typeof r.prediction === "number"
                            ? num(r.prediction, 3)
                            : String(r.prediction),
                      },
                      {
                        key: "c",
                        header: "Confidence",
                        numeric: true,
                        cell: (r) =>
                          r.confidence === null ? (
                            <span className="muted">n/a</span>
                          ) : (
                            pct(r.confidence, 0)
                          ),
                      },
                      {
                        key: "l",
                        header: "Model time",
                        numeric: true,
                        cell: (r) => `${num(r.latency_ms, 1)} ms`,
                      },
                      {
                        key: "y",
                        header: "True value",
                        cell: (r) =>
                          r.true_value !== null ? (
                            <span>{r.true_value}</span>
                          ) : (
                            <button
                              type="button"
                              className="link-btn"
                              onClick={() => setLabelling(r.id)}
                            >
                              Record true value
                            </button>
                          ),
                      },
                    ]}
                  />
                </Panel>
              )}
              {list.hasNextPage && (
                <div>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => void list.fetchNextPage()}
                    disabled={list.isFetchingNextPage}
                  >
                    {list.isFetchingNextPage ? "Loading…" : "Load more"}
                  </button>
                </div>
              )}
            </div>
          )
        }
      </Async>
      <ConfirmDialog
        open={labelling !== null}
        title="Record the true value"
        confirmLabel="Record"
        busy={label.isPending}
        error={label.error ? label.error.message : null}
        onCancel={() => {
          setLabelling(null);
          setValue("");
        }}
        onConfirm={() => value.trim() && label.mutate()}
      >
        <div className="field">
          <label htmlFor="truth">What actually happened?</label>
          <input id="truth" type="text" value={value} onChange={(e) => setValue(e.target.value)} />
          <span className="hint">
            A label is permanent: it cannot be changed afterwards. A class name for classification,
            a number for regression.
          </span>
        </div>
      </ConfirmDialog>
    </div>
  );
}
