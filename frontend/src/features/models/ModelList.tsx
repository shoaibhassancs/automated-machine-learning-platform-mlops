import { Link } from "react-router-dom";
import { useModels } from "@/api/queries";
import { DataTable } from "@/components/DataTable";
import { Async, Empty } from "@/components/States";
import { Term } from "@/components/Term";
import { PageHeader, Panel } from "@/components/ui";
import { when } from "@/lib/format";

export default function ModelList() {
  const query = useModels();
  return (
    <div className="stack-lg">
      <PageHeader
        title="Models"
        lead={
          <>
            The <Term k="registry">registry</Term> keeps every registered model version and where it
            stands. A version moves from development to staging to production, and each change is
            recorded with a reason.
          </>
        }
      />
      <Async query={query} label="Loading models">
        {(data) =>
          data.models.length === 0 ? (
            <Panel>
              <Empty
                title="No registered models yet"
                action={
                  <Link className="btn primary" to="/training">
                    Open training jobs
                  </Link>
                }
              >
                Run a final training job, then register its model package from the job's result
                page.
              </Empty>
            </Panel>
          ) : (
            <Panel flush>
              <DataTable
                caption="Registered models"
                hideCaption
                rows={data.models}
                rowKey={(m) => m.name}
                columns={[
                  {
                    key: "n",
                    header: "Model",
                    cell: (m) => (
                      <Link className="row-link" to={`/models/${m.name}`}>
                        {m.name}
                      </Link>
                    ),
                  },
                  { key: "v", header: "Versions", numeric: true, cell: (m) => m.versions },
                  {
                    key: "p",
                    header: "In production",
                    cell: (m) =>
                      m.production_version ? (
                        `v${m.production_version}`
                      ) : (
                        <span className="muted">none</span>
                      ),
                  },
                  {
                    key: "l",
                    header: "Latest",
                    cell: (m) => (m.latest_version ? `v${m.latest_version}` : "n/a"),
                  },
                  { key: "c", header: "Created", cell: (m) => when(m.created_at) },
                ]}
              />
            </Panel>
          )
        }
      </Async>
    </div>
  );
}
