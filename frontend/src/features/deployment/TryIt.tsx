import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, postJson, SERVING_URL } from "@/api/client";
import { useActiveDeployment, useServingModel } from "@/api/queries";
import { Async, ErrorPanel } from "@/components/States";
import { Term } from "@/components/Term";
import { useToast } from "@/components/Toast";
import { Callout, KV, Metric, PageHeader, Panel, Section } from "@/components/ui";
import { num, pct } from "@/lib/format";

type PredictResponse = {
  predictions: {
    prediction_id: string;
    prediction: string | number;
    probabilities: Record<string, number> | null;
  }[];
  latency_ms: number;
  deployment: { model: string; version: number };
};
type ModelInfo = {
  input_columns: { name: string; kind: string }[];
  classes: string[] | null;
  target: string;
};

export default function TryIt() {
  const active = useActiveDeployment();
  const info = useServingModel(!!active.data);
  const client = useQueryClient();
  const toast = useToast();
  const [values, setValues] = useState<Record<string, string>>({});
  const [truth, setTruth] = useState("");
  const [labelled, setLabelled] = useState(false);

  const predict = useMutation({
    mutationFn: (columns: { name: string }[]) => {
      const row: Record<string, string | null> = {};
      for (const c of columns) row[c.name] = values[c.name]?.trim() ? values[c.name]!.trim() : null;
      return postJson<PredictResponse>(SERVING_URL, "/predict", { rows: [row] });
    },
    onSuccess: () => {
      setLabelled(false);
      setTruth("");
    },
  });
  const feedback = useMutation({
    mutationFn: (id: string) =>
      postJson(SERVING_URL, "/feedback", { prediction_id: id, true_value: truth.trim() }),
    onSuccess: async () => {
      setLabelled(true);
      toast("True value recorded. It now counts in live performance.");
      await client.invalidateQueries({ queryKey: ["performance"] });
    },
  });

  const result = predict.data?.predictions[0];
  return (
    <div className="stack-lg">
      <PageHeader
        crumbs={[{ to: "/deployment", label: "Deployment" }, { label: "Try a prediction" }]}
        title="Try a prediction"
        lead="Fill in one row and ask the live model. Leave a field empty to send it as missing. The request goes to the serving app and is logged as a real (not simulated) prediction."
      />
      <Async query={active} label="Loading deployment">
        {(dep) =>
          !dep ? (
            <Callout tone="warn" title="Nothing is deployed">
              <p>
                Deploy a model first on the <Link to="/deployment">Deployment page</Link>.
              </p>
            </Callout>
          ) : (
            <Async query={info} label="Loading the model's input fields">
              {(raw) => {
                const model = raw as unknown as ModelInfo;
                return (
                  <div className="grid-2" style={{ alignItems: "start" }}>
                    <Panel>
                      <form
                        className="stack"
                        onSubmit={(e) => {
                          e.preventDefault();
                          predict.mutate(model.input_columns);
                        }}
                      >
                        <h2>
                          {dep.model} · v{dep.version}
                        </h2>
                        <div className="form-grid">
                          {model.input_columns.map((c) => (
                            <div className="field" key={c.name}>
                              <label htmlFor={`f-${c.name}`}>
                                {c.name} <span className="muted">({c.kind})</span>
                              </label>
                              <input
                                id={`f-${c.name}`}
                                type={c.kind === "numeric" ? "number" : "text"}
                                step="any"
                                value={values[c.name] ?? ""}
                                onChange={(e) =>
                                  setValues((v) => ({ ...v, [c.name]: e.target.value }))
                                }
                              />
                            </div>
                          ))}
                        </div>
                        {predict.isError && (
                          <ErrorPanel error={predict.error} title="The prediction failed" />
                        )}
                        <div>
                          <button
                            type="submit"
                            className="btn primary"
                            disabled={predict.isPending}
                          >
                            {predict.isPending ? "Predicting…" : "Predict"}
                          </button>
                        </div>
                      </form>
                    </Panel>
                    <div className="stack" aria-live="polite">
                      {result ? (
                        <Panel>
                          <div className="stack">
                            <h2>Answer</h2>
                            <div className="metrics">
                              <Metric
                                label={`Predicted ${model.target}`}
                                value={
                                  typeof result.prediction === "number"
                                    ? num(result.prediction, 3)
                                    : String(result.prediction)
                                }
                              />
                              <Metric
                                label="Model time"
                                value={`${num(predict.data!.latency_ms, 1)} ms`}
                              />
                            </div>
                            {result.probabilities && (
                              <KV
                                items={Object.entries(result.probabilities)
                                  .sort((a, b) => b[1] - a[1])
                                  .map(([k, p]) => [k, pct(p)] as [string, string])}
                              />
                            )}
                            <hr
                              style={{
                                border: 0,
                                borderTop: "1px solid var(--border)",
                                width: "100%",
                              }}
                            />
                            <form
                              className="stack"
                              onSubmit={(e) => {
                                e.preventDefault();
                                if (truth.trim()) feedback.mutate(result.prediction_id);
                              }}
                            >
                              <h3>Later: record what really happened</h3>
                              <p className="muted">
                                Sending the true value (<Term k="feedback">feedback</Term>) lets the
                                platform measure <Term k="live performance">live performance</Term>.
                              </p>
                              <div className="field" style={{ maxWidth: 300 }}>
                                <label htmlFor="tv">True {model.target}</label>
                                <input
                                  id="tv"
                                  type="text"
                                  value={truth}
                                  onChange={(e) => setTruth(e.target.value)}
                                  disabled={labelled}
                                />
                              </div>
                              {feedback.error instanceof ApiError && (
                                <p role="alert" style={{ color: "var(--danger)", fontWeight: 600 }}>
                                  {feedback.error.message}
                                </p>
                              )}
                              <div>
                                <button
                                  type="submit"
                                  className="btn"
                                  disabled={!truth.trim() || feedback.isPending || labelled}
                                >
                                  {labelled ? "Recorded" : "Record true value"}
                                </button>
                              </div>
                            </form>
                          </div>
                        </Panel>
                      ) : (
                        <Panel>
                          <p className="muted">The answer appears here.</p>
                        </Panel>
                      )}
                      <Section title="Where this goes">
                        <p className="muted">
                          The prediction is logged. See it in{" "}
                          <Link to="/monitoring/predictions">Predictions and feedback</Link>.
                        </p>
                      </Section>
                    </div>
                  </div>
                );
              }}
            </Async>
          )
        }
      </Async>
    </div>
  );
}
