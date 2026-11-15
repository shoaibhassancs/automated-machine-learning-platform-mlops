export interface Stack {
  api: string;
  serving: string;
  ui: string;
  apiPort: number;
  servingPort: number;
  uiPort: number;
  seed: "none" | "demo";
}

function stack(offset: number, seed: Stack["seed"]): Stack {
  const apiPort = 18101 + offset;
  const servingPort = 18201 + offset;
  const uiPort = 5301 + offset;
  return {
    api: `http://127.0.0.1:${apiPort}`,
    serving: `http://127.0.0.1:${servingPort}`,
    ui: `http://localhost:${uiPort}`,
    apiPort,
    servingPort,
    uiPort,
    seed,
  };
}

/** An empty stack: the whole lifecycle is built through the UI. */
export const FLOW = stack(0, "none");
/** A stack seeded with the end-to-end demo story (Abalone): model, deployment, traffic, alerts. */
export const SEEDED = stack(1, "demo");
