import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ApiError, control, serving, unwrap, type Schemas } from "./client";

export type Health = Schemas["Health"];
export type Deployment = Schemas["DeploymentDetail"];

/** Polling is paused by TanStack Query while the tab is hidden (refetchIntervalInBackground is off). */
export const keys = {
  health: ["health"] as const,
  datasets: ["datasets"] as const,
  version: (name: string, n: number) => ["version", name, n] as const,
  jobs: (filter?: object) => ["jobs", filter ?? {}] as const,
  job: (id: number) => ["job", id] as const,
  algorithms: ["algorithms"] as const,
  models: ["models"] as const,
  model: (name: string) => ["model", name] as const,
  versions: (name: string) => ["versions", name] as const,
  active: ["deployment-active"] as const,
  deployments: ["deployments"] as const,
  monitoring: (id: number) => ["monitoring", id] as const,
};

export function useHealth() {
  return useQuery({
    queryKey: keys.health,
    queryFn: () => unwrap(control.GET("/health")),
    refetchInterval: 10_000,
    retry: false,
  });
}

export function useDatasets() {
  return useQuery({
    queryKey: keys.datasets,
    queryFn: () => unwrap(control.GET("/datasets")),
  });
}

export function useJobs(
  filter: { status?: string; kind?: string; dataset?: string } = {},
  limit = 50,
) {
  return useQuery({
    queryKey: keys.jobs({ ...filter, limit }),
    queryFn: () =>
      unwrap(
        control.GET("/jobs", {
          params: {
            query: {
              limit,
              ...(filter.status ? { status: filter.status as never } : {}),
              ...(filter.kind ? { kind: filter.kind as never } : {}),
              ...(filter.dataset ? { dataset: filter.dataset } : {}),
            },
          },
        }),
      ),
    refetchInterval: (q) =>
      q.state.data?.jobs.some((j) => j.status === "queued" || j.status === "running")
        ? 2000
        : false,
  });
}

export function useModels() {
  return useQuery({
    queryKey: keys.models,
    queryFn: () => unwrap(control.GET("/models")),
  });
}

/** The active deployment, or null when nothing is deployed (the API answers 404 then). */
export function useActiveDeployment() {
  return useQuery<Deployment | null, Error>({
    queryKey: keys.active,
    queryFn: async () => {
      try {
        return await unwrap(control.GET("/deployments/active"));
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    refetchInterval: 10_000,
  });
}

export function useAlerts(
  filter: { status?: string; deployment_id?: number; scope?: string } = {},
  limit = 100,
) {
  return useQuery({
    queryKey: ["alerts", filter, limit],
    queryFn: () =>
      unwrap(
        control.GET("/alerts", {
          params: {
            query: {
              limit,
              ...(filter.status ? { status: filter.status as never } : {}),
              ...(filter.deployment_id ? { deployment_id: filter.deployment_id } : {}),
              ...(filter.scope ? { scope: filter.scope as never } : {}),
            },
          },
        }),
      ),
    refetchInterval: 15_000,
    placeholderData: keepPreviousData,
  });
}

export function useServingModel(enabled = true) {
  return useQuery({
    queryKey: ["serving-model"],
    enabled,
    retry: false,
    queryFn: () => unwrap(serving.GET("/model")),
  });
}
