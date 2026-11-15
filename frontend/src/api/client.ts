import createClient from "openapi-fetch";
import type { components, paths } from "./control";
import type { paths as ServingPaths } from "./serving";

export type { components };
export type Schemas = components["schemas"];

export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8100";
export const SERVING_URL: string = import.meta.env.VITE_SERVING_URL ?? "http://127.0.0.1:8200";

export const control = createClient<paths>({ baseUrl: API_URL });
export const serving = createClient<ServingPaths>({ baseUrl: SERVING_URL });

/** What every failed call becomes: the API's own error envelope, or "cannot reach the backend". */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;
  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
  get unreachable(): boolean {
    return this.status === 0;
  }
}

interface Envelope {
  error?: { code?: string; message?: string; details?: unknown };
}

function toApiError(status: number, body: unknown): ApiError {
  const env = (body ?? {}) as Envelope;
  const e = env.error;
  if (e && typeof e.message === "string") {
    return new ApiError(status, e.code ?? "error", e.message, e.details);
  }
  return new ApiError(status, "error", `The server answered ${status}.`);
}

/** Await an openapi-fetch call and return its data, or throw an `ApiError`. */
export async function unwrap<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  let result;
  try {
    result = await call;
  } catch {
    throw new ApiError(
      0,
      "unreachable",
      "Cannot reach the backend. Is it running? (make api and make serve)",
    );
  }
  if (result.error !== undefined || !result.response.ok) {
    throw toApiError(result.response.status, result.error);
  }
  return result.data as T;
}

/** Multipart upload (openapi-fetch does not build FormData bodies). */
export async function uploadDataset(file: File, name: string): Promise<Schemas["UploadResult"]> {
  const form = new FormData();
  form.set("name", name);
  form.set("file", file);
  let response: Response;
  try {
    response = await fetch(`${API_URL}/datasets`, { method: "POST", body: form });
  } catch {
    throw new ApiError(0, "unreachable", "Cannot reach the backend. Is it running?");
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw toApiError(response.status, body);
  return body as Schemas["UploadResult"];
}

/** Plain JSON call for the few endpoints whose generated types are awkward (kept typed by the caller). */
export async function postJson<T>(base: string, path: string, body: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${base}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "unreachable", "Cannot reach the backend. Is it running?");
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) throw toApiError(response.status, data);
  return data as T;
}
