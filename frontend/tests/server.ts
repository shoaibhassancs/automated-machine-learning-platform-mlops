import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { fx } from "./fixtures";

export const API = "http://127.0.0.1:8100";
export const SERVING = "http://127.0.0.1:8200";

/** The API as recorded from a real, seeded, throwaway stack (tests/fixtures). Tests override per case. */
export const defaults = [
  http.get(`${API}/health`, () => HttpResponse.json(fx.health)),
  http.get(`${API}/datasets`, () => HttpResponse.json(fx.datasets)),
  http.get(`${API}/algorithms`, () => HttpResponse.json(fx.algorithms)),
  http.get(`${API}/jobs`, () => HttpResponse.json(fx.jobs)),
  http.get(`${API}/jobs/:id`, () => HttpResponse.json(fx.job)),
  http.get(`${API}/models`, () => HttpResponse.json(fx.models)),
  http.get(`${API}/deployments/active`, () => HttpResponse.json(fx.active)),
  http.get(`${API}/alerts`, () => HttpResponse.json(fx.alerts)),
];

export const server = setupServer(...defaults);
export { http, HttpResponse };
