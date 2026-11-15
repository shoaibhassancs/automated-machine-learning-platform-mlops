import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterAll, afterEach } from "vitest";
import { server } from "./server";

// jsdom has no <dialog> modal support, no ResizeObserver and no matchMedia: minimal stand-ins.
HTMLDialogElement.prototype.showModal = function showModal(this: HTMLDialogElement) {
  this.setAttribute("open", "");
};
HTMLDialogElement.prototype.close = function close(this: HTMLDialogElement) {
  this.removeAttribute("open");
};
class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = RO as unknown as typeof ResizeObserver;
window.HTMLElement.prototype.scrollIntoView = () => {};

// Listening at import time (not in beforeAll): openapi-fetch keeps the fetch that exists when the
// client module is created, so the mock server must patch fetch before any test file imports it.
server.listen({ onUnhandledFrame: "error" });
afterEach(() => {
  cleanup();
  server.resetHandlers();
});
afterAll(() => server.close());
