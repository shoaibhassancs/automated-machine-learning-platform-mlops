import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// The backend allows exactly one browser origin (AUTOML_UI_ORIGIN, default http://localhost:5173),
// so the dev server must keep that port instead of silently moving to another one.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: { host: "localhost", port: 5173, strictPort: true },
  preview: { host: "localhost", port: 4173, strictPort: true },
  build: { sourcemap: false, chunkSizeWarningLimit: 700 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    css: false,
  },
});
