import { describe, expect, it } from "vitest";
import { control, ApiError, unwrap, uploadDataset } from "@/api/client";
import { API, http, HttpResponse, server } from "./server";

describe("the typed client", () => {
  it("returns data on success", async () => {
    const health = await unwrap(control.GET("/health"));
    expect(health.status).toBe("ok");
  });

  it("turns the API's error envelope into an ApiError with its code and message", async () => {
    server.use(
      http.get(`${API}/datasets`, () =>
        HttpResponse.json(
          {
            error: {
              code: "invalid_input",
              message: "Bad value for limit",
              details: [{ loc: ["limit"] }],
            },
          },
          { status: 422 },
        ),
      ),
    );
    const error = await unwrap(control.GET("/datasets")).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 422,
      code: "invalid_input",
      message: "Bad value for limit",
    });
    expect((error as ApiError).unreachable).toBe(false);
  });

  it("says plainly when the backend cannot be reached", async () => {
    server.use(http.get(`${API}/datasets`, () => HttpResponse.error()));
    const error = (await unwrap(control.GET("/datasets")).catch((e: unknown) => e)) as ApiError;
    expect(error.unreachable).toBe(true);
    expect(error.message).toMatch(/Cannot reach the backend/);
  });

  it("falls back to a generic message for a non-envelope error", async () => {
    server.use(http.get(`${API}/datasets`, () => new HttpResponse("oops", { status: 500 })));
    const error = (await unwrap(control.GET("/datasets")).catch((e: unknown) => e)) as ApiError;
    expect(error.status).toBe(500);
    expect(error.message).toContain("500");
  });

  it("uploads a file as multipart form data and surfaces a rejection", async () => {
    let seen: { name: string | null; content: string } | null = null;
    server.use(
      http.post(`${API}/datasets`, async ({ request }) => {
        const form = await request.formData();
        seen = { name: String(form.get("name")), content: await (form.get("file") as File).text() };
        return HttpResponse.json(
          { error: { code: "csv_rejected", message: "The file has no data rows." } },
          { status: 422 },
        );
      }),
    );
    const error = (await uploadDataset(
      new File(["a,b\n"], "bad.csv", { type: "text/csv" }),
      "bad",
    ).catch((e: unknown) => e)) as ApiError;
    expect(seen).toEqual({ name: "bad", content: "a,b\n" });
    expect(error).toMatchObject({
      status: 422,
      code: "csv_rejected",
      message: "The file has no data rows.",
    });
  });
});
