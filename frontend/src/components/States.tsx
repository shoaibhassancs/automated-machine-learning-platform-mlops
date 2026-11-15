import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { ApiError } from "@/api/client";
import { Callout } from "./ui";

export function Loading({ label = "Loading", rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div role="status" aria-live="polite" aria-busy="true" className="stack">
      <span className="sr-only">{label}…</span>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton" style={{ width: `${92 - i * 14}%`, height: 16 }} />
      ))}
    </div>
  );
}

export function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function ErrorPanel({
  error,
  retry,
  title = "Something went wrong",
}: {
  error: unknown;
  retry?: () => void;
  title?: string;
}) {
  const api = error instanceof ApiError ? error : null;
  const message = api ? api.message : error instanceof Error ? error.message : "Unknown error";
  return (
    <div role="alert" className="callout bad">
      <div>
        <p>
          <strong>{api?.unreachable ? "The backend is not reachable" : title}</strong>
        </p>
        <p>{message}</p>
        {api && !api.unreachable && (
          <p className="muted small">
            Error code <code>{api.code}</code> (HTTP {api.status})
          </p>
        )}
        {retry && (
          <p style={{ marginTop: 10 }}>
            <button type="button" className="btn small" onClick={retry}>
              Try again
            </button>
          </p>
        )}
      </div>
    </div>
  );
}

/** One place that turns a query's state into a loading skeleton, an error panel, or the content. */
export function Async<T>({
  query,
  children,
  rows,
  label,
}: {
  query: UseQueryResult<T, Error>;
  children: (data: T) => ReactNode;
  rows?: number;
  label?: string;
}) {
  if (query.isPending) return <Loading rows={rows} label={label} />;
  if (query.isError) return <ErrorPanel error={query.error} retry={() => void query.refetch()} />;
  return <>{children(query.data)}</>;
}

export function NotReady({ children }: { children: ReactNode }) {
  return <Callout tone="warn">{children}</Callout>;
}
