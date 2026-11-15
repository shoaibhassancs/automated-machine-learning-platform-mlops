import type { ReactNode } from "react";

/** Every chart: a title, a plain-language "what am I looking at", the chart, a text summary and the
 *  same numbers as a table (so the chart is never the only way to read the data). */
export function ChartFrame({
  title,
  help,
  summary,
  table,
  children,
}: {
  title: string;
  help?: ReactNode;
  summary: string;
  table?: ReactNode;
  children: ReactNode;
}) {
  return (
    <figure className="chart-frame">
      <figcaption>
        <div className="chart-title">{title}</div>
        {help && <div className="chart-help">{help}</div>}
      </figcaption>
      <div role="img" aria-label={`${title}. ${summary}`}>
        {children}
      </div>
      <p className="sr-only">{summary}</p>
      {table && (
        <details>
          <summary>Show the data as a table</summary>
          {table}
        </details>
      )}
    </figure>
  );
}
