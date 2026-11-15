import type { ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: ReactNode;
  cell: (row: T) => ReactNode;
  numeric?: boolean;
}

export function DataTable<T>({
  rows,
  columns,
  caption,
  rowKey,
  hideCaption,
}: {
  rows: T[];
  columns: Column<T>[];
  caption: string;
  rowKey: (row: T) => string | number;
  hideCaption?: boolean;
}) {
  return (
    // Scrollable regions must be keyboard-focusable (WCAG 2.1.1), hence tabIndex on a region.
    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
    <div className="table-wrap" tabIndex={0} role="region" aria-label={caption}>
      <table className="table">
        <caption className={hideCaption ? "sr-only" : undefined}>{caption}</caption>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col" className={c.numeric ? "num" : undefined}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={rowKey(row)}>
              {columns.map((c) => (
                <td key={c.key} className={c.numeric ? "num" : undefined}>
                  {c.cell(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
