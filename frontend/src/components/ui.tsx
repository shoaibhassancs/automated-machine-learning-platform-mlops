import type { ReactNode } from "react";
import { Link, NavLink } from "react-router-dom";
import { Icon } from "./icons";

export function PageHeader({
  title,
  lead,
  actions,
  crumbs,
}: {
  title: string;
  lead?: ReactNode;
  actions?: ReactNode;
  crumbs?: { to?: string; label: string }[];
}) {
  return (
    <header className="page-header">
      <div>
        {crumbs && crumbs.length > 0 && (
          <nav className="crumbs" aria-label="Breadcrumb">
            {crumbs.map((c, i) => (
              <span key={c.label}>
                {i > 0 && " / "}
                {c.to ? <Link to={c.to}>{c.label}</Link> : c.label}
              </span>
            ))}
          </nav>
        )}
        <h1 id="page-title" tabIndex={-1}>
          {title}
        </h1>
        {lead && <p className="lead">{lead}</p>}
      </div>
      {actions && <div className="row">{actions}</div>}
    </header>
  );
}

export function Section({
  title,
  lead,
  actions,
  children,
  id,
}: {
  title: string;
  lead?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  const headingId = id ? `${id}-h` : undefined;
  return (
    <section className="section" aria-labelledby={headingId} id={id}>
      <div className="row-between">
        <h2 id={headingId}>{title}</h2>
        {actions}
      </div>
      {lead && <p className="lead">{lead}</p>}
      {children}
    </section>
  );
}

export function Panel({
  children,
  flush,
  className = "",
}: {
  children: ReactNode;
  flush?: boolean;
  className?: string;
}) {
  return <div className={`panel${flush ? " flush" : ""} ${className}`}>{children}</div>;
}

export type Tone = "ok" | "warn" | "bad" | "info" | "neutral" | "sim";

const TONE_ICON: Record<Tone, ReactNode> = {
  ok: <Icon.Check />,
  warn: <Icon.Alert />,
  bad: <Icon.Cross />,
  info: <Icon.Dot />,
  neutral: <Icon.Dot />,
  sim: <Icon.Dot />,
};

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className={`badge ${tone}`}>
      {TONE_ICON[tone]}
      {children}
    </span>
  );
}

const STATUS_TONE: Record<string, Tone> = {
  succeeded: "ok",
  running: "info",
  queued: "neutral",
  failed: "bad",
  cancelled: "warn",
  canceled: "warn",
  development: "neutral",
  staging: "info",
  production: "ok",
  archived: "neutral",
  active: "ok",
  ended: "neutral",
  open: "bad",
  acknowledged: "warn",
  resolved: "ok",
  ok: "ok",
  insufficient_data: "warn",
  critical: "bad",
  warning: "warn",
  in_sync: "ok",
  out_of_sync: "bad",
  unreachable: "bad",
  logged: "ok",
  pending: "warn",
  error: "bad",
  info: "info",
};

/** Status text is always shown with an icon; colour is never the only signal. */
export function StatusBadge({ status, label }: { status: string; label?: string }) {
  return (
    <Badge tone={STATUS_TONE[status] ?? "neutral"}>{label ?? status.replace(/_/g, " ")}</Badge>
  );
}

export function SimulatedBadge() {
  return (
    <span className="badge sim" title="Replayed demo traffic, not real requests">
      Simulated
    </span>
  );
}

export function Callout({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn" | "bad" | "ok";
  title?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={`callout ${tone === "info" ? "" : tone}`}
      role={tone === "bad" ? "alert" : undefined}
    >
      <span className="icon">
        {tone === "info" || tone === "ok" ? <Icon.Info /> : <Icon.Warn />}
      </span>
      <div>
        {title && (
          <p>
            <strong>{title}</strong>
          </p>
        )}
        {children}
      </div>
    </div>
  );
}

export function Metric({
  label,
  value,
  sub,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function KV({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="kv">
      {items.map(([k, v], i) => (
        <div key={i} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function NavTabs({
  tabs,
  label,
}: {
  tabs: { to: string; label: string; end?: boolean }[];
  label: string;
}) {
  return (
    <nav className="tabs" aria-label={label}>
      {tabs.map((t) => (
        <NavLink key={t.to} to={t.to} end={t.end ?? true}>
          {t.label}
        </NavLink>
      ))}
    </nav>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function BarCell({
  value,
  max = 1,
  mark,
  hot,
  text,
}: {
  value: number | null;
  max?: number;
  mark?: number;
  hot?: boolean;
  text: string;
}) {
  const width = value === null ? 0 : Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div className="bar-cell">
      <div className="bar-track" aria-hidden="true">
        <div className={`bar-fill${hot ? " hot" : ""}`} style={{ width: `${width}%` }} />
        {mark !== undefined && (
          <div className="bar-mark" style={{ left: `${(mark / max) * 100}%` }} />
        )}
      </div>
      <span className="num nowrap">{text}</span>
    </div>
  );
}
