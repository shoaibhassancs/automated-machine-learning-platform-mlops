import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useHealth } from "@/api/queries";
import { Icon } from "./icons";
import { ToastProvider } from "./Toast";

const STAGES = [
  { to: "/datasets", n: 1, label: "Datasets", sub: [] as { to: string; label: string }[] },
  {
    to: "/training",
    n: 2,
    label: "Training",
    sub: [{ to: "/experiments", label: "Compare runs" }],
  },
  { to: "/models", n: 3, label: "Models", sub: [] },
  { to: "/deployment", n: 4, label: "Deployment", sub: [] },
  { to: "/monitoring", n: 5, label: "Monitoring", sub: [] },
];

function Environment() {
  const health = useHealth();
  if (health.isPending) return <span className="status-pill">Checking backend…</span>;
  if (health.isError) {
    return (
      <span className="status-pill bad" role="status">
        <span className="dot" /> Backend not reachable
      </span>
    );
  }
  const h = health.data;
  const throwaway = h.isolation?.isolated === true;
  return (
    <>
      <span className={`status-pill ${h.status === "ok" ? "ok" : "warn"}`} role="status">
        <span className="dot" /> {h.status === "ok" ? "Backend connected" : "Backend degraded"}
      </span>
      <span className={`status-pill ${h.worker === "running" ? "ok" : "warn"}`}>
        <span className="dot" /> Training worker {h.worker === "running" ? "running" : h.worker}
      </span>
      <span
        className="status-pill"
        title={
          throwaway
            ? "This backend uses a throwaway database and folders; nothing here touches your real data."
            : "This backend is your normal local stack. Uploads, training and deployments are saved."
        }
      >
        {throwaway ? "Throwaway demo stack" : "Local stack: actions are saved"}
      </span>
    </>
  );
}

export function Layout() {
  const [openAt, setOpenAt] = useState<string | null>(null);
  const location = useLocation();
  const open = openAt === location.pathname; // a navigation closes the menu
  const setOpen = (value: boolean) => setOpenAt(value ? location.pathname : null);

  const lastPath = useRef<string | null>(null);

  useEffect(() => {
    // After a navigation (not the first page load) move focus to the new page's title, so keyboard
    // and screen-reader users start at the top of the new content. Pages load lazily, so wait for
    // the heading to exist.
    // (React StrictMode runs effects twice on mount: only a *changed* path counts as a navigation.)
    const navigated = lastPath.current !== null && lastPath.current !== location.pathname;
    lastPath.current = location.pathname;
    let tries = 0;
    const timer = window.setInterval(() => {
      const title = document.getElementById("page-title");
      tries += 1;
      if (title) {
        document.title = `${title.textContent ? `${title.textContent} · ` : ""}AutoML Ops`;
        if (navigated) title.focus();
        window.clearInterval(timer);
      } else if (tries > 40) {
        window.clearInterval(timer);
        if (navigated) document.getElementById("main")?.focus();
      }
    }, 50);
    return () => window.clearInterval(timer);
  }, [location.pathname]);

  return (
    <ToastProvider>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <div className="app">
        <aside className={`sidebar${open ? " open" : ""}`} aria-label="Main">
          <NavLink to="/" className="brand" end>
            <span className="brand-mark">
              <Icon.Logo />
            </span>
            AutoML Ops
          </NavLink>
          <nav className="nav" aria-label="Lifecycle">
            <NavLink to="/" end>
              Overview
            </NavLink>
          </nav>
          <nav className="nav" aria-label="ML lifecycle stages">
            <div className="nav-label">ML lifecycle</div>
            {STAGES.map((s) => (
              <div key={s.to} className="nav">
                <NavLink to={s.to}>
                  <span className="step" aria-hidden="true">
                    {s.n}
                  </span>
                  {s.label}
                </NavLink>
                {s.sub.map((x) => (
                  <NavLink key={x.to} to={x.to} style={{ paddingLeft: 42 }}>
                    {x.label}
                  </NavLink>
                ))}
              </div>
            ))}
          </nav>
          <div className="sidebar-foot muted">
            A local, single-user MLOps platform for tabular CSV data.
          </div>
        </aside>
        {open && <div className="scrim" onClick={() => setOpen(false)} aria-hidden="true" />}
        <div className="main-col">
          <div className="topbar">
            <button
              type="button"
              className="btn small menu-btn"
              aria-expanded={open}
              aria-controls="main"
              onClick={() => setOpen(!open)}
            >
              <Icon.Menu /> Menu
            </button>
            <Environment />
          </div>
          <main id="main" className="main" tabIndex={-1}>
            <Outlet />
          </main>
        </div>
      </div>
    </ToastProvider>
  );
}
