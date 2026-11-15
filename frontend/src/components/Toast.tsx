import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

interface ToastItem {
  id: number;
  text: string;
  tone: "ok" | "bad";
}
const Ctx = createContext<(text: string, tone?: "ok" | "bad") => void>(() => undefined);

export function useToast() {
  return useContext(Ctx);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const push = useCallback((text: string, tone: "ok" | "bad" = "ok") => {
    const id = Date.now() + Math.random();
    setItems((cur) => [...cur, { id, text, tone }]);
    window.setTimeout(() => setItems((cur) => cur.filter((t) => t.id !== id)), 6000);
  }, []);
  const value = useMemo(() => push, [push]);
  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className={`toast ${t.tone === "bad" ? "bad" : ""}`}>
            {t.text}
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}
