import { useId, useState } from "react";
import { GLOSSARY, type GlossaryKey } from "@/lib/glossary";

/** A technical word with a plain-language explanation that opens on click or keyboard (not hover only). */
export function Term({ k, children }: { k: GlossaryKey; children?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span>
      <button
        type="button"
        className="term"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
      >
        {children ?? k}
      </button>
      {open && (
        <span id={id} role="note" className="term-pop">
          {GLOSSARY[k]}
        </span>
      )}
    </span>
  );
}
