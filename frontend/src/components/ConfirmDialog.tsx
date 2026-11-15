import { useEffect, useId, useRef, useState, type ReactNode } from "react";

/** A modal confirmation built on the native <dialog>: focus is trapped, Escape closes, and the
 *  consequence is spelled out. With `askReason` a reason (1-500 characters) is required. */
export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel,
  danger,
  askReason,
  busy,
  error,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  children: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  askReason?: boolean;
  busy?: boolean;
  error?: string | null;
  onConfirm: (reason: string) => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [reason, setReason] = useState("");
  const titleId = useId();
  const reasonId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      setReason("");
      dialog.showModal();
    }
    if (!open && dialog.open) dialog.close();
  }, [open]);

  const reasonOk = !askReason || (reason.trim().length >= 1 && reason.length <= 500);
  return (
    <dialog
      ref={ref}
      className="dialog"
      aria-labelledby={titleId}
      onCancel={(e) => {
        e.preventDefault();
        onCancel();
      }}
    >
      <form
        method="dialog"
        onSubmit={(e) => {
          e.preventDefault();
          if (reasonOk && !busy) onConfirm(reason.trim());
        }}
      >
        <h2 id={titleId}>{title}</h2>
        <div className="stack">{children}</div>
        {askReason && (
          <div className="field">
            <label htmlFor={reasonId}>Reason (recorded in the history)</label>
            <input
              id={reasonId}
              type="text"
              value={reason}
              maxLength={500}
              onChange={(e) => setReason(e.target.value)}
              aria-required="true"
            />
          </div>
        )}
        {error && (
          <p role="alert" className="error" style={{ color: "var(--danger)", fontWeight: 600 }}>
            {error}
          </p>
        )}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="submit"
            className={`btn ${danger ? "danger" : "primary"}`}
            disabled={!reasonOk || busy}
          >
            {busy ? "Working…" : confirmLabel}
          </button>
        </div>
      </form>
    </dialog>
  );
}
