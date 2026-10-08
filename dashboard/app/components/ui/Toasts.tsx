import { useEffect } from "react";
import { Icon } from "./Icon";

export type Toast = { id: number; tone: "success" | "error" | "info"; message: string };

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(toast.id), toast.tone === "error" ? 9000 : 5500);
    return () => clearTimeout(timer);
  }, [toast.id, toast.tone, onDismiss]);

  return (
    <div className={`toast toast--${toast.tone}`} role={toast.tone === "error" ? "alert" : "status"}>
      <span className="toast__text">{toast.message}</span>
      <button type="button" className="btn btn--ghost btn--sm btn--icon" aria-label="Dismiss" onClick={() => onDismiss(toast.id)}>
        <Icon name="x" size={14} />
      </button>
    </div>
  );
}

export function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  );
}
