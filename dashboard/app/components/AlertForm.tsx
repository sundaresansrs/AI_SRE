import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { EXAMPLE_ALERTS } from "../lib/examples";
import { Icon } from "./ui/Icon";

export function AlertForm({
  onSubmit,
}: {
  /** Resolves true when the alert was analyzed and saved. */
  onSubmit: (alert: string) => Promise<boolean>;
}) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!busy) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [busy]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const alert = text.trim();
    if (!alert || busy) return;
    setElapsed(0);
    setBusy(true);
    try {
      if (await onSubmit(alert)) {
        setText("");
        setOpen(false);
      }
    } finally {
      setBusy(false);
    }
  }

  const expanded = open || busy;

  return (
    <section className="panel" aria-labelledby="new-alert-title">
      <div className="panel__header">
        <div>
          <h2 className="panel__title" id="new-alert-title">
            Analyze a new alert
          </h2>
          <p className="panel__sub">
            The agent inspects the live cluster, searches the runbooks, and proposes a fix for you to review.
          </p>
        </div>
        <button
          type="button"
          className={expanded ? "btn" : "btn btn--primary"}
          aria-expanded={expanded}
          aria-controls="new-alert-body"
          onClick={() => setOpen((value) => !value)}
          disabled={busy}
        >
          {expanded ? "Hide" : "New alert"}
        </button>
      </div>

      {expanded ? (
        <form className="panel__body" id="new-alert-body" onSubmit={submit}>
          <div style={{ display: "grid", gap: 12 }}>
            <label className="field">
              <span className="field__label">Start from an example</span>
              <select
                className="select"
                value=""
                disabled={busy}
                onChange={(event) => event.target.value && setText(event.target.value)}
              >
                <option value="">Choose a ready-made alert…</option>
                {EXAMPLE_ALERTS.map((group) => (
                  <optgroup key={group.group} label={group.group}>
                    {group.items.map((item) => (
                      <option key={item.label} value={item.alert}>
                        {item.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>

            <label className="field">
              <span className="field__label">Alert text</span>
              <textarea
                className="textarea"
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Name the namespace and deployment, e.g. Deployment cartservice in namespace online-boutique keeps restarting."
                disabled={busy}
                maxLength={2000}
                required
              />
            </label>

            <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
              <button type="submit" className="btn btn--primary" disabled={busy || !text.trim()}>
                {busy ? <span className="spinner" aria-hidden="true" /> : <Icon name="bolt" />}
                {busy ? `Analyzing… ${elapsed}s` : "Analyze alert"}
              </button>
              <span className="panel__sub" style={{ margin: 0 }}>
                {busy
                  ? "This usually takes 1-3 minutes. You can keep using the page."
                  : "Naming the namespace and deployment in the text gives the best results."}
              </span>
            </div>
          </div>
          {busy ? <div className="progress" role="progressbar" aria-label="Analysis in progress" /> : null}
        </form>
      ) : null}
    </section>
  );
}
