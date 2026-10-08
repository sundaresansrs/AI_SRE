import { needsReview } from "../lib/incidents";
import type { Incident } from "../lib/types";

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
      <span className="stat__hint">{hint}</span>
    </div>
  );
}

export function StatCards({ incidents }: { incidents: Incident[] | null }) {
  const list = incidents ?? [];
  const show = (n: number) => (incidents === null ? "-" : String(n));

  const pending = list.filter(needsReview).length;
  const approved = list.filter((i) => i.approval_status === "approved").length;
  const executed = list.filter((i) => i.executed).length;

  const scored = list.filter((i) => i.trust_score !== null);
  const average = scored.length ? scored.reduce((sum, i) => sum + (i.trust_score ?? 0), 0) / scored.length : null;

  const mix = {
    ACCEPT: list.filter((i) => i.classification === "ACCEPT").length,
    REVIEW: list.filter((i) => i.classification === "REVIEW").length,
    REJECT: list.filter((i) => i.classification === "REJECT").length,
  };
  const classified = mix.ACCEPT + mix.REVIEW + mix.REJECT;

  return (
    <section className="stats" aria-label="Summary">
      <Stat label="Needs review" value={show(pending)} hint="Waiting for a human decision" />
      <Stat label="Approved" value={show(approved)} hint="Approved by a reviewer" />
      <Stat label="Executed" value={show(executed)} hint="Applied to the live cluster" />
      <Stat
        label="Avg trust score"
        value={average === null ? "-" : average.toFixed(2)}
        hint="Runbook match, ACCEPT at 0.80+"
      />
      <div className="stat">
        <span className="stat__label">Verdict mix</span>
        <div className="mix" role="img" aria-label={`${mix.ACCEPT} accept, ${mix.REVIEW} review, ${mix.REJECT} reject`}>
          {(["ACCEPT", "REVIEW", "REJECT"] as const).map((verdict) =>
            mix[verdict] > 0 ? (
              <span
                key={verdict}
                className={`mix__seg--${verdict.toLowerCase()}`}
                style={{ width: `${(mix[verdict] / classified) * 100}%` }}
              />
            ) : null,
          )}
        </div>
        <div className="legend">
          {(["ACCEPT", "REVIEW", "REJECT"] as const).map((verdict) => (
            <span key={verdict} className="legend__item">
              <span className="legend__swatch" style={{ background: `var(--${verdict.toLowerCase()})` }} />
              {verdict.charAt(0) + verdict.slice(1).toLowerCase()} {mix[verdict]}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
