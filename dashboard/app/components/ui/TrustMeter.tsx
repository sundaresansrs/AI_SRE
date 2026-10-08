import { STRONG_THRESHOLD, WEAK_THRESHOLD, trustBand } from "../../lib/format";

/**
 * Runbook-match score as a bar with ticks at the weak (0.55) and strong (0.80) thresholds.
 * Built from spans so it can sit inside a <button> (the incident row summary).
 */
export function TrustMeter({ score, large = false }: { score: number | null; large?: boolean }) {
  if (score === null) {
    return <span className="time">No score</span>;
  }
  const percent = Math.min(100, Math.max(0, score * 100));
  return (
    <span
      className={large ? "meter meter--lg" : "meter"}
      title={`Runbook match ${score.toFixed(3)}. ACCEPT needs at least ${STRONG_THRESHOLD.toFixed(2)} plus real tool evidence.`}
    >
      <span className="meter__bar" role="img" aria-label={`Trust score ${score.toFixed(2)} out of 1`}>
        <span className={`meter__fill meter__fill--${trustBand(score)}`} style={{ width: `${percent}%` }} />
        <span className="meter__tick" style={{ left: `${WEAK_THRESHOLD * 100}%` }} />
        <span className="meter__tick" style={{ left: `${STRONG_THRESHOLD * 100}%` }} />
      </span>
      <span className="meter__value">{score.toFixed(2)}</span>
    </span>
  );
}
