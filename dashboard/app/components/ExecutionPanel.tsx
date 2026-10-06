import type { DeploymentSnapshot, Incident } from "../lib/types";
import { Badge } from "./ui/Badge";
import type { Tone } from "./ui/Badge";

const ROWS: [keyof DeploymentSnapshot, string][] = [
  ["desired_replicas", "Desired replicas"],
  ["available_replicas", "Available replicas"],
  ["ready_replicas", "Ready replicas"],
  ["updated_replicas", "Updated replicas"],
  ["unavailable_replicas", "Unavailable replicas"],
  ["generation", "Generation"],
];

const VERIFICATION_TONE: Record<string, Tone> = { verified: "accept", timeout: "review" };

function show(value: unknown): string {
  return value === undefined || value === null ? "-" : String(value);
}

/** What the executor did: the deployment before and after, whether the rollout was verified, and the raw result. */
export function ExecutionPanel({ incident }: { incident: Incident }) {
  const result = incident.execution_result;
  if (!result) {
    return <p className="panel__sub">This incident has not been executed.</p>;
  }
  const before = result.before_state ?? null;
  const after = result.after_state ?? null;
  const verification = result.verification_status;

  return (
    <div className="detail__panel">
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        {incident.executed ? <Badge tone="accept">Executed</Badge> : <Badge tone="reject">Not executed</Badge>}
        {verification ? (
          <Badge tone={VERIFICATION_TONE[verification] ?? "neutral"}>
            {verification === "verified" ? "Rollout verified" : `Verification: ${verification}`}
          </Badge>
        ) : null}
        {incident.execution_trigger ? <Badge>{incident.execution_trigger === "auto" ? "Automatic" : "Manual"}</Badge> : null}
      </div>

      {result.reason ? (
        <div className="banner" role="alert">
          <span className="banner__text">{result.reason}</span>
        </div>
      ) : null}

      {before && after ? (
        <div className="card">
          <span className="card__title">Deployment before and after</span>
          <table className="compare">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Before</th>
                <th scope="col">After</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map(([key, label]) => {
                const changed = before[key] !== after[key];
                return (
                  <tr key={key}>
                    <td>{label}</td>
                    <td>{show(before[key])}</td>
                    <td className={changed ? "compare__changed" : undefined}>{show(after[key])}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      <details className="raw">
        <summary>Raw execution result</summary>
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </details>
    </div>
  );
}
