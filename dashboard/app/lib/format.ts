import type { Incident } from "./types";

/** ACCEPT needs a runbook match at or above this score (plus real tool evidence). */
export const STRONG_THRESHOLD = 0.8;
/** Below this the runbook match is considered weak. */
export const WEAK_THRESHOLD = 0.55;

/** "5 min ago". `now` is passed in (0 means unknown) so rendering stays pure. */
export function timeAgo(iso: string, now: number): string {
  const time = new Date(iso).getTime();
  if (!now || Number.isNaN(time)) {
    return formatDateTime(iso);
  }
  const seconds = Math.max(0, Math.round((now - time) / 1000));
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} d ago`;
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export type TrustBand = "low" | "mid" | "high";

export function trustBand(score: number): TrustBand {
  if (score >= STRONG_THRESHOLD) return "high";
  if (score >= WEAK_THRESHOLD) return "mid";
  return "low";
}

export const ACTION_LABELS: Record<string, string> = {
  restart_deployment: "Restart deployment",
  scale_deployment: "Scale deployment",
  none: "No automated action",
};

export type ActionSummary = {
  /** Short text for the list chip, for example "scale to 1". */
  short: string;
  /** Long text for the confirmation dialog and the detail view. */
  label: string;
  target: string | null;
  executable: boolean;
};

export function summarizeAction(incident: Incident): ActionSummary {
  const action = incident.recommended_action;
  if (action !== "restart_deployment" && action !== "scale_deployment") {
    return { short: "No action", label: ACTION_LABELS.none, target: null, executable: false };
  }
  const target =
    incident.action_namespace && incident.action_deployment_name
      ? `${incident.action_namespace}/${incident.action_deployment_name}`
      : null;
  const short =
    action === "scale_deployment"
      ? `Scale to ${incident.action_replicas ?? "?"}`
      : "Restart";
  return { short, label: ACTION_LABELS[action], target, executable: target !== null };
}
