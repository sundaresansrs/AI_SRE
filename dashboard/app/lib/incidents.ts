import type { Incident, Verdict } from "./types";

export type Tab = "review" | "approved" | "rejected" | "all";
export type VerdictFilter = "all" | Verdict | "unscored";
export type ActionFilter = "all" | "restart_deployment" | "scale_deployment" | "none";
export type SortKey = "newest" | "oldest" | "trust-desc" | "trust-asc";

export type Filters = {
  tab: Tab;
  query: string;
  verdict: VerdictFilter;
  action: ActionFilter;
  sort: SortKey;
};

export const DEFAULT_FILTERS: Filters = { tab: "review", query: "", verdict: "all", action: "all", sort: "newest" };

/** Waiting for a human decision: the verifier did not reject it and nobody has decided yet. */
export function needsReview(incident: Incident): boolean {
  return (
    incident.approval_status === "pending" &&
    (incident.classification === "REVIEW" || incident.classification === "ACCEPT")
  );
}

export function inTab(incident: Incident, tab: Tab): boolean {
  switch (tab) {
    case "review":
      return needsReview(incident);
    case "approved":
      return incident.approval_status === "approved";
    case "rejected":
      return incident.approval_status === "rejected";
    default:
      return true;
  }
}

function matchesVerdict(incident: Incident, verdict: VerdictFilter): boolean {
  if (verdict === "all") return true;
  if (verdict === "unscored") return incident.classification === null;
  return incident.classification === verdict;
}

function matchesAction(incident: Incident, action: ActionFilter): boolean {
  if (action === "all") return true;
  if (action === "none") return !incident.recommended_action || incident.recommended_action === "none";
  return incident.recommended_action === action;
}

function matchesQuery(incident: Incident, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  const haystack = [
    `#${incident.id}`,
    incident.alert,
    incident.diagnosis,
    incident.proposed_fix,
    incident.action_deployment_name,
    incident.action_namespace,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(needle);
}

export function applyFilters(incidents: Incident[], filters: Filters): Incident[] {
  const result = incidents.filter(
    (incident) =>
      inTab(incident, filters.tab) &&
      matchesVerdict(incident, filters.verdict) &&
      matchesAction(incident, filters.action) &&
      matchesQuery(incident, filters.query),
  );

  const byTime = (a: Incident, b: Incident) => b.created_at.localeCompare(a.created_at) || b.id - a.id;
  // Incidents without a score sort last in either trust order.
  const trust = (incident: Incident, missing: number) => incident.trust_score ?? missing;

  switch (filters.sort) {
    case "oldest":
      return result.sort((a, b) => -byTime(a, b));
    case "trust-desc":
      return result.sort((a, b) => trust(b, -1) - trust(a, -1) || byTime(a, b));
    case "trust-asc":
      return result.sort((a, b) => trust(a, 2) - trust(b, 2) || byTime(a, b));
    default:
      return result.sort(byTime);
  }
}

export function isFiltered(filters: Filters): boolean {
  return filters.query.trim() !== "" || filters.verdict !== "all" || filters.action !== "all";
}
