"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { summarizeAction } from "../lib/format";
import { DEFAULT_FILTERS, applyFilters, inTab, isFiltered, needsReview } from "../lib/incidents";
import type { Filters, Tab } from "../lib/incidents";
import type { Incident } from "../lib/types";
import { AlertForm } from "./AlertForm";
import { IncidentRow } from "./IncidentRow";
import type { RowAction } from "./IncidentDetail";
import { StatCards } from "./StatCards";
import { Toolbar } from "./Toolbar";
import { ConfirmDialog } from "./ui/ConfirmDialog";
import { Icon } from "./ui/Icon";
import { Toasts } from "./ui/Toasts";
import type { Toast } from "./ui/Toasts";
import { panelId, tabId } from "./ui/Tabs";

const PAGE_SIZE = 20;
const REFRESH_MS = 20000;

/** "#incident-42" in the URL opens incident 42, so a specific incident can be shared or bookmarked. */
function incidentFromHash(): number | null {
  const match = /^#incident-(\d+)$/.exec(window.location.hash);
  return match ? Number(match[1]) : null;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function ListSkeleton() {
  return (
    <div aria-busy="true" aria-label="Loading incidents">
      {[0, 1, 2, 3].map((row) => (
        <div className="skeleton" key={row}>
          <div className="skeleton__line" style={{ width: "70%" }} />
          <div className="skeleton__line" style={{ width: "40%" }} />
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const [incidents, setIncidents] = useState<Incident[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [updatedAt, setUpdatedAt] = useState("");
  const [now, setNow] = useState(0);

  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [limit, setLimit] = useState(PAGE_SIZE);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [busy, setBusy] = useState<Record<number, RowAction>>({});
  const [confirming, setConfirming] = useState<Incident | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastCounter = useRef(0);
  const deepLinkHandled = useRef(false);

  const pushToast = useCallback((tone: Toast["tone"], text: string) => {
    toastCounter.current += 1;
    const id = toastCounter.current;
    setToasts((previous) => [...previous, { id, tone, message: text }]);
  }, []);
  // The current time lives in state (and is refreshed with the data) so rendering stays pure.
  const markNow = useCallback(() => setNow(Date.now()), []);
  const dismissToast = useCallback((id: number) => setToasts((previous) => previous.filter((t) => t.id !== id)), []);

  const load = useCallback(async () => {
    try {
      const data = await api.list();
      setIncidents(data);
      setLoadError(null);
      markNow();
      setUpdatedAt(new Date().toLocaleTimeString());

      if (!deepLinkHandled.current) {
        deepLinkHandled.current = true;
        const linked = incidentFromHash();
        if (linked !== null && data.some((incident) => incident.id === linked)) {
          setFilters((current) => ({ ...current, tab: "all" }));
          setExpandedId(linked);
          setTimeout(() => document.getElementById(`incident-${linked}`)?.scrollIntoView({ block: "start" }), 80);
        }
      }
    } catch (error) {
      setLoadError(message(error));
    }
  }, [markNow]);

  useEffect(() => {
    const first = setTimeout(() => void load(), 0);
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, REFRESH_MS);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [load]);

  async function refresh() {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }

  function updateFilters(patch: Partial<Filters>) {
    setFilters((current) => ({ ...current, ...patch }));
    setLimit(PAGE_SIZE);
  }

  function replaceIncident(updated: Incident) {
    setIncidents((previous) =>
      previous ? previous.map((incident) => (incident.id === updated.id ? updated : incident)) : [updated],
    );
  }

  async function runAction(incident: Incident, action: RowAction) {
    setBusy((previous) => ({ ...previous, [incident.id]: action }));
    try {
      const updated = await api[action](incident.id);
      replaceIncident(updated);
      markNow();

      if (action === "approve") {
        // Follow the incident to its new tab so the Execute button is right where the user is looking.
        updateFilters({ tab: "approved" });
        pushToast(
          "success",
          summarizeAction(updated).executable
            ? `Incident #${updated.id} approved. Review the target, then click Execute.`
            : `Incident #${updated.id} approved.`,
        );
      } else if (action === "reject") {
        updateFilters({ tab: "rejected" });
        pushToast("info", `Incident #${updated.id} rejected.`);
      } else if (updated.execution_status === "succeeded") {
        pushToast(
          "success",
          `Incident #${updated.id} executed${updated.execution_result?.verification_status === "verified" ? " and verified" : ""}.`,
        );
      } else {
        pushToast("error", `Incident #${updated.id} was not executed: ${updated.execution_result?.reason ?? "unknown reason"}`);
      }
    } catch (error) {
      pushToast("error", `Could not ${action} incident #${incident.id}: ${message(error)}`);
      void load();
    } finally {
      setBusy((previous) => {
        const next = { ...previous };
        delete next[incident.id];
        return next;
      });
    }
  }

  async function analyze(alert: string): Promise<boolean> {
    try {
      const incident = await api.analyze(alert);
      setIncidents((previous) => [incident, ...(previous ?? []).filter((item) => item.id !== incident.id)]);
      markNow();
      setFilters({ ...DEFAULT_FILTERS, tab: needsReview(incident) ? "review" : "all" });
      setLimit(PAGE_SIZE);
      setExpandedId(incident.id);
      pushToast("success", `Analysis complete: incident #${incident.id} is ${incident.classification ?? "unscored"}.`);
      return true;
    } catch (error) {
      // The API already prefixes its detail with "analysis failed:".
      pushToast("error", `Analysis failed: ${message(error).replace(/^analysis failed:\s*/i, "")}`);
      return false;
    }
  }

  const counts = useMemo<Record<Tab, number>>(() => {
    const list = incidents ?? [];
    return {
      review: list.filter((i) => inTab(i, "review")).length,
      approved: list.filter((i) => inTab(i, "approved")).length,
      rejected: list.filter((i) => inTab(i, "rejected")).length,
      all: list.length,
    };
  }, [incidents]);

  const filtered = useMemo(() => (incidents ? applyFilters(incidents, filters) : []), [incidents, filters]);
  const visible = filtered.slice(0, limit);
  const confirmAction = confirming ? summarizeAction(confirming) : null;

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true">
            SRE
          </span>
          <div>
            <h1 className="brand__title">AI-SRE Console</h1>
            <p className="brand__sub">Incident triage with human approval</p>
          </div>
        </div>
        <div className="topbar__right">
          <span
            className={loadError ? "status status--offline" : incidents === null ? "status" : "status status--online"}
            role="status"
          >
            <span className="status__dot" aria-hidden="true" />
            {loadError ? "API unreachable" : incidents === null ? "Connecting…" : "API online"}
          </span>
        </div>
      </header>

      <main className="content">
        <StatCards incidents={incidents} />
        <AlertForm onSubmit={analyze} />

        <section className="panel" aria-label="Incidents">
          <Toolbar
            filters={filters}
            counts={counts}
            updatedAt={updatedAt}
            refreshing={refreshing}
            onChange={updateFilters}
            onRefresh={refresh}
          />

          {loadError ? (
            <div style={{ padding: 20 }}>
              <div className="banner" role="alert">
                <Icon name="alert" />
                <span className="banner__text">
                  Could not load incidents: {loadError}. Check that the API is running and AI_SRE_API_URL points to it.
                </span>
                <button type="button" className="btn btn--sm" onClick={refresh}>
                  Retry
                </button>
              </div>
            </div>
          ) : null}

          <div role="tabpanel" id={panelId("main", filters.tab)} aria-labelledby={tabId("main", filters.tab)}>
            {incidents === null && !loadError ? <ListSkeleton /> : null}

            {incidents !== null && filtered.length === 0 ? (
              <div className="empty">
                <span className="empty__title">
                  {isFiltered(filters) ? "No incidents match your filters" : "Nothing here yet"}
                </span>
                <span>
                  {isFiltered(filters)
                    ? "Try a different search or clear the filters."
                    : filters.tab === "review"
                      ? "Alerts the agent analyzes will wait here for your decision."
                      : "Incidents in this state will appear here."}
                </span>
                {isFiltered(filters) ? (
                  <button
                    type="button"
                    className="btn btn--sm"
                    onClick={() => updateFilters({ query: "", verdict: "all", action: "all" })}
                  >
                    Clear filters
                  </button>
                ) : null}
              </div>
            ) : null}

            {visible.length > 0 ? (
              <div className="list" data-testid="incident-list">
                {visible.map((incident) => (
                  <IncidentRow
                    key={incident.id}
                    incident={incident}
                    now={now}
                    expanded={expandedId === incident.id}
                    busyAction={busy[incident.id]}
                    onToggle={() => {
                      const opening = expandedId !== incident.id;
                      setExpandedId(opening ? incident.id : null);
                      window.history.replaceState(null, "", opening ? `#incident-${incident.id}` : window.location.pathname);
                    }}
                    onApprove={() => void runAction(incident, "approve")}
                    onReject={() => void runAction(incident, "reject")}
                    onExecute={() => setConfirming(incident)}
                  />
                ))}
              </div>
            ) : null}

            {filtered.length > visible.length ? (
              <div className="load-more">
                <button type="button" className="btn btn--sm" onClick={() => setLimit((current) => current + PAGE_SIZE)}>
                  Show {Math.min(PAGE_SIZE, filtered.length - visible.length)} more ({filtered.length - visible.length} remaining)
                </button>
              </div>
            ) : null}
          </div>
        </section>
      </main>

      <ConfirmDialog
        open={confirming !== null}
        title="Execute on the live cluster?"
        confirmLabel="Execute"
        onCancel={() => setConfirming(null)}
        onConfirm={() => {
          const target = confirming;
          setConfirming(null);
          if (target) void runAction(target, "execute");
        }}
      >
        {confirming && confirmAction ? (
          <>
            <dl className="kv">
              <dt>Incident</dt>
              <dd>#{confirming.id}</dd>
              <dt>Action</dt>
              <dd>{confirmAction.label}</dd>
              <dt>Target</dt>
              <dd>
                <code>{confirmAction.target}</code>
              </dd>
              {confirming.recommended_action === "scale_deployment" ? (
                <>
                  <dt>Replicas</dt>
                  <dd>{confirming.action_replicas}</dd>
                </>
              ) : null}
            </dl>
            <p className="callout">
              <Icon name="alert" />
              This changes real resources. The executor records the deployment before and after, and checks that the rollout
              succeeds. It can take up to a minute.
            </p>
          </>
        ) : null}
      </ConfirmDialog>

      <Toasts toasts={toasts} onDismiss={dismissToast} />
    </>
  );
}
