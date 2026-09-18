"use client";

import { useCallback, useEffect, useState } from "react";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const REVIEW_QUEUE_URL = `${API_BASE_URL}/incidents/review-queue`;
const INCIDENT_HISTORY_URL = `${API_BASE_URL}/incidents`;

type IncidentView = "pending" | "approved" | "rejected" | "all";

type ReviewIncident = {
  id: number;
  alert: string;
  trust_score: number | null;
  classification: string | null;
  approval_status: "pending" | "approved" | "rejected";
  created_at: string;
  recommended_action: string | null;
  executed: boolean;
  executed_at: string | null;
  execution_result: ExecutionResult | null;
};

type DeploymentSnapshot = {
  name?: string;
  desired_replicas?: number;
  available_replicas?: number;
  ready_replicas?: number;
  updated_replicas?: number;
  unavailable_replicas?: number;
  generation?: number;
  observed_generation?: number;
  error?: string;
};

type ExecutionResult = {
  before_state?: DeploymentSnapshot | null;
  after_state?: DeploymentSnapshot | null;
  verification_status?: string;
  [key: string]: unknown;
};

type QueueLoadState =
  | { status: "loading" }
  | { status: "empty" }
  | { status: "ready"; rows: ReviewIncident[] }
  | { status: "error"; message: string };

type ActionState = {
  loading: boolean;
  action: "approve" | "reject" | "execute";
  error?: string;
};

export default function ReviewQueuePage() {
  const [view, setView] = useState<IncidentView>("pending");
  const [state, setState] = useState<QueueLoadState>({ status: "loading" });
  const [actionState, setActionState] = useState<Record<number, ActionState>>({});

  const loadQueue = useCallback(async (selectedView: IncidentView = view) => {
    const endpoint = selectedView === "pending"
      ? REVIEW_QUEUE_URL
      : `${INCIDENT_HISTORY_URL}${selectedView === "all" ? "" : `?status=${selectedView}`}`;
    const response = await fetch(endpoint);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }
    const payload: unknown = await response.json();
    if (!Array.isArray(payload)) {
      throw new Error("Review queue response was not a JSON array");
    }
    if (payload.length === 0) {
      setState({ status: "empty" });
      return;
    }
    setState({ status: "ready", rows: payload as ReviewIncident[] });
  }, [view]);

  async function handleRowAction(incidentId: number, action: "approve" | "reject" | "execute") {
    const endpoint = `${API_BASE_URL}/incidents/${incidentId}/${action}`;

    setActionState((previous) => ({
      ...previous,
      [incidentId]: { loading: true, action, error: undefined },
    }));

    try {
      const response = await fetch(endpoint, {
        method: "POST",
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`HTTP ${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`);
      }

      await loadQueue();
      setActionState((previous) => {
        const next = { ...previous };
        delete next[incidentId];
        return next;
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setActionState((previous) => ({
        ...previous,
        [incidentId]: { loading: false, action, error: message },
      }));
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function loadQueueWithCancellation() {
      try {
        await loadQueue();
      } catch (error) {
        if (cancelled) {
          return;
        }
        const message = error instanceof Error ? error.message : String(error);
        setState({ status: "error", message });
      }
    }

    void loadQueueWithCancellation();
    return () => {
      cancelled = true;
    };
  }, [loadQueue]);

  function classificationBadge(classification: string | null) {
    const colors: Record<string, { background: string; color: string }> = {
      ACCEPT: { background: "#e8f5e9", color: "#1b5e20" },
      REVIEW: { background: "#fff8e1", color: "#8d6e00" },
      REJECT: { background: "#ffebee", color: "#b71c1c" },
    };
    return colors[classification ?? ""] ?? { background: "#f5f5f5", color: "#616161" };
  }

  function approvalBadge(status: ReviewIncident["approval_status"]) {
    const colors = {
      pending: { background: "#f5f5f5", color: "#616161" },
      approved: { background: "#e3f2fd", color: "#0d47a1" },
      rejected: { background: "#ffebee", color: "#b71c1c" },
    };
    return colors[status];
  }

  function deploymentSummary(snapshot: DeploymentSnapshot) {
    if (snapshot.error) {
      return snapshot.error;
    }
    return `Desired ${snapshot.desired_replicas ?? "-"}, available ${snapshot.available_replicas ?? "-"}, ready ${snapshot.ready_replicas ?? "-"}, generation ${snapshot.generation ?? "-"}`;
  }

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 960, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>AI-SRE incidents</h1>
      <p>Review incidents and inspect execution outcomes.</p>

      <nav aria-label="Incident views" style={{ display: "flex", gap: "0.5rem", margin: "1rem 0" }}>
        {(["pending", "approved", "rejected", "all"] as const).map((option) => {
          const labels: Record<IncidentView, string> = {
            pending: "Pending Review",
            approved: "Approved",
            rejected: "Rejected",
            all: "All",
          };
          return (
            <button
              key={option}
              type="button"
              onClick={() => setView(option)}
              aria-pressed={view === option}
              style={{
                padding: "0.5rem 0.8rem",
                border: "1px solid #bdbdbd",
                borderRadius: 6,
                background: view === option ? "#212121" : "white",
                color: view === option ? "white" : "#212121",
                cursor: "pointer",
              }}
            >
              {labels[option]}
            </button>
          );
        })}
      </nav>

      {state.status === "loading" ? (
        <p data-testid="queue-loading">Loading review queue…</p>
      ) : null}

      {state.status === "empty" ? (
        <section
          data-testid="empty-queue"
          style={{
            border: "1px solid #1b5e20",
            background: "#e8f5e9",
            color: "#1b5e20",
            padding: "1rem",
            borderRadius: 8,
          }}
        >
          <h2>Review queue is empty</h2>
          <p>No incidents are waiting for human review. This is a successful empty response from the API, not a load failure.</p>
        </section>
      ) : null}

      {state.status === "error" ? (
        <section
          data-testid="fetch-error"
          role="alert"
          style={{
            border: "2px solid #b71c1c",
            background: "#ffebee",
            color: "#b71c1c",
            padding: "1rem",
            borderRadius: 8,
          }}
        >
          <h2>Could not load the review queue</h2>
          <p>The dashboard could not reach the FastAPI backend. This is a fetch error, not an empty queue.</p>
          <pre data-testid="fetch-error-message" style={{ whiteSpace: "pre-wrap" }}>
            {state.message}
          </pre>
        </section>
      ) : null}

      {state.status === "ready" ? (
        <ul data-testid="review-queue" style={{ listStyle: "none", padding: 0 }}>
          {state.rows.map((row) => {
            const rowAction = actionState[row.id];
            const isLoading = rowAction?.loading ?? false;
            const actionError = rowAction?.error;
            const isEligibleForExecution = row.classification === "ACCEPT" && !row.executed;

            return (
              <li
                key={row.id}
                data-testid={`incident-${row.id}`}
                style={{ border: "1px solid #ccc", borderRadius: 8, padding: "1rem", marginBottom: "1rem" }}
              >
                <p>
                  <strong>id:</strong> {row.id}
                </p>
                <p>
                  <strong>alert:</strong> {row.alert}
                </p>
                <p>
                  <strong>trust_score:</strong> {row.trust_score === null ? "null" : String(row.trust_score)}
                </p>
                <p>
                  <strong>classification:</strong> {row.classification}
                </p>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", margin: "0.75rem 0" }}>
                  <span style={{ ...classificationBadge(row.classification), padding: "0.25rem 0.5rem", borderRadius: 999, fontSize: "0.85rem", fontWeight: 600 }}>
                    Classification: {row.classification ?? "Unknown"}
                  </span>
                  <span style={{ ...approvalBadge(row.approval_status), padding: "0.25rem 0.5rem", borderRadius: 999, fontSize: "0.85rem", fontWeight: 600 }}>
                    Approval: {row.approval_status}
                  </span>
                  {row.executed ? (
                    <span style={{ background: "#e0f2f1", color: "#00695c", padding: "0.25rem 0.5rem", borderRadius: 999, fontSize: "0.85rem", fontWeight: 600 }}>
                      Executed{row.executed_at ? `: ${row.executed_at}` : ""}
                    </span>
                  ) : null}
                </div>
                <p>
                  <strong>created_at:</strong> {row.created_at}
                </p>
                {row.execution_result ? (
                  <>
                    {row.execution_result.before_state && row.execution_result.after_state ? (
                      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", margin: "0.75rem 0" }}>
                        <div style={{ flex: "1 1 260px", background: "#f5f5f5", padding: "0.75rem", borderRadius: 6 }}>
                          <strong>Before</strong>
                          <p style={{ marginBottom: 0 }}>{deploymentSummary(row.execution_result.before_state)}</p>
                        </div>
                        <div style={{ flex: "1 1 260px", background: "#e8f5e9", padding: "0.75rem", borderRadius: 6 }}>
                          <strong>After</strong>
                          <p style={{ marginBottom: 0 }}>{deploymentSummary(row.execution_result.after_state)}</p>
                        </div>
                      </div>
                    ) : null}
                    {row.execution_result.verification_status ? (
                      <p>
                        <strong>Verification:</strong> {row.execution_result.verification_status}
                      </p>
                    ) : null}
                  </>
                ) : null}

                <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem", alignItems: "center" }}>
                  <button
                    type="button"
                    onClick={() => void handleRowAction(row.id, "approve")}
                    disabled={isLoading}
                    data-testid={`approve-${row.id}`}
                    style={{
                      padding: "0.5rem 0.9rem",
                      background: isLoading && rowAction?.action === "approve" ? "#a5d6a7" : "#2e7d32",
                      color: "white",
                      border: "none",
                      borderRadius: 6,
                      cursor: isLoading ? "not-allowed" : "pointer",
                    }}
                  >
                    {isLoading && rowAction?.action === "approve" ? "Approving…" : "Approve"}
                  </button>

                  <button
                    type="button"
                    onClick={() => void handleRowAction(row.id, "reject")}
                    disabled={isLoading}
                    data-testid={`reject-${row.id}`}
                    style={{
                      padding: "0.5rem 0.9rem",
                      background: isLoading && rowAction?.action === "reject" ? "#ef9a9a" : "#c62828",
                      color: "white",
                      border: "none",
                      borderRadius: 6,
                      cursor: isLoading ? "not-allowed" : "pointer",
                    }}
                  >
                    {isLoading && rowAction?.action === "reject" ? "Rejecting…" : "Reject"}
                  </button>

                  {isEligibleForExecution ? (
                    <button
                      type="button"
                      onClick={() => void handleRowAction(row.id, "execute")}
                      disabled={isLoading}
                      data-testid={`execute-${row.id}`}
                      style={{
                        padding: "0.5rem 0.9rem",
                        background: isLoading && rowAction?.action === "execute" ? "#90a4ae" : "#1565c0",
                        color: "white",
                        border: "none",
                        borderRadius: 6,
                        cursor: isLoading ? "not-allowed" : "pointer",
                      }}
                    >
                      {isLoading && rowAction?.action === "execute" ? "Executing…" : "Execute"}
                    </button>
                  ) : row.executed ? (
                    <button type="button" disabled data-testid={`execute-${row.id}`}>
                      Executed
                    </button>
                  ) : null}
                </div>

                {actionError ? (
                  <p
                    data-testid={`action-error-${row.id}`}
                    role="alert"
                    style={{
                      marginTop: "0.75rem",
                      color: "#b71c1c",
                      background: "#ffebee",
                      border: "1px solid #ef9a9a",
                      borderRadius: 6,
                      padding: "0.5rem 0.75rem",
                    }}
                  >
                    {actionError}
                  </p>
                ) : null}
              </li>
            );
          })}
        </ul>
      ) : null}
    </main>
  );
}
