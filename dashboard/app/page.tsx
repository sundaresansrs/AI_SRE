"use client";

import { useCallback, useEffect, useState } from "react";

// All backend calls go through the server-side relay in app/api/sre, which holds the API key.
const API_BASE_URL = "/api/sre";
const EXECUTABLE_ACTIONS = new Set(["restart_deployment", "scale_deployment"]);
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
  plan: string | null;
  diagnosis: string | null;
  proposed_fix: string | null;
  verifier_reasoning: string | null;
  recommended_action: string | null;
  action_namespace: string | null;
  action_deployment_name: string | null;
  action_replicas: number | null;
  executed: boolean;
  executed_at: string | null;
  execution_result: ExecutionResult | null;
  execution_status: "running" | "succeeded" | "failed" | null;
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
  reason?: string;
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
  const [newAlert, setNewAlert] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

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

  async function handleAnalyzeAlert(e: React.FormEvent) {
    e.preventDefault();
    if (!newAlert.trim()) return;
    setIsSubmitting(true);
    setSubmitError("");
    try {
      const response = await fetch(`${API_BASE_URL}/incidents/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alert: newAlert.trim() }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`HTTP ${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`);
      }
      setNewAlert("");
      await loadQueue();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSubmitting(false);
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

  function actionSummary(row: ReviewIncident) {
    if (!row.recommended_action || row.recommended_action === "none") {
      return "No automated action (manual follow-up only)";
    }
    const target = `${row.action_namespace ?? "?"}/${row.action_deployment_name ?? "?"}`;
    return row.recommended_action === "scale_deployment"
      ? `scale_deployment ${target} to ${row.action_replicas ?? "?"} replicas`
      : `${row.recommended_action} ${target}`;
  }

  function textBlock(label: string, value: string | null, open = false) {
    if (!value) {
      return null;
    }
    return (
      <details open={open} style={{ margin: "0.5rem 0" }}>
        <summary style={{ cursor: "pointer", fontWeight: 600 }}>{label}</summary>
        <p style={{ whiteSpace: "pre-wrap", margin: "0.5rem 0 0", padding: "0.75rem", background: "#fafafa", borderRadius: 6 }}>
          {value}
        </p>
      </details>
    );
  }

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 960, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>AI-SRE incidents</h1>
      <p>Review incidents and inspect execution outcomes.</p>

      <section style={{ margin: "2rem 0", padding: "1rem", border: "1px solid #ccc", borderRadius: 8 }}>
        <h2>Submit New Alert</h2>
        <form onSubmit={handleAnalyzeAlert} style={{ display: "flex", gap: "1rem", flexDirection: "column" }}>
          <textarea 
            value={newAlert} 
            onChange={e => setNewAlert(e.target.value)}
            placeholder="E.g., Pod coredns-589f44dc88-jz8ft in kube-system is CrashLoopBackOff"
            style={{ padding: "0.5rem", borderRadius: 4, minHeight: "80px", fontFamily: "inherit" }}
            disabled={isSubmitting}
          />
          {submitError && <p style={{ color: "#b71c1c", margin: 0 }}>{submitError}</p>}
          <button 
            type="submit" 
            disabled={isSubmitting || !newAlert.trim()}
            style={{ 
              alignSelf: "flex-start", 
              padding: "0.5rem 1rem", 
              background: "#0d47a1", 
              color: "white", 
              border: "none", 
              borderRadius: 4,
              cursor: isSubmitting || !newAlert.trim() ? "not-allowed" : "pointer",
              opacity: isSubmitting || !newAlert.trim() ? 0.6 : 1
            }}
          >
            {isSubmitting ? "Analyzing..." : "Analyze Alert"}
          </button>
        </form>
      </section>

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
            const hasExecutableAction = EXECUTABLE_ACTIONS.has(row.recommended_action ?? "");
            const isEligibleForExecution = row.approval_status === "approved"
              && hasExecutableAction
              && !row.executed
              && row.execution_status !== "running";
            const canChangeApproval = !row.executed && row.execution_status !== "running";
            const canApprove = canChangeApproval && row.classification !== "REJECT" && row.approval_status !== "approved";
            const canReject = canChangeApproval && row.approval_status !== "rejected";

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
                <p>
                  <strong>Recommended action:</strong>{" "}
                  <code>{actionSummary(row)}</code>
                </p>
                {textBlock("Diagnosis", row.diagnosis, row.approval_status === "pending")}
                {textBlock("Proposed fix", row.proposed_fix, row.approval_status === "pending")}
                {textBlock("Verifier reasoning", row.verifier_reasoning)}
                {textBlock("Investigation plan", row.plan)}
                {row.execution_status === "running" ? (
                  <p><strong>Execution:</strong> in progress…</p>
                ) : null}
                {row.execution_status === "failed" && row.execution_result?.reason ? (
                  <p style={{ color: "#b71c1c" }}>
                    <strong>Execution failed:</strong> {row.execution_result.reason}
                  </p>
                ) : null}
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
                  {canApprove ? (
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
                  ) : null}

                  {canReject ? (
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
                  ) : null}

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
