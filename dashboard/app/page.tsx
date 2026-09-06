"use client";

import { useEffect, useState } from "react";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const REVIEW_QUEUE_URL = `${API_BASE_URL}/incidents/review-queue`;

type ReviewIncident = {
  id: number;
  alert: string;
  trust_score: number | null;
  classification: string | null;
  created_at: string;
  recommended_action: string | null;
  executed: boolean;
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
  const [state, setState] = useState<QueueLoadState>({ status: "loading" });
  const [actionState, setActionState] = useState<Record<number, ActionState>>({});

  async function loadQueue() {
    const response = await fetch(REVIEW_QUEUE_URL);
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
  }

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
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 960, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>AI-SRE review queue</h1>
      <p>Pending incidents awaiting review or eligible execution.</p>

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
                <p>
                  <strong>created_at:</strong> {row.created_at}
                </p>

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
