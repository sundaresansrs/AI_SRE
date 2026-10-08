from __future__ import annotations

import hmac
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from src.executor.executor import execute_recommended_action

DB_PATH = Path(os.getenv("AI_SRE_DB_PATH") or Path(__file__).resolve().with_name("ai_sre.db"))
DASHBOARD_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "DASHBOARD_ORIGINS",
        "http://localhost:3000,https://ai-sre-eight.vercel.app",
    ).split(",")
    if origin.strip()
]
logger = logging.getLogger(__name__)
EXECUTABLE_ACTIONS = {"restart_deployment", "scale_deployment"}


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard state-changing endpoints with a shared secret when AI_SRE_API_KEY is configured."""
    expected = os.getenv("AI_SRE_API_KEY", "")
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")


if not os.getenv("AI_SRE_API_KEY"):
    logger.warning("AI_SRE_API_KEY is not set: state-changing endpoints are unauthenticated")

app = FastAPI(title="AI-SRE incident API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=DASHBOARD_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class IncidentCreate(BaseModel):
    model_config = ConfigDict(extra="allow")

    alert: str
    diagnosis: str | None = None
    proposed_fix: str | None = None
    trust_score: float | None = None
    classification: str | None = None
    verifier_reasoning: str | None = None
    plan: str | None = None
    approval_status: Literal["pending", "approved", "rejected"] = "pending"
    recommended_action: Literal["restart_deployment", "scale_deployment", "none"] | None = None
    action_namespace: str | None = None
    action_deployment_name: str | None = None
    action_replicas: int | None = None


class IncidentRecord(IncidentCreate):
    id: int
    created_at: str
    approval_updated_at: str | None = None
    executed: bool = False
    executed_at: str | None = None
    execution_result: dict[str, Any] | None = None
    execution_trigger: str | None = None
    execution_status: Literal["running", "succeeded", "failed"] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert TEXT NOT NULL,
                diagnosis TEXT,
                proposed_fix TEXT,
                trust_score REAL,
                classification TEXT,
                verifier_reasoning TEXT,
                plan TEXT,
                created_at TEXT NOT NULL,
                approval_status TEXT NOT NULL DEFAULT 'pending' CHECK (approval_status IN ('pending', 'approved', 'rejected')),
                approval_updated_at TEXT
            )
            """
        )
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(incidents)").fetchall()
        }
        migrations = {
            "plan": "TEXT",
            "recommended_action": "TEXT",
            "action_namespace": "TEXT",
            "action_deployment_name": "TEXT",
            "action_replicas": "INTEGER",
            "executed": "INTEGER NOT NULL DEFAULT 0",
            "executed_at": "TEXT",
            "execution_result": "TEXT",
            "execution_trigger": "TEXT",
            "execution_status": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE incidents ADD COLUMN {column} {definition}")
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.commit()


init_db()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["executed"] = bool(result.get("executed", 0))
    if result.get("execution_result"):
        result["execution_result"] = json.loads(result["execution_result"])
    return result


def _auto_execute_enabled() -> bool:
    return os.getenv("AUTO_EXECUTE_ON_ACCEPT", "").strip().lower() in {"true", "1"}


def _action_is_eligible(state: dict[str, Any]) -> bool:
    action = state.get("recommended_action")
    if action not in {"restart_deployment", "scale_deployment"}:
        return False
    namespace = state.get("action_namespace")
    deployment_name = state.get("action_deployment_name")
    if not isinstance(namespace, str) or not namespace.strip():
        return False
    if not isinstance(deployment_name, str) or not deployment_name.strip():
        return False
    if action == "scale_deployment":
        replicas = state.get("action_replicas")
        if isinstance(replicas, bool) or not isinstance(replicas, int):
            return False
    return True


def _action_state(source: Any) -> dict[str, Any]:
    return {
        "recommended_action": source["recommended_action"],
        "action_namespace": source["action_namespace"],
        "action_deployment_name": source["action_deployment_name"],
        "action_replicas": source["action_replicas"],
    }


def _fetch_incident(conn: sqlite3.Connection, incident_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return row


def _claim_execution(incident_id: int, *, require_approval: bool) -> dict[str, Any]:
    """Atomically mark an incident as executing so concurrent requests cannot run the action twice.

    The claim is committed before the slow Kubernetes call, so the database is not locked while
    the executor waits for the rollout to verify.
    """
    with get_db_connection() as conn:
        row = _fetch_incident(conn, incident_id)
        if bool(row["executed"]):
            raise HTTPException(status_code=409, detail="incident has already been executed")
        if require_approval and row["approval_status"] != "approved":
            raise HTTPException(status_code=409, detail="incident must be approved before execution")
        if row["recommended_action"] not in EXECUTABLE_ACTIONS:
            raise HTTPException(status_code=409, detail="incident has no executable recommended action")
        claimed = conn.execute(
            """
            UPDATE incidents SET execution_status = 'running'
            WHERE id = ? AND executed = 0 AND COALESCE(execution_status, '') != 'running'
              AND (? = 0 OR approval_status = 'approved')
            """,
            (incident_id, int(require_approval)),
        ).rowcount
        if not claimed:
            raise HTTPException(status_code=409, detail="incident execution is already in progress")
        return _action_state(row)


def _persist_execution(incident_id: int, execution_result: dict[str, Any], execution_trigger: str) -> None:
    executed = bool(execution_result.get("executed"))
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE incidents
            SET executed = ?, executed_at = ?, execution_result = ?, execution_trigger = ?, execution_status = ?
            WHERE id = ?
            """,
            (
                int(executed),
                utc_now() if executed else None,
                json.dumps(execution_result, default=str),
                execution_trigger,
                "succeeded" if executed else "failed",
                incident_id,
            ),
        )


def _run_claimed_execution(incident_id: int, state: dict[str, Any], execution_trigger: str) -> dict[str, Any]:
    """Execute a claimed incident and always record the outcome, including unexpected errors."""
    try:
        execution_result = execute_recommended_action(state)
    except Exception as error:
        logger.exception("Execution failed for incident %s", incident_id)
        execution_result = {"executed": False, "error": True, "reason": f"execution error: {error}"}
    execution_result = {**execution_result, "execution_trigger": execution_trigger}
    _persist_execution(incident_id, execution_result, execution_trigger)
    return execution_result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/incidents", response_model=IncidentRecord, dependencies=[Depends(require_api_key)])
def create_incident(payload: IncidentCreate) -> dict[str, Any]:
    return persist_incident(payload.model_dump())


class AnalyzeRequest(BaseModel):
    alert: str


@app.post("/incidents/analyze", response_model=IncidentRecord, dependencies=[Depends(require_api_key)])
def analyze_incident(payload: AnalyzeRequest) -> dict[str, Any]:
    if not payload.alert.strip():
        raise HTTPException(status_code=422, detail="alert must not be empty")
    try:
        # Imported lazily: the agent graph pulls in LLM, RAG, and ML dependencies that the
        # review/approval API does not need, and runner imports persist_incident from this module.
        from src.agents.runner import run_alert_and_persist

        _, incident = run_alert_and_persist(payload.alert.strip())
        return incident
    except Exception as error:
        logger.exception("Alert analysis failed")
        raise HTTPException(status_code=500, detail=f"analysis failed: {error}") from error


def persist_incident(fields: dict[str, Any], *, trusted_graph: bool = False) -> dict[str, Any]:
    created_at = utc_now()
    if trusted_graph:
        decision_fields = fields
        approval_status = fields.get("approval_status") or "pending"
    else:
        decision_fields = {
            "trust_score": None,
            "classification": None,
            "verifier_reasoning": None,
            "recommended_action": None,
            "action_namespace": None,
            "action_deployment_name": None,
            "action_replicas": None,
        }
        approval_status = "pending"

    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO incidents (
                alert,
                diagnosis,
                proposed_fix,
                trust_score,
                classification,
                verifier_reasoning,
                plan,
                created_at,
                approval_status,
                recommended_action,
                action_namespace,
                action_deployment_name,
                action_replicas
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fields["alert"],
                fields.get("diagnosis"),
                fields.get("proposed_fix"),
                decision_fields.get("trust_score"),
                decision_fields.get("classification"),
                decision_fields.get("verifier_reasoning"),
                fields.get("plan"),
                created_at,
                approval_status,
                decision_fields.get("recommended_action"),
                decision_fields.get("action_namespace"),
                decision_fields.get("action_deployment_name"),
                decision_fields.get("action_replicas"),
            ),
        )
        incident_id = cursor.lastrowid

    # The incident is committed before any auto-execution, so an execution failure can never
    # roll back the record of what the agent diagnosed.
    state = {
        "recommended_action": decision_fields.get("recommended_action"),
        "action_namespace": decision_fields.get("action_namespace"),
        "action_deployment_name": decision_fields.get("action_deployment_name"),
        "action_replicas": decision_fields.get("action_replicas"),
    }
    if _auto_execute_enabled() and decision_fields.get("classification") == "ACCEPT" and _action_is_eligible(state):
        logger.info("Auto-execution enabled: executing incident %s with action %s", incident_id, state["recommended_action"])
        _run_claimed_execution(incident_id, _claim_execution(incident_id, require_approval=False), "auto")
    elif _auto_execute_enabled():
        logger.info("No auto-execution for incident %s: classification/action is not eligible", incident_id)

    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create incident record")
    return _row_to_dict(row)


@app.get("/incidents/review-queue")
def get_review_queue() -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM incidents
            WHERE classification IN ('REVIEW', 'ACCEPT') AND approval_status = 'pending'
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


@app.get("/incidents")
def get_incident_history(
    status: Literal["pending", "approved", "rejected"] | None = Query(default=None),
) -> list[dict[str, Any]]:
    query = "SELECT * FROM incidents"
    parameters: tuple[str, ...] = ()
    if status is not None:
        query += " WHERE approval_status = ?"
        parameters = (status,)
    query += " ORDER BY created_at DESC"

    with get_db_connection() as conn:
        rows = conn.execute(query, parameters).fetchall()
    return [_row_to_dict(row) for row in rows]


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: int) -> dict[str, Any]:
    with get_db_connection() as conn:
        row = _fetch_incident(conn, incident_id)
    return _row_to_dict(row)


@app.post("/incidents/{incident_id}/execute", dependencies=[Depends(require_api_key)])
def execute_incident(incident_id: int) -> dict[str, Any]:
    state = _claim_execution(incident_id, require_approval=True)
    execution_result = _run_claimed_execution(incident_id, state, "manual")
    if execution_result.get("error"):
        raise HTTPException(status_code=500, detail=f"incident {execution_result['reason']}")

    with get_db_connection() as conn:
        row = _fetch_incident(conn, incident_id)
    return _row_to_dict(row)


@app.post("/incidents/{incident_id}/approve", dependencies=[Depends(require_api_key)])
def approve_incident(incident_id: int) -> dict[str, Any]:
    return _set_approval_status(incident_id, "approved")


@app.post("/incidents/{incident_id}/reject", dependencies=[Depends(require_api_key)])
def reject_incident(incident_id: int) -> dict[str, Any]:
    return _set_approval_status(incident_id, "rejected")


def _set_approval_status(incident_id: int, status: Literal["approved", "rejected"]) -> dict[str, Any]:
    with get_db_connection() as conn:
        row = _fetch_incident(conn, incident_id)
        if bool(row["executed"]) or row["execution_status"] == "running":
            raise HTTPException(status_code=409, detail="approval cannot change after execution has started")
        if status == "approved" and row["classification"] == "REJECT":
            raise HTTPException(status_code=409, detail="incidents the verifier classified as REJECT cannot be approved")

        conn.execute(
            "UPDATE incidents SET approval_status = ?, approval_updated_at = ? WHERE id = ?",
            (status, utc_now(), incident_id),
        )
        row = _fetch_incident(conn, incident_id)
    return _row_to_dict(row)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
