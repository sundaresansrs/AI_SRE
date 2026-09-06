from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from src.executor.executor import execute_recommended_action

DB_PATH = Path(__file__).resolve().with_name("ai_sre.db")
DASHBOARD_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "DASHBOARD_ORIGINS",
        "http://localhost:3000,https://ai-sre-eight.vercel.app",
    ).split(",")
    if origin.strip()
]
logger = logging.getLogger(__name__)

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


def _persist_execution(
    conn: sqlite3.Connection,
    incident_id: int,
    execution_result: dict[str, Any],
    execution_trigger: str,
) -> None:
    executed = bool(execution_result.get("executed"))
    executed_at = utc_now() if executed else None
    conn.execute(
        "UPDATE incidents SET executed = ?, executed_at = ?, execution_result = ?, execution_trigger = ? WHERE id = ?",
        (int(executed), executed_at, json.dumps(execution_result, default=str), execution_trigger, incident_id),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/incidents", response_model=IncidentRecord)
def create_incident(payload: IncidentCreate) -> dict[str, Any]:
    return persist_incident(payload.model_dump())


def persist_incident(fields: dict[str, Any]) -> dict[str, Any]:
    created_at = utc_now()
    approval_status = fields.get("approval_status") or "pending"

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
                fields.get("trust_score"),
                fields.get("classification"),
                fields.get("verifier_reasoning"),
                fields.get("plan"),
                created_at,
                approval_status,
                fields.get("recommended_action"),
                fields.get("action_namespace"),
                fields.get("action_deployment_name"),
                fields.get("action_replicas"),
            ),
        )
        incident_id = cursor.lastrowid
        state = {
            "recommended_action": fields.get("recommended_action"),
            "action_namespace": fields.get("action_namespace"),
            "action_deployment_name": fields.get("action_deployment_name"),
            "action_replicas": fields.get("action_replicas"),
        }
        if _auto_execute_enabled() and fields.get("classification") == "ACCEPT" and _action_is_eligible(state):
            logger.info("Auto-execution enabled: executing incident %s with action %s", incident_id, fields.get("recommended_action"))
            try:
                execution_result = execute_recommended_action(state)
            except Exception as error:
                logger.exception("Auto-execution failed for incident %s", incident_id)
                raise HTTPException(status_code=500, detail=f"incident auto-execution failed: {error}") from error
            execution_result = {**execution_result, "execution_trigger": "auto"}
            _persist_execution(conn, incident_id, execution_result, "auto")
        elif _auto_execute_enabled():
            logger.info("No auto-execution for incident %s: classification/action is not eligible", incident_id)
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


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: int) -> dict[str, Any]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return _row_to_dict(row)


@app.post("/incidents/{incident_id}/execute")
def execute_incident(incident_id: int) -> dict[str, Any]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="incident not found")
        if bool(row["executed"]):
            raise HTTPException(status_code=409, detail="incident has already been executed")

        state = {
            "recommended_action": row["recommended_action"],
            "action_namespace": row["action_namespace"],
            "action_deployment_name": row["action_deployment_name"],
            "action_replicas": row["action_replicas"],
        }

        try:
            execution_result = execute_recommended_action(state)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"incident execution failed: {error}") from error

        _persist_execution(conn, incident_id, execution_result, "manual")
        updated_row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()

    if updated_row is None:
        raise HTTPException(status_code=500, detail="Failed to persist execution result")
    return _row_to_dict(updated_row)


@app.post("/incidents/{incident_id}/approve")
def approve_incident(incident_id: int) -> dict[str, Any]:
    return _set_approval_status(incident_id, "approved")


@app.post("/incidents/{incident_id}/reject")
def reject_incident(incident_id: int) -> dict[str, Any]:
    return _set_approval_status(incident_id, "rejected")


def _set_approval_status(incident_id: int, status: Literal["approved", "rejected"]) -> dict[str, Any]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="incident not found")

        updated_at = utc_now()
        conn.execute(
            "UPDATE incidents SET approval_status = ?, approval_updated_at = ? WHERE id = ?",
            (status, updated_at, incident_id),
        )
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Failed to update incident record")
    return _row_to_dict(row)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
