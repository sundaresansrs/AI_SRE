"""Entrypoints that connect the LangGraph result to incident persistence."""

from typing import Any

from src.api.main import persist_incident
from src.agents.graph import GraphState, graph


def _initial_graph_state(alert: str) -> GraphState:
    return {
        "alert": alert,
        "plan": None,
        "diagnosis": None,
        "proposed_fix": None,
        "recommended_action": None,
        "action_namespace": None,
        "action_deployment_name": None,
        "action_replicas": None,
        "trust_score": None,
        "classification": None,
        "log": [],
        "model_used": {},
        "tool_calls": [],
        "retrieved_chunks": [],
        "diagnosis_degraded": False,
    }


def _incident_fields_from_graph(state: GraphState) -> dict[str, Any]:
    return {
        "alert": state["alert"],
        "plan": state.get("plan"),
        "diagnosis": state.get("diagnosis"),
        "proposed_fix": state.get("proposed_fix"),
        "trust_score": state.get("trust_score"),
        "classification": state.get("classification"),
        "verifier_reasoning": state.get("verifier_reasoning"),
        "recommended_action": state.get("recommended_action"),
        "action_namespace": state.get("action_namespace"),
        "action_deployment_name": state.get("action_deployment_name"),
        "action_replicas": state.get("action_replicas"),
        "approval_status": "pending",
    }


def run_alert_and_persist(alert: str) -> tuple[GraphState, dict[str, Any]]:
    """Run the real graph for an alert and persist its final state as an incident."""
    final_state = graph.invoke(_initial_graph_state(alert))
    incident = persist_incident(_incident_fields_from_graph(final_state))
    return final_state, incident