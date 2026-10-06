import pytest
from langgraph.graph import END, START, StateGraph

import src.api.main as api_main
from src.agents.runner import _incident_fields_from_graph

from src.agents.graph import (
    GraphState,
    RAG_STRONG_THRESHOLD,
    RAG_WEAK_THRESHOLD,
    graph,
    validate_recommended_action,
    verifier_node,
)


@pytest.fixture
def offline_verifier_llm(monkeypatch):
    monkeypatch.setattr(
        "src.agents.graph.call_llm_with_fallback",
        lambda prompt, node_name: ("offline verifier test reasoning", "test"),
    )


def _make_state(**overrides):
    state = {
        "alert": "",
        "plan": None,
        "diagnosis": None,
        "proposed_fix": None,
        "recommended_action": None,
        "action_namespace": None,
        "action_deployment_name": None,
        "action_replicas": None,
        "trust_score": None,
        "classification": None,
        "verifier_reasoning": None,
        "log": [],
        "model_used": {},
        "tool_calls": [],
        "retrieved_chunks": [],
    }
    state.update(overrides)
    return state


def test_verifier_reasoning_survives_graph_and_persistence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agents.graph.call_llm_with_fallback",
        lambda prompt, node_name: ("mock verifier reasoning", "test"),
    )
    builder = StateGraph(GraphState)
    builder.add_node("verifier", verifier_node)
    builder.add_edge(START, "verifier")
    builder.add_edge("verifier", END)
    compiled_graph = builder.compile()

    final_state = compiled_graph.invoke(_make_state(
        alert="paymentservice is unhealthy",
        plan="Inspect the paymentservice workload.",
        diagnosis="The workload is unhealthy.",
        proposed_fix="Inspect pod health and probe configuration.",
    ))
    assert final_state["verifier_reasoning"] == "mock verifier reasoning"

    fields = _incident_fields_from_graph(final_state)
    assert fields["verifier_reasoning"] == "mock verifier reasoning"

    monkeypatch.setattr(api_main, "DB_PATH", tmp_path / "test_ai_sre.db")
    incident = api_main.persist_incident(fields, trusted_graph=True)
    with api_main.get_db_connection() as connection:
        row = connection.execute(
            "SELECT verifier_reasoning FROM incidents WHERE id = ?",
            (incident["id"],),
        ).fetchone()
    assert row["verifier_reasoning"] == "mock verifier reasoning"


def test_verifier_prompt_states_the_gate_verdict_instead_of_asking_the_llm_to_decide(monkeypatch):
    prompts = []

    def capture(prompt, node_name):
        prompts.append(prompt)
        return "REVIEW because the runbook match is weak.", "test"

    monkeypatch.setattr("src.agents.graph.call_llm_with_fallback", capture)
    state = _make_state(
        alert="checkoutservice slow",
        diagnosis="CPU throttling on checkoutservice",
        tool_calls=[{"tool": "list_pods", "args": {}, "result": [{"name": "checkoutservice-1", "status": "Running"}]}],
        retrieved_chunks=[{"score": 0.6, "chunk_id": "c", "source_file": "f.txt", "chunk_text": "t"}],
    )

    result = verifier_node(state)

    assert result["classification"] == "REVIEW"
    assert "already classified this proposed fix as REVIEW" in prompts[0]
    assert "Respond with ACCEPT, REVIEW, or REJECT" not in prompts[0]
    assert result["verifier_reasoning"].startswith("REVIEW")


def test_verifier_rejects_when_no_evidence_and_no_rag_match(offline_verifier_llm):
    state = _make_state(
        alert="Namespace prod has an unrelated application issue",
        diagnosis="The issue is not related to DNS or Kubernetes control-plane startup; there are no relevant logs.",
        tool_calls=[
            {"tool": "list_pods", "args": {"namespace": "prod"}, "result": []},
            {"tool": "list_deployments", "args": {"namespace": "prod"}, "result": []},
        ],
        retrieved_chunks=[],
    )

    actual = verifier_node(state)
    assert actual["trust_score"] == pytest.approx(0.0)
    assert actual["classification"] == "REJECT"


def test_verifier_review_for_real_kube_system_case(offline_verifier_llm):
    state = _make_state(
        alert="Pod coredns-589f44dc88-jz8ft in kube-system is CrashLoopBackOff",
        plan="Inspect coredns pod status in kube-system and review relevant logs.",
        diagnosis=(
            "The coredns-589f44dc88-jz8ft pod in kube-system is Running with Restart Count: 5. "
            "Logs show [INFO] plugin/kubernetes: waiting for Kubernetes API before starting server "
            "and [WARNING] plugin/kubernetes: starting server with unsynced Kubernetes API."
        ),
        tool_calls=[
            {
                "tool": "list_pods",
                "args": {"namespace": "kube-system"},
                "result": [
                    {
                        "name": "coredns-589f44dc88-jz8ft",
                        "status": "Running",
                        "restart_count": 5,
                        "reason": "plugin/kubernetes: starting server with unsynced Kubernetes API",
                    }
                ],
            }
        ],
        retrieved_chunks=[
            {
                "score": 0.52840364,
                "chunk_id": "2025_github_k8s-upgrade-kube-proxy-iptables_chunk0",
                "source_file": "2025_github_k8s-upgrade-kube-proxy-iptables.txt",
                "chunk_text": "A Kubernetes networking upgrade left kube-proxy and kubelet out of sync.",
            }
        ],
    )

    actual = verifier_node(state)
    assert actual["trust_score"] == pytest.approx(0.52840364)
    assert actual["classification"] == "REVIEW"


def test_verifier_accepts_when_strong_rag_and_tool_evidence_both_present(offline_verifier_llm):
    real_chunk_text = (
        "https://web.archive.org/web/20211201033341/https://codeascraft.com/2012/01/23/solr-bittorrent-index-replication/ "
        "Title: Etsy: Sending multicast traffic without properly configured switches causes a global outage. "
        "Category: Networking / Config Error."
    )
    state = _make_state(
        alert="Global network outage caused by multicast switch misconfiguration",
        plan="Review network configuration and switch health.",
        diagnosis=real_chunk_text,
        tool_calls=[
            {
                "tool": "get_switch_health",
                "args": {"site": "global"},
                "result": [{"switch": "core-1", "status": "misconfigured", "reason": "multicast routing mismatch"}],
            }
        ],
        retrieved_chunks=[
            {
                "score": 1.0,
                "chunk_id": "2012_etsy_multicast-switch-misconfig-outage_chunk0",
                "source_file": "2012_etsy_multicast-switch-misconfig-outage.txt",
                "chunk_text": real_chunk_text,
            }
        ],
    )

    actual = verifier_node(state)
    assert actual["trust_score"] == pytest.approx(1.0)
    assert actual["classification"] == "ACCEPT"
    assert RAG_STRONG_THRESHOLD > RAG_WEAK_THRESHOLD


def test_action_validation_forces_invalid_enum_to_none():
    tool_calls = [{
        "tool": "list_deployments",
        "args": {"namespace": "kube-system"},
        "result": [{"name": "coredns", "desired_replicas": 2, "available_replicas": 1}],
    }]

    validated, log = validate_recommended_action({
        "recommended_action": "delete_namespace",
        "action_namespace": "kube-system",
        "action_deployment_name": "coredns",
        "action_replicas": None,
    }, tool_calls)

    assert validated["recommended_action"] == "none"
    assert validated["action_deployment_name"] is None
    assert any("invalid recommended_action" in entry.lower() for entry in log)


def test_action_validation_rejects_hallucinated_deployment_target():
    tool_calls = [{
        "tool": "list_deployments",
        "args": {"namespace": "kube-system"},
        "result": [{"name": "coredns", "desired_replicas": 2, "available_replicas": 1}],
    }]

    validated, log = validate_recommended_action({
        "recommended_action": "restart_deployment",
        "action_namespace": "kube-system",
        "action_deployment_name": "not-the-real-deployment",
        "action_replicas": None,
    }, tool_calls)

    assert validated["recommended_action"] == "none"
    assert validated["action_deployment_name"] is None
    assert any("not observed" in entry.lower() or "not found" in entry.lower() for entry in log)


@pytest.mark.live
def test_real_coredns_graph_run_has_valid_action_recommendation():
    state = _make_state(
        alert="Pod coredns-589f44dc88-jz8ft in kube-system is CrashLoopBackOff",
        plan="Inspect coredns pod status in kube-system and review logs.",
        diagnosis=(
            "The coredns-589f44dc88-jz8ft pod in kube-system is Running with Restart Count: 5. "
            "Logs show [INFO] plugin/kubernetes: waiting for Kubernetes API before starting server "
            "and [WARNING] plugin/kubernetes: starting server with unsynced Kubernetes API."
        ),
        tool_calls=[
            {
                "tool": "list_pods",
                "args": {"namespace": "kube-system"},
                "result": [{"name": "coredns-589f44dc88-jz8ft", "status": "Running", "restart_count": 5}],
            },
            {
                "tool": "list_deployments",
                "args": {"namespace": "kube-system"},
                "result": [{"name": "coredns", "desired_replicas": 2, "available_replicas": 1}],
            },
        ],
        retrieved_chunks=[
            {
                "score": 0.52840364,
                "chunk_id": "2025_github_k8s-upgrade-kube-proxy-iptables_chunk0",
                "source_file": "2025_github_k8s-upgrade-kube-proxy-iptables.txt",
                "chunk_text": "A Kubernetes networking upgrade left kube-proxy and kubelet out of sync.",
            }
        ],
    )

    state = graph.invoke(state)
    assert state["recommended_action"] == "restart_deployment"
    assert state["action_namespace"] == "kube-system"
    assert state["action_deployment_name"] == "coredns"
    assert state["action_replicas"] in (None, 2)
