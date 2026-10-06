import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "DB_PATH", tmp_path / "test_ai_sre.db")
    monkeypatch.delenv("AI_SRE_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_EXECUTE_ON_ACCEPT", raising=False)
    return TestClient(api_main.app)


@pytest.fixture
def executor_calls(monkeypatch):
    calls = []

    def fake_execute(state):
        calls.append(state)
        return {"executed": True, "action": state["recommended_action"], "verification_status": "verified"}

    monkeypatch.setattr(api_main, "execute_recommended_action", fake_execute)
    return calls


def _incident(classification="ACCEPT", action="restart_deployment"):
    return api_main.persist_incident(
        {
            "alert": "Deployment cartservice in namespace online-boutique is crash-looping",
            "diagnosis": "cartservice exits with OOMKilled",
            "proposed_fix": "Restart cartservice",
            "trust_score": 0.9,
            "classification": classification,
            "verifier_reasoning": "test reasoning",
            "recommended_action": action,
            "action_namespace": "online-boutique",
            "action_deployment_name": "cartservice",
            "action_replicas": None,
            "approval_status": "pending",
        },
        trusted_graph=True,
    )


def test_execute_requires_approval(client, executor_calls):
    incident = _incident()

    response = client.post(f"/incidents/{incident['id']}/execute")

    assert response.status_code == 409
    assert "approved" in response.json()["detail"]
    assert executor_calls == []


def test_execute_runs_the_action_exactly_once(client, executor_calls):
    incident = _incident()
    assert client.post(f"/incidents/{incident['id']}/approve").status_code == 200

    first = client.post(f"/incidents/{incident['id']}/execute")
    second = client.post(f"/incidents/{incident['id']}/execute")

    assert first.status_code == 200
    assert first.json()["executed"] is True
    assert first.json()["execution_status"] == "succeeded"
    assert second.status_code == 409
    assert len(executor_calls) == 1


def test_execution_in_progress_blocks_a_concurrent_execute(client, executor_calls):
    incident = _incident()
    client.post(f"/incidents/{incident['id']}/approve")

    api_main._claim_execution(incident["id"], require_approval=True)
    response = client.post(f"/incidents/{incident['id']}/execute")

    assert response.status_code == 409
    assert "in progress" in response.json()["detail"]
    assert executor_calls == []


def test_execution_error_is_recorded_and_can_be_retried(client, monkeypatch, executor_calls):
    incident = _incident()
    client.post(f"/incidents/{incident['id']}/approve")

    def failing_execute(state):
        raise RuntimeError("kubernetes MCP server crashed")

    monkeypatch.setattr(api_main, "execute_recommended_action", failing_execute)
    failed = client.post(f"/incidents/{incident['id']}/execute")
    stored = client.get(f"/incidents/{incident['id']}").json()

    assert failed.status_code == 500
    assert stored["executed"] is False
    assert stored["execution_status"] == "failed"
    assert "kubernetes MCP server crashed" in stored["execution_result"]["reason"]

    monkeypatch.setattr(api_main, "execute_recommended_action", lambda state: {"executed": True})
    retried = client.post(f"/incidents/{incident['id']}/execute")
    assert retried.status_code == 200
    assert retried.json()["execution_status"] == "succeeded"


def test_auto_execution_failure_keeps_the_incident(client, monkeypatch):
    monkeypatch.setenv("AUTO_EXECUTE_ON_ACCEPT", "true")

    def failing_execute(state):
        raise RuntimeError("cluster unreachable")

    monkeypatch.setattr(api_main, "execute_recommended_action", failing_execute)

    incident = _incident()

    stored = client.get(f"/incidents/{incident['id']}")
    assert stored.status_code == 200
    assert stored.json()["execution_status"] == "failed"
    assert stored.json()["execution_trigger"] == "auto"


def test_incident_without_executable_action_cannot_be_executed(client, executor_calls):
    incident = _incident(classification="REVIEW", action="none")
    client.post(f"/incidents/{incident['id']}/approve")

    response = client.post(f"/incidents/{incident['id']}/execute")

    assert response.status_code == 409
    assert "no executable" in response.json()["detail"]
    assert client.get(f"/incidents/{incident['id']}").json()["execution_status"] is None
    assert executor_calls == []


def test_reject_classified_incident_cannot_be_approved(client):
    incident = _incident(classification="REJECT", action="none")

    response = client.post(f"/incidents/{incident['id']}/approve")

    assert response.status_code == 409
    assert client.post(f"/incidents/{incident['id']}/reject").status_code == 200


def test_approval_cannot_change_after_execution(client, executor_calls):
    incident = _incident()
    client.post(f"/incidents/{incident['id']}/approve")
    client.post(f"/incidents/{incident['id']}/execute")

    response = client.post(f"/incidents/{incident['id']}/reject")

    assert response.status_code == 409


def test_state_changing_endpoints_require_api_key_when_configured(client, monkeypatch):
    incident = _incident()
    monkeypatch.setenv("AI_SRE_API_KEY", "test-secret")

    assert client.post(f"/incidents/{incident['id']}/approve").status_code == 401
    assert client.post(
        f"/incidents/{incident['id']}/approve", headers={"X-API-Key": "wrong"}
    ).status_code == 401
    assert client.post("/incidents/analyze", json={"alert": "x"}).status_code == 401
    assert client.get("/incidents").status_code == 200
    assert client.get("/health").status_code == 200

    approved = client.post(f"/incidents/{incident['id']}/approve", headers={"X-API-Key": "test-secret"})
    assert approved.status_code == 200
    assert approved.json()["approval_status"] == "approved"


def test_untrusted_incident_create_cannot_set_decision_fields(client):
    response = client.post("/incidents", json={
        "alert": "manual alert",
        "classification": "ACCEPT",
        "recommended_action": "restart_deployment",
        "action_namespace": "default",
        "action_deployment_name": "web",
    })

    assert response.status_code == 200
    assert response.json()["classification"] is None
    assert response.json()["recommended_action"] is None


def test_analyze_endpoint_runs_the_graph_runner(client, monkeypatch):
    import src.agents.runner as runner

    def fake_run(alert):
        incident = api_main.persist_incident({"alert": alert, "classification": "REVIEW"}, trusted_graph=True)
        return {}, incident

    monkeypatch.setattr(runner, "run_alert_and_persist", fake_run)

    response = client.post("/incidents/analyze", json={"alert": "  checkoutservice is down  "})

    assert response.status_code == 200
    assert response.json()["alert"] == "checkoutservice is down"
    assert client.post("/incidents/analyze", json={"alert": "   "}).status_code == 422
