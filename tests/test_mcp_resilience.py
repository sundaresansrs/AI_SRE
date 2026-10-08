import anyio
import pytest

import src.agents.graph as graph_module
from src.agents.graph import _has_real_tool_evidence, _is_tool_error, classify_verifier_state


def _state(tool_calls, retrieved_chunks=None):
    return {
        "tool_calls": tool_calls,
        "retrieved_chunks": retrieved_chunks or [],
        "diagnosis_degraded": False,
    }


@pytest.mark.parametrize("call", [
    {"tool": "list_pods", "result": "Unable to load the current kubeconfig context: Invalid kube-config file."},
    {"tool": "list_pods", "result": "Kubernetes API error (403): Forbidden"},
    {"tool": "list_workflow_runs", "result": "GitHub error: GitHub is unavailable: the GITHUB_PAT environment variable is not set."},
    {"tool": "get_issue", "result": {"error": "GitHub API error (404): Not Found"}},
    {"tool": "restart_deployment", "result": {"error": "deployment not found", "namespace": "default"}},
    {"tool": "instant_query", "result": ["Error executing tool instant_query: Prometheus connection error: refused"]},
    {"tool": "instant_query", "result": ["anything"], "is_error": True},
])
def test_tool_error_results_are_not_evidence(call):
    assert _is_tool_error(call)
    assert not _has_real_tool_evidence([call])


def test_real_observations_still_count_as_evidence():
    calls = [
        {"tool": "get_pod_logs", "result": "ERROR: payment gateway timeout after 30s"},
        {"tool": "list_pods", "result": [{"name": "cartservice-abc", "status": "Running"}]},
    ]

    assert not any(_is_tool_error(call) for call in calls)
    assert _has_real_tool_evidence(calls)


def test_unreachable_cluster_with_strong_rag_match_is_never_accepted():
    state = _state(
        tool_calls=[
            {"tool": "list_pods", "args": {"namespace": "default"}, "result": "Unable to load the current kubeconfig context: x"},
            {"tool": "list_deployments", "args": {"namespace": "default"}, "result": "Unable to load the current kubeconfig context: x"},
        ],
        retrieved_chunks=[{"score": 0.95, "chunk_id": "c0", "source_file": "f.txt", "chunk_text": "t"}],
    )

    _, classification = classify_verifier_state(state)

    assert classification == "REVIEW"


def test_unreachable_cluster_without_rag_match_is_rejected():
    state = _state(tool_calls=[
        {"tool": "list_pods", "args": {"namespace": "default"}, "result": "Unable to load the current kubeconfig context: x"},
    ])

    _, classification = classify_verifier_state(state)

    assert classification == "REJECT"


def test_unknown_tool_name_is_recorded_as_error():
    raw_result, tool_call, _ = anyio.run(
        graph_module._execute_mcp_tool,
        "delete_cluster",
        {"namespace": "default"},
        {},
        {"namespace": None, "deployment_name": None},
    )

    assert tool_call["is_error"] is True
    assert "no connected MCP server" in raw_result


def test_github_server_starts_without_token_and_reports_unavailable(monkeypatch):
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.setattr(graph_module, "MCP_SERVER_NAMES", ("github",))

    async def scenario():
        stack, sessions = await graph_module._open_mcp_servers()
        try:
            tools, owners = await graph_module._fetch_mcp_tools(sessions)
            _, tool_call, _ = await graph_module._execute_mcp_tool(
                "list_workflow_runs",
                {"repo_full_name": "octo/repo"},
                owners,
                {"namespace": None, "deployment_name": None},
            )
            return {tool.name for tool in tools}, tool_call
        finally:
            await graph_module._close_mcp_servers(stack, sessions)

    tool_names, tool_call = anyio.run(scenario)

    assert "list_workflow_runs" in tool_names
    assert "GITHUB_PAT" in str(tool_call["result"])
    assert _is_tool_error(tool_call)


def test_failing_mcp_server_is_skipped(monkeypatch):
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.setattr(graph_module, "MCP_SERVER_NAMES", ("does_not_exist", "github"))
    monkeypatch.setattr(graph_module, "MCP_STARTUP_TIMEOUT_SECONDS", 15)

    async def scenario():
        stack, sessions = await graph_module._open_mcp_servers()
        await graph_module._close_mcp_servers(stack, sessions)
        return set(sessions)

    assert anyio.run(scenario) == {"github"}


def test_no_available_mcp_server_raises(monkeypatch):
    monkeypatch.setattr(graph_module, "MCP_SERVER_NAMES", ("does_not_exist",))
    monkeypatch.setattr(graph_module, "MCP_STARTUP_TIMEOUT_SECONDS", 15)

    with pytest.raises(RuntimeError, match="No MCP server could be started"):
        anyio.run(graph_module._open_mcp_servers)


def test_log_analysis_is_marked_degraded_when_gemini_fallback_is_unavailable(monkeypatch):
    async def fake_analysis(state):
        # Real tool evidence was gathered before Groq failed, but Gemini could not synthesize.
        text = f"{graph_module.GEMINI_UNAVAILABLE_PREFIX}: ClientError. No diagnosis was produced."
        return text, "gemini", [{"tool": "list_pods", "args": {}, "result": [{"name": "p", "status": "Running"}]}]

    monkeypatch.setattr(graph_module, "_run_log_analysis_with_tools", fake_analysis)
    state = {"alert": "a", "plan": "p", "log": [], "model_used": {}, "tool_calls": []}

    updated = graph_module._run_log_analysis(state)

    assert updated["diagnosis_degraded"] is True
    assert classify_verifier_state(updated)[1] != "ACCEPT"


def test_default_gemini_model_is_not_the_retired_2_5_flash():
    assert "gemini-2.5-flash" not in graph_module.GEMINI_MODEL
