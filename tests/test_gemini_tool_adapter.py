import os
from types import SimpleNamespace

import pytest
from google import genai
from google.genai import types
from google.genai.errors import ClientError

import src.agents.graph as graph_module
from src.agents.graph import (
    GEMINI_MODEL,
    MCPToolDescriptor,
    _build_gemini_function_response,
    _format_gemini_tools,
    _parse_gemini_function_calls,
)


GET_POD_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "namespace": {"type": "string"},
        "pod_name": {"type": "string"},
    },
    "required": ["namespace", "pod_name"],
}


def _get_pod_status_descriptor() -> MCPToolDescriptor:
    return MCPToolDescriptor(
        name="get_pod_status",
        description="Return phase, ready-container count, and restart count for a pod.",
        input_schema=GET_POD_STATUS_SCHEMA,
        owner_session=None,  # type: ignore[arg-type]
    )


def test_format_gemini_tools_preserves_mcp_schema():
    formatted = _format_gemini_tools([_get_pod_status_descriptor()])

    assert len(formatted) == 1
    declaration = formatted[0].function_declarations[0]
    assert declaration.name == "get_pod_status"
    assert declaration.description.startswith("Return phase")
    assert declaration.parameters_json_schema == GET_POD_STATUS_SCHEMA


def test_parse_gemini_function_calls_extracts_name_and_args():
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call-7",
                                name="get_pod_status",
                                args={"namespace": "online-boutique", "pod_name": "paymentservice"},
                            )
                        )
                    ],
                )
            )
        ]
    )

    assert _parse_gemini_function_calls(response) == [{
        "name": "get_pod_status",
        "args": {"namespace": "online-boutique", "pod_name": "paymentservice"},
        "call_id_or_index": "call-7",
    }]


def test_build_gemini_function_response():
    content = _build_gemini_function_response(
        "get_pod_status",
        {"phase": "Running", "restart_count": 3},
    )

    assert isinstance(content, types.Content)
    assert content.role == "tool"
    assert content.parts[0].function_response.name == "get_pod_status"
    assert content.parts[0].function_response.response == {
        "result": {"phase": "Running", "restart_count": 3}
    }


def test_gemini_tool_loop_executes_calls_and_returns_grounded_diagnosis(monkeypatch):
    responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(function_call=types.FunctionCall(
                            name="get_pod_status",
                            args={"namespace": "online-boutique", "pod_name": "paymentservice"},
                        ))],
                    )
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="paymentservice is crash-looping.")],
                    )
                )
            ]
        ),
    ]

    class FakeModels:
        def generate_content(self, **kwargs):
            return responses.pop(0)

    class FakeSession:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(
                structuredContent={"result": {"phase": "Running", "restart_count": 3}},
                content=[],
            )

    monkeypatch.setattr(graph_module, "gemini_client", SimpleNamespace(models=FakeModels()))
    descriptor = _get_pod_status_descriptor()
    state = {
        "alert": "paymentservice pod in online-boutique is crash-looping",
        "plan": "Inspect paymentservice pod status.",
        "tool_calls": [],
    }

    diagnosis, provider, tool_calls = __import__("anyio").run(
        graph_module._run_gemini_tool_loop,
        state,
        [descriptor],
        {"get_pod_status": FakeSession()},
        {"namespace": "online-boutique", "deployment_name": None},
    )

    assert diagnosis == "paymentservice is crash-looping."
    assert provider == "gemini"
    assert tool_calls == [{
        "tool": "get_pod_status",
        "args": {"namespace": "online-boutique", "pod_name": "paymentservice"},
        "result": {"phase": "Running", "restart_count": 3},
    }]


def test_gemini_synthesis_failure_preserves_evidence_and_marks_diagnosis_degraded(monkeypatch):
    function_call_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=types.FunctionCall(
                        name="get_pod_status",
                        args={"namespace": "online-boutique", "pod_name": "paymentservice"},
                    ))],
                )
            )
        ]
    )

    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return function_call_response
            raise ClientError(429, {
                "error": {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}
            })

    class FakeSession:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(
                structuredContent={"result": {"phase": "Running", "restart_count": 9}},
                content=[],
            )

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(graph_module, "gemini_client", fake_client)
    descriptor = _get_pod_status_descriptor()
    state = {
        "alert": "paymentservice pod in online-boutique is crash-looping",
        "plan": "Inspect paymentservice pod status.",
        "tool_calls": [],
    }

    diagnosis, provider, tool_calls = __import__("anyio").run(
        graph_module._run_gemini_tool_loop,
        state,
        [descriptor],
        {"get_pod_status": FakeSession()},
        {"namespace": "online-boutique", "deployment_name": None},
    )
    monkeypatch.setattr(
        graph_module.anyio,
        "run",
        lambda function, state: (diagnosis, provider, tool_calls),
    )
    updated_state = graph_module._run_log_analysis({
        "alert": state["alert"],
        "plan": state["plan"],
        "log": [],
        "model_used": {},
        "tool_calls": [],
    })

    assert provider == "gemini"
    assert tool_calls[0]["tool"] == "get_pod_status"
    assert "get_pod_status" in diagnosis
    assert "ClientError" in diagnosis
    assert diagnosis.startswith(graph_module.GEMINI_SYNTHESIS_FAILURE_PREFIX)
    assert updated_state["diagnosis_degraded"] is True


@pytest.mark.live
def test_live_gemini_accepts_converted_tool_schema_and_calls_tool():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            "You must call get_pod_status before answering. Check pod paymentservice "
            "in namespace online-boutique."
        ),
        config=types.GenerateContentConfig(
            tools=_format_gemini_tools([_get_pod_status_descriptor()]),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        ),
    )

    function_calls = _parse_gemini_function_calls(response)
    assert function_calls
    assert function_calls[0]["name"] == "get_pod_status"