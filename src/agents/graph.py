"""LangGraph orchestration for the AI-SRE agent pipeline."""

import os
import json
from contextlib import AsyncExitStack
from pprint import pprint
from pathlib import Path
from typing import Any, Optional, TypedDict

import anyio
import httpx
import joblib
import numpy as np
import pandas as pd
from google import genai
from groq import BadRequestError, Groq, RateLimitError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


try:
    GROQ_API_KEY = os.environ["GROQ_API_KEY"]
except KeyError:
    raise RuntimeError("GROQ_API_KEY environment variable is required") from None

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "openai/gpt-oss-20b"

try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    raise RuntimeError("GEMINI_API_KEY environment variable is required") from None

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "models/gemini-2.5-flash"

# Test-only switch; production runs leave this unset.
SIMULATE_RATE_LIMIT_ON_NODE: str | None = None
QDRANT_URL = "http://localhost:6333"
RAG_COLLECTION = "runbook_chunks"
RAG_MODEL_NAME = "all-MiniLM-L6-v2"
RAG_TOP_K = 3
RAG_MIN_SCORE = 0.35
_rag_model: SentenceTransformer | None = None
ANOMALY_THRESHOLD = 0.588
ANOMALY_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lightgbm_final_classifier.joblib"
_anomaly_model = joblib.load(ANOMALY_MODEL_PATH)


class GraphState(TypedDict):
    alert: str
    plan: str | None
    diagnosis: str | None
    proposed_fix: str | None
    trust_score: str | None
    log: list[str]
    model_used: dict[str, str]
    tool_calls: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]


def _engineer_final_features(rows: pd.DataFrame) -> pd.DataFrame:
    base_columns = [column for column in _anomaly_model.feature_name() if column not in {
        "lag_1_diff", "lag_3_ratio", "rolling_zscore_5", "velocity_sign_change",
        "rolling_max_5", "rolling_min_5", "rolling_mean_15", "rolling_std_15",
        "rolling_mean_30", "rolling_std_30", "zscore_vs_15", "zscore_vs_30",
    }]
    prepared = rows.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared = prepared.sort_values(["service_name", "timestamp"]).reset_index(drop=True)
    values = prepared["value"].astype(float)
    grouped = prepared.groupby("service_name", sort=False)
    prepared["feature_0_raw_value"] = values
    prepared["feature_1_log_value"] = np.log1p(np.abs(values))
    prepared["feature_2_high_value"] = (values > values.quantile(0.75)).astype(float)
    prepared["feature_3_hour"] = prepared["timestamp"].dt.hour.astype(float)
    prepared["rolling_mean_5"] = grouped["value"].transform(lambda series: series.rolling(5, min_periods=1).mean())
    prepared["rolling_std_5"] = grouped["value"].transform(lambda series: series.rolling(5, min_periods=1).std().fillna(0))
    prepared["velocity"] = grouped["value"].diff().fillna(0)
    prepared["acceleration"] = prepared.groupby("service_name", sort=False)["velocity"].diff().fillna(0)
    prepared["hour_sine"] = np.sin(2 * np.pi * prepared["feature_3_hour"] / 24.0)
    prepared["hour_cosine"] = np.cos(2 * np.pi * prepared["feature_3_hour"] / 24.0)
    for column in base_columns:
        if column.startswith("service_"):
            prepared[column] = (prepared["service_name"] == column.removeprefix("service_")).astype(float)
    prepared["lag_1_diff"] = grouped["value"].diff().abs().fillna(0)
    lag_3 = grouped["value"].shift(3).fillna(0)
    prepared["lag_3_ratio"] = np.where(np.abs(lag_3) > 1e-6, np.abs(values / (lag_3 + 1e-8)), 1.0).clip(0.01, 100.0)
    prepared["rolling_zscore_5"] = np.where(prepared["rolling_std_5"] > 1e-6, (values - prepared["rolling_mean_5"]) / (prepared["rolling_std_5"] + 1e-8), 0.0).clip(-10, 10)
    prepared["velocity_sign_change"] = (grouped["velocity"].shift(1).fillna(0) * prepared["velocity"] < 0).astype(float)
    for window in (5, 15, 30):
        prepared[f"rolling_max_{window}"] = grouped["value"].transform(lambda series: series.rolling(window, min_periods=1).max()) if window == 5 else None
        if window == 5:
            prepared["rolling_min_5"] = grouped["value"].transform(lambda series: series.rolling(5, min_periods=1).min())
        else:
            prepared[f"rolling_mean_{window}"] = grouped["value"].transform(lambda series, window=window: series.rolling(window, min_periods=1).mean())
            prepared[f"rolling_std_{window}"] = grouped["value"].transform(lambda series, window=window: series.rolling(window, min_periods=1).std().fillna(0))
    prepared["zscore_vs_15"] = np.where(prepared["rolling_std_15"] > 1e-6, (values - prepared["rolling_mean_15"]) / (prepared["rolling_std_15"] + 1e-8), 0.0).clip(-10, 10)
    prepared["zscore_vs_30"] = np.where(prepared["rolling_std_30"] > 1e-6, (values - prepared["rolling_mean_30"]) / (prepared["rolling_std_30"] + 1e-8), 0.0).clip(-10, 10)
    return prepared[_anomaly_model.feature_name()]


def _score_metric_rows(row: dict[str, Any] | pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows = row if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
    prepared_rows = rows.copy()
    prepared_rows["timestamp"] = pd.to_datetime(prepared_rows["timestamp"])
    prepared_rows = prepared_rows.sort_values(["service_name", "timestamp"]).reset_index(drop=True)
    probabilities = _anomaly_model.predict(_engineer_final_features(prepared_rows))
    return prepared_rows, probabilities


def check_for_anomaly_trigger(row: dict[str, Any] | pd.DataFrame) -> Optional[str]:
    rows, probabilities = _score_metric_rows(row)
    crossing = np.flatnonzero(probabilities >= ANOMALY_THRESHOLD)
    if len(crossing) == 0:
        return None
    index = int(crossing[np.argmax(probabilities[crossing])])
    source = rows.iloc[index]
    probability = float(probabilities[index])
    return (
        f"Anomaly detected: service={source['service_name']}, metric={source['metric_name']}, "
        f"value={source['value']}, timestamp={source['timestamp']}, probability={probability:.6f}"
    )


def _groq_completion(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
    arguments: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        arguments["tools"] = tools
        arguments["tool_choice"] = "auto"
    return groq_client.chat.completions.create(**arguments)


def call_llm_with_fallback(prompt: str, node_name: str) -> tuple[str, str]:
    try:
        if SIMULATE_RATE_LIMIT_ON_NODE == node_name:
            raise RateLimitError(
                "Simulated rate limit for fallback verification",
                response=httpx.Response(
                    429,
                    request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
                ),
                body=None,
            )
        response = _groq_completion([
            {
                "role": "system",
                "content": "You are an SRE assistant. Be concise and technically specific.",
            },
            {"role": "user", "content": prompt},
        ])
        return response.choices[0].message.content.strip(), "groq"
    except RateLimitError:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip(), "gemini"


def _server_parameters(name: str) -> StdioServerParameters:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    server_path = os.path.join(root, "src", "mcp_servers", name, "server.py")
    environment = os.environ.copy()
    return StdioServerParameters(
        command=os.sys.executable,
        args=[server_path],
        env=environment,
        cwd=root,
    )


async def _open_mcp_servers() -> Any:
    stack = AsyncExitStack()
    await stack.__aenter__()
    sessions: dict[str, ClientSession] = {}
    for name in ("kubernetes", "github", "observability"):
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(_server_parameters(name))
        )
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        sessions[name] = session
    return stack, sessions


async def _close_mcp_servers(stack: Any, sessions: dict[str, ClientSession]) -> None:
    await stack.aclose()


async def _fetch_mcp_tools(sessions: dict[str, ClientSession]) -> tuple[list[dict[str, Any]], dict[str, ClientSession]]:
    tools: list[dict[str, Any]] = []
    owners: dict[str, ClientSession] = {}
    for session in sessions.values():
        response = await session.list_tools()
        for tool in response.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            })
            owners[tool.name] = session
    return tools, owners


def _tool_result_value(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured and "result" in structured:
        return structured["result"]
    return [getattr(block, "text", str(block)) for block in getattr(result, "content", [])]


def _is_tool_validation_error(error: BadRequestError) -> bool:
    message = str(error).lower()
    return "tool_use_failed" in message or "not in request.tools" in message


async def _run_log_analysis_with_tools(state: GraphState) -> tuple[str, str, list[dict[str, Any]]]:
    stack, sessions = await _open_mcp_servers()
    try:
        tools, owners = await _fetch_mcp_tools(sessions)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an SRE log analyst. Use the available MCP tools to inspect real data before diagnosing. "
                    "For a CrashLoopBackOff, inspect Kubernetes resources and logs when appropriate. "
                    "Do not invent observations. After tool results, provide a concise diagnosis grounded in them."
                ),
            },
            {
                "role": "user",
                "content": f"Alert: {state['alert']}\nInvestigation plan: {state['plan']}",
            },
        ]
        tool_calls: list[dict[str, Any]] = []
        available_tool_names = ", ".join(sorted(owners))
        validation_retries = 0
        tool_rounds = 0
        while tool_rounds < 5:
            tool_rounds += 1
            try:
                response = _groq_completion(messages, tools)
            except BadRequestError as error:
                if not _is_tool_validation_error(error):
                    raise
                if validation_retries == 2:
                    return (
                        "Tool selection failed after two corrective retries. "
                        f"The model requested a tool that was not available. Available tools: {available_tool_names}.",
                        "groq",
                        tool_calls,
                    )
                validation_retries += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "The requested tool does not exist in the available MCP tools. "
                        f"Choose only from these exact tool names: {available_tool_names}. "
                        "Retry your tool selection using one or more of those tools."
                    ),
                })
                continue
            choice = response.choices[0]
            assistant = choice.message
            requested = assistant.tool_calls or []
            if not requested:
                content = (assistant.content or "").strip()
                if content:
                    return content, "groq", tool_calls
                messages.append({
                    "role": "user",
                    "content": (
                        "The tool results are available above. Now provide the final diagnosis grounded in those "
                        "results, including specific resource names and observed statuses where available."
                    ),
                })
                final_response = _groq_completion(messages)
                return (final_response.choices[0].message.content or "").strip(), "groq", tool_calls
            messages.append({
                "role": "assistant",
                "content": assistant.content or "",
                "tool_calls": [call.model_dump() for call in requested],
            })
            for call in requested:
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                result = await owners[call.function.name].call_tool(call.function.name, arguments)
                raw_result = _tool_result_value(result)
                tool_calls.append({"tool": call.function.name, "args": arguments, "result": raw_result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(raw_result, default=str),
                })
            if validation_retries:
                validation_retries = 0
        messages.append({
            "role": "user",
            "content": (
                "Stop selecting tools and provide the final diagnosis now. Ground it in the real tool results above, "
                "including specific resource names and observed statuses where available."
            ),
        })
        final_response = _groq_completion(messages)
        return (final_response.choices[0].message.content or "").strip(), "groq", tool_calls
    finally:
        await _close_mcp_servers(stack, sessions)


def _run_log_analysis(state: GraphState) -> GraphState:
    diagnosis, provider, tool_calls = anyio.run(_run_log_analysis_with_tools, state)
    updated_state = dict(state)
    updated_state["log"] = [*state["log"], "log_analysis"]
    updated_state["model_used"] = {**state["model_used"], "log_analysis": provider}
    updated_state["tool_calls"] = [*state["tool_calls"], *tool_calls]
    updated_state["diagnosis"] = diagnosis
    return updated_state


def _with_node_result(state: GraphState, node_name: str, field: str, prompt: str) -> GraphState:
    text, provider = call_llm_with_fallback(prompt, node_name)
    updated_state = dict(state)
    updated_state["log"] = [*state["log"], node_name]
    updated_state["model_used"] = {**state["model_used"], node_name: provider}
    updated_state[field] = text
    return updated_state


def _retrieve_runbook_chunks(diagnosis: str) -> list[dict[str, Any]]:
    global _rag_model
    if _rag_model is None:
        _rag_model = SentenceTransformer(RAG_MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)
    query_vector = _rag_model.encode(
        [diagnosis],
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].tolist()
    points = client.query_points(
        collection_name=RAG_COLLECTION,
        query=query_vector,
        limit=RAG_TOP_K,
        with_payload=True,
    ).points
    return [
        {
            "score": point.score,
            "chunk_id": (point.payload or {}).get("chunk_id"),
            "source_file": (point.payload or {}).get("source_file"),
            "chunk_text": (point.payload or {}).get("chunk_text", ""),
        }
        for point in points
        if point.score >= RAG_MIN_SCORE
    ]


def _format_retrieved_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No runbook chunk met the minimum similarity threshold; do not claim a runbook match."
    return "\n\n".join(
        f"Runbook chunk {chunk['chunk_id']} from {chunk['source_file']} "
        f"(similarity={chunk['score']:.4f}):\n{chunk['chunk_text']}"
        for chunk in chunks
    )


def planner_node(state: GraphState) -> GraphState:
    return _with_node_result(
        state,
        "planner",
        "plan",
        f"Create a short investigation plan for this alert:\n{state['alert']}",
    )


def log_analysis_node(state: GraphState) -> GraphState:
    return _run_log_analysis(state)


def fix_proposer_node(state: GraphState) -> GraphState:
    retrieved_chunks = _retrieve_runbook_chunks(state["diagnosis"] or "")
    updated_state = _with_node_result(
        state,
        "fix_proposer",
        "proposed_fix",
        f"Propose a practical fix based on the alert, plan, diagnosis, and retrieved runbook context below. "
        f"Explicitly cite the relevant runbook source or chunk when using it. If no chunk is a strong match, "
        f"state low confidence and do not fabricate runbook-backed guidance.\n"
        f"Alert: {state['alert']}\nPlan: {state['plan']}\nDiagnosis: {state['diagnosis']}\n"
        f"Retrieved runbook context:\n{_format_retrieved_chunks(retrieved_chunks)}",
    )
    updated_state["retrieved_chunks"] = retrieved_chunks
    return updated_state


def verifier_node(state: GraphState) -> GraphState:
    return _with_node_result(
        state,
        "verifier",
        "trust_score",
        f"Assess confidence in the proposed fix. Respond with ACCEPT, REVIEW, or REJECT "
        f"and brief reasoning.\nAlert: {state['alert']}\nPlan: {state['plan']}\n"
        f"Diagnosis: {state['diagnosis']}\nProposed fix: {state['proposed_fix']}",
    )


graph_builder = StateGraph(GraphState)
graph_builder.add_node("planner_node", planner_node)
graph_builder.add_node("log_analysis_node", log_analysis_node)
graph_builder.add_node("fix_proposer_node", fix_proposer_node)
graph_builder.add_node("verifier_node", verifier_node)
graph_builder.add_edge(START, "planner_node")
graph_builder.add_edge("planner_node", "log_analysis_node")
graph_builder.add_edge("log_analysis_node", "fix_proposer_node")
graph_builder.add_edge("fix_proposer_node", "verifier_node")
graph_builder.add_edge("verifier_node", END)
graph = graph_builder.compile()


if __name__ == "__main__":
    initial_state: GraphState = {
        "alert": "pod CrashLoopBackOff in namespace prod",
        "plan": None,
        "diagnosis": None,
        "proposed_fix": None,
        "trust_score": None,
        "log": [],
        "model_used": {},
        "tool_calls": [],
        "retrieved_chunks": [],
    }
    SIMULATE_RATE_LIMIT_ON_NODE = "log_analysis"
    final_state = graph.invoke(initial_state)
    print("Final state:")
    pprint(final_state, sort_dicts=False, width=120)

    expected_log = ["planner", "log_analysis", "fix_proposer", "verifier"]
    expected_models = {
        "planner": "groq",
        "log_analysis": "groq",
        "fix_proposer": "groq",
        "verifier": "groq",
    }
    try:
        assert final_state["log"] == expected_log
        assert final_state["model_used"] == expected_models
    except AssertionError:
        print("FAIL")
    else:
        print("PASS")
