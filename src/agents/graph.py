"""LangGraph orchestration for the AI-SRE agent pipeline."""

import os
import json
import re
from contextlib import AsyncExitStack
from pprint import pprint
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

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
# Empirical calibration from the real corpus:
# - self-retrieval of an identical chunk returned 1.000 on the live Qdrant collection
# - the real kube-system/CoreDNS diagnostic only reached ~0.528 against the same corpus
# The strong threshold is therefore set well above the weak floor but below the self-match ceiling.
RAG_WEAK_THRESHOLD = 0.55
RAG_STRONG_THRESHOLD = 0.80
_rag_model: SentenceTransformer | None = None
ANOMALY_THRESHOLD = 0.588
ANOMALY_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lightgbm_final_classifier.joblib"
_anomaly_model = joblib.load(ANOMALY_MODEL_PATH)


VALID_RECOMMENDED_ACTIONS = ("restart_deployment", "scale_deployment", "none")


class GraphState(TypedDict):
    alert: str
    plan: str | None
    diagnosis: str | None
    proposed_fix: str | None
    recommended_action: Literal["restart_deployment", "scale_deployment", "none"] | None
    action_namespace: str | None
    action_deployment_name: str | None
    action_replicas: int | None
    trust_score: float | None
    classification: Literal["ACCEPT", "REVIEW", "REJECT"] | None
    log: list[str]
    model_used: dict[str, str]
    tool_calls: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    diagnosis_degraded: bool


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


def _groq_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> Any:
    arguments: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        arguments["tools"] = tools
        arguments["tool_choice"] = tool_choice or "auto"
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
        return _gemini_fallback_text(prompt), "gemini"


def _gemini_fallback_text(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


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


def _degraded_fallback_diagnosis(text: str) -> str:
    return (
        "INVESTIGATION INCOMPLETE - Groq rate-limited during Log-Analysis and Gemini fallback had no MCP tool access. "
        "The following text is unverified and must not be treated as a tool-grounded diagnosis:\n\n"
        f"{text}"
    )


def _run_degraded_gemini_fallback(prompt: str) -> str:
    try:
        text = _gemini_fallback_text(prompt)
    except Exception as error:
        text = f"Gemini fallback was unavailable: {type(error).__name__}. No diagnosis was produced."
    return _degraded_fallback_diagnosis(text)


def _alert_resource_context(alert: str, plan: str | None) -> dict[str, str | None]:
    text = "\n".join(value for value in (alert, plan or "") if value)
    namespace_match = re.search(
        r"\b(?:in|namespace)\s+namespace\s+([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)|"
        r"\bnamespace\s*[:=]?\s*([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)",
        text,
        re.IGNORECASE,
    )
    deployment_match = re.search(
        r"\bdeployment(?:\.apps)?[ /:`'\"]+([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)",
        text,
        re.IGNORECASE,
    )
    return {
        "namespace": next((group for group in namespace_match.groups() if group), None) if namespace_match else None,
        "deployment_name": deployment_match.group(1) if deployment_match else None,
    }


def _correct_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    expected: dict[str, str | None],
) -> tuple[dict[str, Any], list[str]]:
    corrected = dict(arguments)
    corrections: list[str] = []
    for field in ("namespace", "deployment_name"):
        expected_value = expected.get(field)
        actual_value = corrected.get(field)
        if expected_value and field in corrected and actual_value != expected_value:
            corrections.append(
                f"Corrected hallucinated {field} {actual_value!r} -> {expected_value!r} based on alert"
            )
            corrected[field] = expected_value
    return corrected, corrections


async def _run_log_analysis_with_tools(state: GraphState) -> tuple[str, str, list[dict[str, Any]]]:
    stack, sessions = await _open_mcp_servers()
    try:
        tools, owners = await _fetch_mcp_tools(sessions)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an SRE log analyst. Always begin the investigation by using the available MCP tools "
                    "before considering whether more information is needed. The namespace and deployment name "
                    "stated in the alert are authoritative; never guess or derive them from partial name matching. "
                    "Use the available MCP tools to inspect real data before diagnosing. "
                    "For Pending pods or scheduling failures, use get_pod_events to inspect the actual Kubernetes event reason. "
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
        corrections: list[str] = []
        expected_resources = _alert_resource_context(state["alert"], state.get("plan"))
        available_tool_names = ", ".join(sorted(owners))
        validation_retries = 0
        tool_rounds = 0
        while tool_rounds < 5:
            tool_rounds += 1
            try:
                response = _groq_completion(
                    messages,
                    tools,
                    tool_choice="required" if not tool_calls else "auto",
                )
            except RateLimitError:
                fallback_prompt = (
                    "Provide the final SRE diagnosis using the investigation conversation below. "
                    "Ground it only in the available tool results and do not request another tool.\n\n"
                    f"{json.dumps(messages, default=str)}"
                )
                return _run_degraded_gemini_fallback(fallback_prompt), "gemini", tool_calls
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
                try:
                    final_response = _groq_completion(messages)
                    return (final_response.choices[0].message.content or "").strip(), "groq", tool_calls
                except RateLimitError:
                    fallback_prompt = (
                        "Provide the final SRE diagnosis using the investigation conversation below. "
                        "Ground it only in the available tool results and do not request another tool.\n\n"
                        f"{json.dumps(messages, default=str)}"
                    )
                    return _run_degraded_gemini_fallback(fallback_prompt), "gemini", tool_calls
            messages.append({
                "role": "assistant",
                "content": assistant.content or "",
                "tool_calls": [call.model_dump() for call in requested],
            })
            for call in requested:
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                arguments, argument_corrections = _correct_tool_arguments(
                    call.function.name,
                    arguments,
                    expected_resources,
                )
                corrections.extend(argument_corrections)
                result = await owners[call.function.name].call_tool(call.function.name, arguments)
                raw_result = _tool_result_value(result)
                tool_call = {"tool": call.function.name, "args": arguments, "result": raw_result}
                if argument_corrections:
                    tool_call["corrections"] = argument_corrections
                tool_calls.append(tool_call)
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
        try:
            final_response = _groq_completion(messages, tools, tool_choice="none")
            return (final_response.choices[0].message.content or "").strip(), "groq", tool_calls
        except RateLimitError:
            fallback_prompt = (
                "Provide the final SRE diagnosis using the investigation conversation below. "
                "Ground it only in the available tool results and do not request another tool.\n\n"
                f"{json.dumps(messages, default=str)}"
            )
            return _run_degraded_gemini_fallback(fallback_prompt), "gemini", tool_calls
    finally:
        await _close_mcp_servers(stack, sessions)


def _run_log_analysis(state: GraphState) -> GraphState:
    diagnosis, provider, tool_calls = anyio.run(_run_log_analysis_with_tools, state)
    updated_state = dict(state)
    corrections = [
        correction
        for call in tool_calls
        for correction in call.get("corrections", [])
    ]
    updated_state["log"] = [*state["log"], "log_analysis", *corrections]
    updated_state["model_used"] = {**state["model_used"], "log_analysis": provider}
    updated_state["tool_calls"] = [*state["tool_calls"], *tool_calls]
    updated_state["diagnosis"] = diagnosis
    updated_state["diagnosis_degraded"] = provider == "gemini"
    return updated_state


def _with_node_result(state: GraphState, node_name: str, field: str, prompt: str) -> GraphState:
    text, provider = call_llm_with_fallback(prompt, node_name)
    updated_state = dict(state)
    updated_state["log"] = [*state["log"], node_name]
    updated_state["model_used"] = {**state["model_used"], node_name: provider}
    updated_state[field] = text
    return updated_state


def _collect_names_from_result(result: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(result, dict):
        if isinstance(result.get("name"), str) and result["name"].strip():
            names.add(result["name"].strip())
        for value in result.values():
            names.update(_collect_names_from_result(value))
    elif isinstance(result, list):
        for item in result:
            names.update(_collect_names_from_result(item))
    elif isinstance(result, tuple):
        for item in result:
            names.update(_collect_names_from_result(item))
    return names


def _observed_deployment_context(tool_calls: list[dict[str, Any]]) -> tuple[set[str], dict[str, str]]:
    observed_names: set[str] = set()
    observed_namespaces: dict[str, str] = {}
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        if call.get("tool") != "list_deployments":
            continue
        args = call.get("args") or {}
        namespace = args.get("namespace") if isinstance(args, dict) else None
        deployment_names = _collect_names_from_result(call.get("result"))
        if not deployment_names:
            continue
        observed_names.update(deployment_names)
        for deployment_name in deployment_names:
            if namespace and isinstance(namespace, str):
                observed_namespaces[deployment_name] = namespace
    return observed_names, observed_namespaces


def validate_recommended_action(
    payload: dict[str, Any] | None,
    tool_calls: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    log: list[str] = []
    cleaned: dict[str, Any] = {
        "recommended_action": "none",
        "action_namespace": None,
        "action_deployment_name": None,
        "action_replicas": None,
    }
    payload = payload or {}

    action = payload.get("recommended_action")
    if action not in VALID_RECOMMENDED_ACTIONS:
        cleaned["recommended_action"] = "none"
        log.append(
            f"Validation: invalid recommended_action={action!r}; forced to none because it is not in the whitelist."
        )
        return cleaned, log

    cleaned["recommended_action"] = action

    namespace = payload.get("action_namespace")
    if isinstance(namespace, str) and namespace.strip():
        cleaned["action_namespace"] = namespace.strip()

    deployment_name = payload.get("action_deployment_name")
    if isinstance(deployment_name, str) and deployment_name.strip():
        cleaned["action_deployment_name"] = deployment_name.strip()

    replicas = payload.get("action_replicas")
    if cleaned["recommended_action"] == "scale_deployment":
        if isinstance(replicas, bool):
            cleaned["action_replicas"] = None
        elif replicas is None:
            cleaned["action_replicas"] = None
        else:
            try:
                cleaned["action_replicas"] = int(replicas)
            except (TypeError, ValueError):
                cleaned["action_replicas"] = None
                log.append(
                    "Validation: scale_deployment action_replicas is not a valid integer; it was cleared."
                )
    else:
        cleaned["action_replicas"] = None

    observed_names, observed_namespaces = _observed_deployment_context(tool_calls)
    if cleaned["recommended_action"] in {"restart_deployment", "scale_deployment"}:
        observed_namespace = observed_namespaces.get(cleaned["action_deployment_name"]) if cleaned["action_deployment_name"] else None
        if cleaned["action_deployment_name"] is None or cleaned["action_deployment_name"] not in observed_names:
            log.append(
                f"Validation: recommended action '{cleaned['recommended_action']}' targets deployment "
                f"'{cleaned['action_deployment_name']}' but that deployment was not observed as a real tool result; "
                "forced to none."
            )
            cleaned["recommended_action"] = "none"
            cleaned["action_namespace"] = None
            cleaned["action_deployment_name"] = None
            cleaned["action_replicas"] = None
            return cleaned, log
        if cleaned["action_namespace"] is None and observed_namespace:
            cleaned["action_namespace"] = observed_namespace
        elif cleaned["action_namespace"] and observed_namespace and cleaned["action_namespace"] != observed_namespace:
            log.append(
                f"Validation: action_namespace='{cleaned['action_namespace']}' does not match the observed namespace "
                f"'{observed_namespace}' for deployment '{cleaned['action_deployment_name']}' and was corrected."
            )
            cleaned["action_namespace"] = observed_namespace

    if cleaned["recommended_action"] == "none":
        cleaned["action_namespace"] = None
        cleaned["action_deployment_name"] = None
        cleaned["action_replicas"] = None

    return cleaned, log


def _structured_action_schema() -> dict[str, Any]:
    return {
        "name": "sre_action_recommendation",
        "schema": {
            "type": "object",
            "properties": {
                "recommended_action": {
                    "type": "string",
                    "enum": list(VALID_RECOMMENDED_ACTIONS),
                },
                "action_namespace": {"type": ["string", "null"]},
                "action_deployment_name": {"type": ["string", "null"]},
                "action_replicas": {"type": ["integer", "null"]},
            },
            "required": [
                "recommended_action",
                "action_namespace",
                "action_deployment_name",
                "action_replicas",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _llm_structured_action_recommendation(state: GraphState) -> dict[str, Any]:
    prompt = (
        "Choose the single most appropriate Kubernetes action for the diagnosis. "
        "Select exactly one of: restart_deployment, scale_deployment, none. "
        "Use 'none' if no safe automatable action applies. "
        "If you choose restart_deployment or scale_deployment, the deployment name must be a deployment that was actually observed in the real tool_calls results from this run. "
        "Do not invent or infer deployments that were not seen. "
        "If a deployment is safe and observed, set action_namespace to its namespace and action_deployment_name to the exact observed deployment name. "
        "When the action is scale_deployment, set action_replicas to an integer value; otherwise use null.\n\n"
        f"Alert: {state['alert']}\n"
        f"Plan: {state['plan']}\n"
        f"Diagnosis: {state['diagnosis']}\n"
        f"Observed tool_calls: {json.dumps(state['tool_calls'], default=str)}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an SRE planner. Return JSON matching the requested schema exactly. "
                "Use only real observed deployment names from the provided tool_calls. "
                "Never invent a deployment or namespace."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": _structured_action_schema()},
        )
        content = response.choices[0].message.content or "{}"
        structured = json.loads(content)
        if isinstance(structured, dict):
            return structured
    except Exception:
        pass
    return {
        "recommended_action": "none",
        "action_namespace": None,
        "action_deployment_name": None,
        "action_replicas": None,
    }


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

    structured_action = _llm_structured_action_recommendation(updated_state)
    validated_action, validation_log = validate_recommended_action(structured_action, updated_state["tool_calls"])
    updated_state["recommended_action"] = validated_action["recommended_action"]
    updated_state["action_namespace"] = validated_action["action_namespace"]
    updated_state["action_deployment_name"] = validated_action["action_deployment_name"]
    updated_state["action_replicas"] = validated_action["action_replicas"]
    updated_state["log"] = [*updated_state["log"], *validation_log]
    return updated_state


def _max_rag_score(retrieved_chunks: list[dict[str, Any]]) -> float:
    if not retrieved_chunks:
        return 0.0
    scores = []
    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue
        try:
            score = float(chunk.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        scores.append(score)
    return max(scores, default=0.0)


def _has_real_tool_evidence(tool_calls: list[dict[str, Any]]) -> bool:
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if result is None:
            continue
        if isinstance(result, str):
            if result.strip():
                return True
            continue
        if isinstance(result, (list, tuple, dict)):
            if len(result) > 0:
                return True
            continue
        return True
    return False


def classify_verifier_state(state: GraphState) -> tuple[float, str]:
    rag_score = _max_rag_score(state.get("retrieved_chunks", []))
    tool_evidence = _has_real_tool_evidence(state.get("tool_calls", []))

    if state.get("diagnosis_degraded"):
        return rag_score, "REVIEW" if (tool_evidence or rag_score > 0.0) else "REJECT"

    if (not tool_evidence) and rag_score == 0.0:
        return rag_score, "REJECT"
    if tool_evidence and rag_score >= RAG_STRONG_THRESHOLD:
        return rag_score, "ACCEPT"
    return rag_score, "REVIEW"


def verifier_node(state: GraphState) -> GraphState:
    rag_score, classification = classify_verifier_state(state)
    explanation, provider = call_llm_with_fallback(
        f"Assess confidence in the proposed fix. Respond with ACCEPT, REVIEW, or REJECT "
        f"and brief reasoning.\nAlert: {state['alert']}\nPlan: {state['plan']}\n"
        f"Diagnosis: {state['diagnosis']}\nProposed fix: {state['proposed_fix']}\n"
        f"Measured evidence: rag_score={rag_score:.6f}, tool_evidence_present={_has_real_tool_evidence(state.get('tool_calls', []))}, "
        f"rag_weak_threshold={RAG_WEAK_THRESHOLD:.2f}, rag_strong_threshold={RAG_STRONG_THRESHOLD:.2f}",
        "verifier",
    )
    updated_state = dict(state)
    updated_state["log"] = [*state["log"], "verifier"]
    updated_state["model_used"] = {**state["model_used"], "verifier": provider}
    updated_state["trust_score"] = rag_score
    updated_state["classification"] = classification
    updated_state["verifier_reasoning"] = explanation
    return updated_state


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
    def _demo_case(name: str, state: GraphState) -> None:
        result = verifier_node(state)
        print(f"{name}: rag_score={result['trust_score']:.6f} | classification={result['classification']} | "
              f"thresholds=weak:{RAG_WEAK_THRESHOLD:.2f}, strong:{RAG_STRONG_THRESHOLD:.2f}")

    demo_review_state = {
        "alert": "Pod coredns-589f44dc88-jz8ft in kube-system is CrashLoopBackOff",
        "plan": "Inspect coredns pod status in kube-system and review logs.",
        "diagnosis": (
            "The coredns-589f44dc88-jz8ft pod in kube-system is Running with Restart Count: 5. "
            "Logs show [INFO] plugin/kubernetes: waiting for Kubernetes API before starting server "
            "and [WARNING] plugin/kubernetes: starting server with unsynced Kubernetes API."
        ),
        "proposed_fix": "Restart the coredns deployment and verify the DNS plugin has time to warm up.",
        "recommended_action": None,
        "action_namespace": None,
        "action_deployment_name": None,
        "action_replicas": None,
        "trust_score": None,
        "classification": None,
        "log": [],
        "model_used": {},
        "tool_calls": [
            {
                "tool": "list_pods",
                "args": {"namespace": "kube-system"},
                "result": [{"name": "coredns-589f44dc88-jz8ft", "status": "Running", "restart_count": 5}],
            }
        ],
        "retrieved_chunks": [
            {
                "score": 0.52840364,
                "chunk_id": "2025_github_k8s-upgrade-kube-proxy-iptables_chunk0",
                "source_file": "2025_github_k8s-upgrade-kube-proxy-iptables.txt",
                "chunk_text": "Kubernetes networking upgrade left kube-proxy and kubelet out of sync.",
            }
        ],
    }
    demo_reject_state = {
        "alert": "Namespace prod has an unrelated application issue.",
        "plan": "Check workload status.",
        "diagnosis": "The issue is unrelated to DNS or Kubernetes control-plane startup; there are no relevant logs.",
        "proposed_fix": "No fix proposed because evidence is absent.",
        "recommended_action": None,
        "action_namespace": None,
        "action_deployment_name": None,
        "action_replicas": None,
        "trust_score": None,
        "classification": None,
        "log": [],
        "model_used": {},
        "tool_calls": [
            {"tool": "list_pods", "args": {"namespace": "prod"}, "result": []},
            {"tool": "list_deployments", "args": {"namespace": "prod"}, "result": []},
        ],
        "retrieved_chunks": [],
    }
    demo_accept_state = {
        "alert": "Global network outage caused by multicast switch misconfiguration.",
        "plan": "Review network configuration and switch health.",
        "diagnosis": (
            "https://web.archive.org/web/20211201033341/https://codeascraft.com/2012/01/23/solr-bittorrent-index-replication/ "
            "Title: Etsy: Sending multicast traffic without properly configured switches causes a global outage. "
            "Category: Networking / Config Error."
        ),
        "proposed_fix": "Fix switch multicast configuration and verify the route policy is consistent.",
        "recommended_action": None,
        "action_namespace": None,
        "action_deployment_name": None,
        "action_replicas": None,
        "trust_score": None,
        "classification": None,
        "log": [],
        "model_used": {},
        "tool_calls": [
            {
                "tool": "get_switch_health",
                "args": {"site": "global"},
                "result": [{"switch": "core-1", "status": "misconfigured", "reason": "multicast routing mismatch"}],
            }
        ],
        "retrieved_chunks": [
            {
                "score": 1.0,
                "chunk_id": "2012_etsy_multicast-switch-misconfig-outage_chunk0",
                "source_file": "2012_etsy_multicast-switch-misconfig-outage.txt",
                "chunk_text": "https://web.archive.org/web/20211201033341/https://codeascraft.com/2012/01/23/solr-bittorrent-index-replication/ Title: Etsy: Sending multicast traffic without properly configured switches causes a global outage. Category: Networking / Config Error.",
            }
        ],
    }

    print(f"RAG thresholds: weak={RAG_WEAK_THRESHOLD:.2f}, strong={RAG_STRONG_THRESHOLD:.2f}")
    _demo_case("REVIEW", demo_review_state)
    _demo_case("REJECT", demo_reject_state)
    _demo_case("ACCEPT", demo_accept_state)
