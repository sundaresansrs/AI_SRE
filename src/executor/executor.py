"""Execute validated Kubernetes actions through the Kubernetes MCP server."""

import asyncio
import json
import logging
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


VALID_ACTIONS = {"restart_deployment", "scale_deployment", "none"}


def _server_parameters() -> StdioServerParameters:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    server_path = os.path.join(root, "src", "mcp_servers", "kubernetes", "server.py")
    return StdioServerParameters(
        command=sys.executable,
        args=[server_path],
        env=os.environ.copy(),
        cwd=root,
    )


def _tool_result_value(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured and "result" in structured:
        return structured["result"]
    return [getattr(block, "text", str(block)) for block in getattr(result, "content", [])]


async def _call_kubernetes_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    async with stdio_client(_server_parameters()) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _tool_result_value(result)


def _missing_fields(action: str, state: dict[str, Any]) -> list[str]:
    required = ["action_namespace", "action_deployment_name"]
    if action == "scale_deployment":
        required.append("action_replicas")
    return [field for field in required if state.get(field) is None]


def execute_recommended_action(state: dict[str, Any]) -> dict[str, Any]:
    """Execute one validated recommendation through the Kubernetes MCP server."""
    action = state.get("recommended_action")

    if action == "none":
        reason = "recommended_action was none"
        logger.warning("No MCP tool call: %s", reason)
        return {"executed": False, "reason": reason}

    if action not in VALID_ACTIONS:
        reason = f"unrecognized recommended_action: {action}"
        logger.warning("No MCP tool call: %s", reason)
        return {"executed": False, "reason": reason}

    missing = _missing_fields(action, state)
    if missing:
        reason = f"missing required fields for {action}: {', '.join(missing)}"
        logger.warning("No MCP tool call: %s", reason)
        return {"executed": False, "reason": reason}

    namespace = state["action_namespace"]
    deployment_name = state["action_deployment_name"]
    if not isinstance(namespace, str) or not namespace.strip():
        reason = "action_namespace must be a non-empty string"
        logger.warning("No MCP tool call: %s", reason)
        return {"executed": False, "reason": reason}
    if not isinstance(deployment_name, str) or not deployment_name.strip():
        reason = "action_deployment_name must be a non-empty string"
        logger.warning("No MCP tool call: %s", reason)
        return {"executed": False, "reason": reason}

    arguments: dict[str, Any] = {
        "namespace": namespace,
        "deployment_name": deployment_name,
    }
    if action == "scale_deployment":
        replicas = state["action_replicas"]
        if isinstance(replicas, bool) or not isinstance(replicas, int):
            reason = "action_replicas must be an integer for scale_deployment"
            logger.warning("No MCP tool call: %s", reason)
            return {"executed": False, "reason": reason}
        arguments["replicas"] = replicas

    logger.info("Calling Kubernetes MCP tool %s with %s", action, json.dumps(arguments, default=str))
    tool_result = asyncio.run(_call_kubernetes_tool(action, arguments))
    result = dict(tool_result) if isinstance(tool_result, dict) else {"result": tool_result}
    result.update({
        "executed": True,
        "action": action,
        "namespace": namespace,
        "deployment_name": deployment_name,
    })
    return result
