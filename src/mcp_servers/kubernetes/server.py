"""MCP server exposing read-only Kubernetes inspection tools."""

from typing import Any

from kubernetes import client, config
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("kubernetes")

_core_api: client.CoreV1Api | None = None
_apps_api: client.AppsV1Api | None = None
_config_error: str | None = None

try:
    config.load_kube_config()
    _core_api = client.CoreV1Api()
    _apps_api = client.AppsV1Api()
except Exception as error:
    _config_error = f"Unable to load the current kubeconfig context: {error}"


def _api_error(error: Exception) -> str:
    status = getattr(error, "status", None)
    reason = getattr(error, "reason", None)
    detail = reason or str(error)
    if status:
        return f"Kubernetes API error ({status}): {detail}"
    return f"Kubernetes error: {detail}"


def _unavailable() -> str | None:
    if _config_error:
        return _config_error
    if _core_api is None or _apps_api is None:
        return "Kubernetes APIs are unavailable because kubeconfig was not loaded."
    return None


@mcp.tool()
def list_pods(namespace: str) -> list[dict[str, str]] | str:
    """List pod names and their current phase in a namespace."""
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    try:
        pods = _core_api.list_namespaced_pod(namespace=namespace).items
        return [
            {"name": pod.metadata.name, "status": pod.status.phase or "Unknown"}
            for pod in pods
        ]
    except Exception as error:
        return _api_error(error)


@mcp.tool()
def get_pod_logs(namespace: str, pod_name: str) -> str:
    """Return approximately the last 50 lines of a pod's logs."""
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    try:
        return _core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            tail_lines=50,
        )
    except Exception as error:
        return _api_error(error)


@mcp.tool()
def get_pod_status(namespace: str, pod_name: str) -> dict[str, Any] | str:
    """Return phase, ready-container count, and restart count for a pod."""
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    try:
        pod = _core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
        container_statuses = pod.status.container_statuses or []
        return {
            "phase": pod.status.phase or "Unknown",
            "ready_containers": sum(status.ready for status in container_statuses),
            "restart_count": sum(status.restart_count or 0 for status in container_statuses),
        }
    except Exception as error:
        return _api_error(error)


@mcp.tool()
def list_deployments(namespace: str) -> list[dict[str, Any]] | str:
    """List deployment names with desired and available replica counts."""
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    try:
        deployments = _apps_api.list_namespaced_deployment(namespace=namespace).items
        return [
            {
                "name": deployment.metadata.name,
                "desired_replicas": deployment.spec.replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
            }
            for deployment in deployments
        ]
    except Exception as error:
        return _api_error(error)


if __name__ == "__main__":
    mcp.run()