"""MCP server exposing Kubernetes inspection and safe mutation tools."""

from datetime import datetime, timezone
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


def _validate_namespace_and_name(namespace: str, name: str) -> dict[str, Any] | None:
    if not isinstance(namespace, str) or not namespace.strip():
        return {"error": "invalid namespace", "namespace": namespace, "details": "namespace must be a non-empty string"}
    if not isinstance(name, str) or not name.strip():
        return {"error": "invalid deployment_name", "deployment_name": name, "details": "deployment_name must be a non-empty string"}
    return None


def _deployment_not_found(namespace: str, deployment_name: str) -> dict[str, Any]:
    return {
        "error": "deployment not found",
        "namespace": namespace,
        "deployment_name": deployment_name,
    }


@mcp.tool()
def list_pods(namespace: str) -> list[dict[str, str]] | str:
    """List pod names, UIDs, and current phase in a namespace."""
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    try:
        pods = _core_api.list_namespaced_pod(namespace=namespace).items
        return [
            {
                "name": pod.metadata.name,
                "uid": pod.metadata.uid or "",
                "status": pod.status.phase or "Unknown",
            }
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
        terminated = container_statuses[0].last_state.terminated if container_statuses and container_statuses[0].last_state else None
        return {
            "phase": pod.status.phase or "Unknown",
            "ready_containers": sum(status.ready for status in container_statuses),
            "restart_count": sum(status.restart_count or 0 for status in container_statuses),
            "last_terminated_reason": terminated.reason if terminated else None,
            "last_terminated_exit_code": terminated.exit_code if terminated else None,
        }
    except Exception as error:
        return _api_error(error)


@mcp.tool()
def get_pod_events(namespace: str, pod_name: str) -> list[dict[str, str]] | str:
    """Return Kubernetes events associated with a pod."""
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    try:
        events = _core_api.list_namespaced_event(namespace=namespace).items
        return [
            {
                "reason": event.reason or "",
                "message": event.message or "",
                "type": event.type or "",
            }
            for event in events
            if event.involved_object and event.involved_object.name == pod_name
        ]
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


@mcp.tool()
def restart_deployment(namespace: str, deployment_name: str) -> dict[str, Any] | str:
    """Safely restart a deployment by patching the pod template restart annotation."""
    unavailable = _unavailable()
    if unavailable:
        return unavailable

    invalid = _validate_namespace_and_name(namespace, deployment_name)
    if invalid:
        return invalid

    namespace = namespace.strip()
    deployment_name = deployment_name.strip()

    try:
        _apps_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    except client.exceptions.ApiException as exc:
        if exc.status == 404:
            return _deployment_not_found(namespace, deployment_name)
        return {"error": "read deployment failed", "namespace": namespace, "deployment_name": deployment_name, "status": exc.status, "details": getattr(exc, "reason", str(exc))}

    restarted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": restarted_at,
                    }
                }
            }
        }
    }

    try:
        _apps_api.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=patch,
        )
        return {
            "ok": True,
            "namespace": namespace,
            "deployment_name": deployment_name,
            "restarted_at": restarted_at,
        }
    except client.exceptions.ApiException as exc:
        if exc.status == 404:
            return _deployment_not_found(namespace, deployment_name)
        return {"error": "restart deployment failed", "namespace": namespace, "deployment_name": deployment_name, "status": exc.status, "details": getattr(exc, "reason", str(exc))}


@mcp.tool()
def scale_deployment(namespace: str, deployment_name: str, replicas: int) -> dict[str, Any] | str:
    """Scale a deployment to a bounded, non-negative replica count."""
    unavailable = _unavailable()
    if unavailable:
        return unavailable

    invalid = _validate_namespace_and_name(namespace, deployment_name)
    if invalid:
        return invalid

    namespace = namespace.strip()
    deployment_name = deployment_name.strip()

    if isinstance(replicas, bool):
        return {"error": "invalid replicas", "namespace": namespace, "deployment_name": deployment_name, "details": "replicas must be a non-negative integer"}
    if isinstance(replicas, str):
        try:
            replicas = int(replicas)
        except ValueError:
            return {"error": "invalid replicas", "namespace": namespace, "deployment_name": deployment_name, "details": "replicas must be a non-negative integer"}
    if not isinstance(replicas, int):
        return {"error": "invalid replicas", "namespace": namespace, "deployment_name": deployment_name, "details": "replicas must be a non-negative integer"}
    if replicas < 0:
        return {"error": "invalid replicas", "namespace": namespace, "deployment_name": deployment_name, "details": "replicas must be >= 0"}
    if replicas > 10:
        return {"error": "invalid replicas", "namespace": namespace, "deployment_name": deployment_name, "details": "replicas must be <= 10"}

    try:
        _apps_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    except client.exceptions.ApiException as exc:
        if exc.status == 404:
            return _deployment_not_found(namespace, deployment_name)
        return {"error": "read deployment failed", "namespace": namespace, "deployment_name": deployment_name, "status": exc.status, "details": getattr(exc, "reason", str(exc))}

    patch = {"spec": {"replicas": replicas}}
    try:
        _apps_api.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=patch,
        )
        return {
            "ok": True,
            "namespace": namespace,
            "deployment_name": deployment_name,
            "replicas": replicas,
        }
    except client.exceptions.ApiException as exc:
        if exc.status == 404:
            return _deployment_not_found(namespace, deployment_name)
        return {"error": "scale deployment failed", "namespace": namespace, "deployment_name": deployment_name, "status": exc.status, "details": getattr(exc, "reason", str(exc))}


if __name__ == "__main__":
    mcp.run()