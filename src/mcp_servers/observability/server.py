"""MCP server exposing read-only Prometheus observability tools."""

import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
REQUEST_TIMEOUT = 10
mcp = FastMCP("observability")


def _prometheus_get(endpoint: str, **params: str) -> dict[str, Any]:
    response = requests.get(
        f"{PROMETHEUS_URL}{endpoint}",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        error = payload.get("error", "Unknown Prometheus API error")
        error_type = payload.get("errorType")
        detail = f" ({error_type})" if error_type else ""
        raise RuntimeError(f"Prometheus API error{detail}: {error}")
    return payload["data"]


def _prometheus_error(error: Exception) -> str:
    if isinstance(error, requests.RequestException):
        return f"Prometheus connection error: {error}"
    return f"Prometheus error: {error}"


@mcp.tool()
def instant_query(promql: str) -> list[dict[str, Any]] | str:
    """Execute an instant PromQL query and return its result list."""
    try:
        return _prometheus_get("/api/v1/query", query=promql)["result"]
    except Exception as error:
        return _prometheus_error(error)


@mcp.tool()
def range_query(promql: str, start: str, end: str, step: str) -> list[dict[str, Any]] | str:
    """Execute a range PromQL query and return its result list."""
    try:
        return _prometheus_get(
            "/api/v1/query_range",
            query=promql,
            start=start,
            end=end,
            step=step,
        )["result"]
    except Exception as error:
        return _prometheus_error(error)


@mcp.tool()
def list_targets() -> list[dict[str, Any]] | str:
    """List active Prometheus scrape targets with job, instance, and health."""
    try:
        targets = _prometheus_get("/api/v1/targets")["activeTargets"]
        return [
            {
                "job": target.get("labels", {}).get("job"),
                "instance": target.get("labels", {}).get("instance"),
                "health": target.get("health"),
            }
            for target in targets
        ]
    except Exception as error:
        return _prometheus_error(error)


@mcp.tool()
def list_alerts() -> list[dict[str, Any]] | str:
    """List currently active Prometheus alerts."""
    try:
        return _prometheus_get("/api/v1/alerts")["alerts"]
    except Exception as error:
        return _prometheus_error(error)


if __name__ == "__main__":
    mcp.run()
