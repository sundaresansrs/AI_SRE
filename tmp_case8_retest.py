import json
import os
import subprocess
import sys
import time

os.environ["AUTO_EXECUTE_ON_ACCEPT"] = ""
sys.stdout.reconfigure(encoding="utf-8")
from src.agents import graph as graph_module
from src.agents.runner import run_alert_and_persist

NAME = "acc-test-8-multi-cause"
NAMESPACE = "default"
ALERT = (
    "Deployment acc-test-8-multi-cause in namespace default is unavailable and its pod is unhealthy. "
    "Investigate the real Kubernetes state and determine all contributing causes."
)


def kubectl(*args):
    result = subprocess.run(["kubectl", *args], text=True, capture_output=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def pod_evidence():
    pods = kubectl("get", "pods", "-n", NAMESPACE, "-l", "app=" + NAME, "-o", "json")
    try:
        items = json.loads(pods["stdout"]).get("items", [])
    except json.JSONDecodeError:
        items = []
    evidence = []
    for pod in items:
        pod_name = pod["metadata"]["name"]
        events = kubectl(
            "get",
            "events",
            "-n",
            NAMESPACE,
            "--field-selector",
            "involvedObject.name=" + pod_name,
            "-o",
            "json",
        )
        evidence.append(
            {
                "name": pod_name,
                "phase": pod.get("status", {}).get("phase"),
                "container_statuses": pod.get("status", {}).get("containerStatuses", []),
                "events": events,
            }
        )
    return evidence


original = graph_module._groq_completion
headers = []


def traced_completion(messages, tools=None, tool_choice=None):
    args = {"model": graph_module.GROQ_MODEL, "messages": messages, "temperature": 0}
    if tools:
        args["tools"] = tools
        args["tool_choice"] = tool_choice or "auto"
    raw = graph_module.groq_client.chat.completions.with_raw_response.create(**args)
    response_headers = dict(raw.headers)
    headers.append(response_headers)
    print(
        "GROQ_HEADERS:",
        json.dumps(
            {key: value for key, value in response_headers.items() if key.startswith("x-ratelimit-")},
            sort_keys=True,
        ),
    )
    return raw.parse()


def ground_truth():
    return {
        "deployment": kubectl("get", "deployment", NAME, "-n", NAMESPACE, "-o", "wide"),
        "pods": kubectl("get", "pods", "-n", NAMESPACE, "-l", "app=" + NAME, "-o", "wide"),
        "pod_evidence": pod_evidence(),
    }


try:
    print("GROUND_TRUTH_BEFORE:", json.dumps(ground_truth(), default=str))
    graph_module._groq_completion = traced_completion
    started = time.perf_counter()
    state, incident = run_alert_and_persist(ALERT)
    elapsed = time.perf_counter() - started
    print(
        "RESULT:",
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "model_used": state.get("model_used"),
                "classification": state.get("classification"),
                "trust_score": state.get("trust_score"),
                "corrections": [
                    correction
                    for call in state.get("tool_calls", [])
                    for correction in call.get("corrections", [])
                ],
                "tool_order": [call.get("tool") for call in state.get("tool_calls", [])],
                "tool_calls": state.get("tool_calls"),
                "persisted": {
                    key: incident.get(key)
                    for key in ["id", "classification", "trust_score", "executed"]
                },
                "last_groq_headers": {
                    key: value
                    for key, value in (headers[-1].items() if headers else [])
                    if key.startswith("x-ratelimit-")
                },
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
    )
    print("DIAGNOSIS_START")
    print(state.get("diagnosis"))
    print("DIAGNOSIS_END")
finally:
    graph_module._groq_completion = original
    print("GROUND_TRUTH_AFTER:", json.dumps(ground_truth(), default=str))