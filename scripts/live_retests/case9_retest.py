import json
import os
import subprocess
import sys
import time

os.environ["AUTO_EXECUTE_ON_ACCEPT"] = ""
sys.stdout.reconfigure(encoding="utf-8")
from src.agents import graph as graph_module
from src.agents.runner import run_alert_and_persist

NAME = "acc-test-9-dns-migration"
NAMESPACE = "default"
ALERT = (
    "Deployment acc-test-9-dns-migration in namespace default is unavailable because its application cannot resolve "
    "kubernetes.default.svc.cluster.local after a DNS configuration migration. Investigate the real Kubernetes state "
    "and determine the cause."
)


def kubectl(*args):
    result = subprocess.run(["kubectl", *args], text=True, capture_output=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


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
        "logs": kubectl("logs", "-n", NAMESPACE, "-l", "app=" + NAME, "--tail=20"),
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
                "tool_order": [call.get("tool") for call in state.get("tool_calls", [])],
                "tool_calls": state.get("tool_calls"),
                "retrieved_chunks": state.get("retrieved_chunks"),
                "proposed_fix": state.get("proposed_fix"),
                "recommended_action": state.get("recommended_action"),
                "action_namespace": state.get("action_namespace"),
                "action_deployment_name": state.get("action_deployment_name"),
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