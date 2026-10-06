import json
import os
import subprocess
import sys
import time

os.environ["AUTO_EXECUTE_ON_ACCEPT"] = ""
sys.stdout.reconfigure(encoding="utf-8")
from src.agents import graph as graph_module
from src.agents.runner import run_alert_and_persist

NAME = "acc-test-6-not-found"
ALERT = "Deployment acc-test-6-not-found in namespace default is unavailable and its workload cannot be located. Investigate the real cluster state."

def kubectl(*args):
    p = subprocess.run(["kubectl", *args], text=True, capture_output=True)
    return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}

original = graph_module._groq_completion
headers = []
def traced_completion(messages, tools=None, tool_choice=None):
    args = {"model": graph_module.GROQ_MODEL, "messages": messages, "temperature": 0}
    if tools:
        args["tools"] = tools
        args["tool_choice"] = tool_choice or "auto"
    raw = graph_module.groq_client.chat.completions.with_raw_response.create(**args)
    h = dict(raw.headers)
    headers.append(h)
    print("GROQ_HEADERS:", json.dumps({k: v for k, v in h.items() if k.startswith("x-ratelimit-")}, sort_keys=True))
    return raw.parse()

try:
    print("GROUND_TRUTH_BEFORE:", json.dumps({
        "deployment": kubectl("get", "deployment", NAME, "-n", "default", "--ignore-not-found=true"),
        "pods": kubectl("get", "pods", "-n", "default", "-l", "app=" + NAME, "--ignore-not-found=true"),
    }))
    graph_module._groq_completion = traced_completion
    started = time.perf_counter()
    state, incident = run_alert_and_persist(ALERT)
    elapsed = time.perf_counter() - started
    print("RESULT:", json.dumps({
        "elapsed_seconds": elapsed,
        "model_used": state.get("model_used"),
        "classification": state.get("classification"),
        "trust_score": state.get("trust_score"),
        "corrections": [c for call in state.get("tool_calls", []) for c in call.get("corrections", [])],
        "tool_order": [call.get("tool") for call in state.get("tool_calls", [])],
        "tool_calls": state.get("tool_calls"),
        "persisted": {k: incident.get(k) for k in ["id", "classification", "trust_score", "executed"]},
        "last_groq_headers": {k: v for k, v in (headers[-1].items() if headers else []) if k.startswith("x-ratelimit-")},
    }, indent=2, ensure_ascii=False, default=str))
    print("DIAGNOSIS_START")
    print(state.get("diagnosis"))
    print("DIAGNOSIS_END")
finally:
    graph_module._groq_completion = original
    print("GROUND_TRUTH_AFTER:", json.dumps({
        "deployment": kubectl("get", "deployment", NAME, "-n", "default", "--ignore-not-found=true"),
        "pods": kubectl("get", "pods", "-n", "default", "-l", "app=" + NAME, "--ignore-not-found=true"),
    }))
