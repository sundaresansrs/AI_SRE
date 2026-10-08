"""Run the full agent pipeline on one alert and print what each stage produced.

Usage (from the repository root, with the lab cluster and Qdrant running and .env loaded):

    python scripts/analyze_alert.py "Deployment lab-oom-killed in namespace sre-lab keeps restarting"
    python scripts/analyze_alert.py --persist "..."      # also save the incident for the dashboard

Without --persist nothing is written to the incident database.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")


def _short(value, limit=220):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("alert", help="Alert text; name the namespace and deployment in it.")
    parser.add_argument("--persist", action="store_true", help="Save the incident to the API database.")
    parser.add_argument("--json", action="store_true", help="Print the final state as JSON instead of a summary.")
    args = parser.parse_args()

    # Imported late so --help works without API keys.
    from src.agents.graph import graph
    from src.agents.runner import _initial_graph_state, run_alert_and_persist

    started = time.perf_counter()
    if args.persist:
        state, incident = run_alert_and_persist(args.alert)
    else:
        state, incident = graph.invoke(_initial_graph_state(args.alert)), None
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(state, indent=2, default=str))
        return 0

    print(f"\nALERT      {args.alert}")
    print(f"STAGES     {' > '.join(entry for entry in state['log'] if entry in ('planner', 'log_analysis', 'fix_proposer', 'verifier'))}")
    print(f"MODELS     {state['model_used']}")
    print(f"\nPLAN       {_short(state['plan'], 400)}")
    print(f"\nTOOL CALLS ({len(state['tool_calls'])})")
    for call in state["tool_calls"]:
        flag = "ERROR " if call.get("is_error") else ""
        print(f"  - {flag}{call['tool']}({_short(call['args'], 90)}) -> {_short(call['result'], 130)}")
    print(f"\nDIAGNOSIS  {_short(state['diagnosis'], 700)}")
    print("\nRAG MATCHES")
    for chunk in state["retrieved_chunks"] or []:
        print(f"  - {chunk['score']:.3f}  {chunk['source_file']}")
    print(f"\nPROPOSED   {_short(state['proposed_fix'], 600)}")
    print(
        f"\nACTION     {state['recommended_action']} "
        f"{state['action_namespace']}/{state['action_deployment_name']} replicas={state['action_replicas']}"
    )
    print(f"VERDICT    {state['classification']} (trust_score={state['trust_score']:.3f}, degraded={state['diagnosis_degraded']})")
    print(f"REASONING  {_short(state['verifier_reasoning'], 400)}")
    validation = [entry for entry in state["log"] if entry.startswith(("Validation", "Corrected"))]
    for entry in validation:
        print(f"GUARDRAIL  {_short(entry, 260)}")
    print(f"\nelapsed {elapsed:.1f}s" + (f" | saved as incident #{incident['id']}" if incident else " | not saved"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
