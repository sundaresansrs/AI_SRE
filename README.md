# AI-SRE

An agentic incident-response assistant for Kubernetes. Given an alert, a LangGraph pipeline plans an
investigation, inspects the live cluster through MCP tool servers, retrieves similar postmortems from a
Qdrant RAG index, proposes a fix, and classifies it ACCEPT / REVIEW / REJECT. A human approves the
recommended action in a Next.js dashboard before it is executed and verified against the cluster.

```
alert ─► planner ─► log_analysis ─► fix_proposer ─► verifier ─► SQLite incident ─► dashboard
                     │  MCP: kubernetes,   │  RAG (Qdrant)  │ rule-based trust gate      │ approve
                     │  github, prometheus │  + validated   │                            ▼
                     ▼                     │  action        │            executor ─► restart/scale ─► verify
```

## Layout

| Path | Contents |
|---|---|
| `src/agents/` | LangGraph pipeline (`graph.py`) and the graph → incident runner |
| `src/mcp_servers/` | MCP servers: `kubernetes` (read + restart/scale), `github` (read-only), `observability` (Prometheus) |
| `src/api/` | FastAPI incident API with SQLite storage, approval and execution endpoints |
| `src/executor/` | Executes an approved action through the Kubernetes MCP server and verifies the rollout |
| `src/rag/` | Runbook chunking, embedding, Qdrant loading, retrieval evaluation |
| `src/data_loaders/` | GAIA dataset loaders used to train the anomaly model |
| `dashboard/` | Next.js human-in-the-loop review dashboard |
| `ml/experiments/` | Phase 2 anomaly-model experiments (EDA, baselines, LightGBM, autoencoders) |
| `scripts/gaia/` | One-off GAIA dataset inspection scripts |
| `scripts/live_retests/` | Live acceptance scenarios against a real cluster |
| `tests/` | Pytest suite; tests marked `live` need a cluster, Qdrant, and LLM keys |

## Running locally

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use .venv/bin/activate elsewhere
pip install -r requirements.txt pytest httpx
cp .env.example .env                                   # then fill in the keys and export them

uvicorn src.api.main:app --port 8000                   # API
cd dashboard && npm install && npm run dev             # dashboard on http://localhost:3000
```

The analysis endpoint additionally needs a reachable kubeconfig context, Qdrant at `QDRANT_URL` with the
`runbook_chunks` collection loaded (`src/rag/`), and `GROQ_API_KEY` / `GEMINI_API_KEY`. MCP servers that
cannot start (for example GitHub without `GITHUB_PAT`) are skipped.

Experiment scripts in `ml/experiments/` log to `mlflow.db` at the repository root (artifacts under
`mlruns/`) unless `MLFLOW_TRACKING_URI` is set. Browse them with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`. Both are local files and are not committed.

Live retest scenarios are run from the repository root as modules, e.g.
`python -m scripts.live_retests.case6_retest`.

## Local lab cluster

`scripts/up.ps1` builds everything the agent investigates, using Terraform and Helm:

| Component | Where | Purpose |
|---|---|---|
| kind cluster `ai-sre` | Docker | One-node Kubernetes (context `kind-ai-sre`, merged into `~/.kube/config`) |
| Online Boutique | namespace `online-boutique` | The demo shop whose failures the agent diagnoses (storefront on `http://127.0.0.1:8081`) |
| Prometheus + kube-state-metrics | namespace `monitoring` | Metrics for the observability MCP server (`http://localhost:9090`) |
| Chaos Mesh | namespace `chaos-mesh` | Fault injection; experiments in `infra/chaos-mesh-experiments/` |
| Qdrant | Docker container `qdrant` | RAG index, kept outside the cluster so it survives rebuilds |

```powershell
.\scripts\up.ps1 -LoadRag          # cluster + Qdrant, then (re)build the runbook index
.\scripts\up.ps1 -Full             # also ad service, load generator, Grafana (needs ~6 GB for Docker)
.\scripts\down.ps1                 # remove the cluster (Qdrant keeps running)
```

The default "light" profile fits in about 3.5 GB of Docker memory. The first run pulls roughly 1.5 GB of
images and takes 10-15 minutes; if a Helm install times out on a slow disk, re-run `up.ps1`.

Break things and let the agent investigate:

```powershell
kubectl apply -f infra/scenarios/scenarios.yaml                    # 8 broken workloads in namespace sre-lab
kubectl apply -f infra/chaos-mesh-experiments/pod-kill-cartservice.yaml
python scripts/analyze_alert.py "Deployment lab-oom-killed in namespace sre-lab keeps restarting"
```

`scripts/analyze_alert.py` runs the full pipeline and prints each stage; add `--persist` to save the incident
for the dashboard. Team runbooks live in `docs/runbooks/` (committed); they are indexed together with the
public postmortems in `data/runbooks/void/` by `src/rag/chunk_runbooks.py`.

## Security model

- Every state-changing endpoint (`/incidents/analyze`, `POST /incidents`, `approve`, `reject`, `execute`)
  requires the `X-API-Key` header when `AI_SRE_API_KEY` is set. Always set it outside local development.
- The dashboard never sees that key: the browser calls `dashboard/app/api/sre/*`, a server-side relay that
  adds it. Set `AI_SRE_API_URL` and `AI_SRE_API_KEY` in the dashboard's server environment, and
  `DASHBOARD_PASSWORD` to put the dashboard behind HTTP Basic auth.
- Execution requires human approval (unless `AUTO_EXECUTE_ON_ACCEPT=true`), is claimed atomically so an
  incident can run at most once, and REJECT-classified incidents cannot be approved.

## Tests

```bash
python -m pytest -m "not live" -q
```
