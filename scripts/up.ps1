<#
.SYNOPSIS
  Brings up the local AI-SRE lab: a kind cluster with Online Boutique, Prometheus and Chaos Mesh,
  plus the Qdrant container the agent's RAG layer reads from.

.PARAMETER Full
  Also run the ad service, load generator and Grafana. Needs roughly 6 GB of memory for Docker.

.PARAMETER LoadRag
  (Re)build the runbook index in Qdrant from data/runbooks/void and docs/runbooks.

.EXAMPLE
  .\scripts\up.ps1
  .\scripts\up.ps1 -Full -LoadRag
#>
param(
    [switch]$Full,
    [switch]$LoadRag
)

# Native tools (kind, terraform, docker) print progress to stderr, which Windows PowerShell 5.1 turns into
# terminating errors under "Stop"; success is decided by the explicit $LASTEXITCODE checks below instead.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$tf = Join-Path $repo "infra\terraform"
$clusterName = "ai-sre"

# Tools installed with winget are on the machine/user PATH but not always in this session yet.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

function Require-Tool($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "'$name' was not found on PATH. Install it (winget install ...) and reopen the terminal."
    }
}

foreach ($tool in "docker", "kind", "helm", "terraform", "kubectl") { Require-Tool $tool }

docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "The Docker daemon is not running. Start Docker Desktop and retry." }

$memGb = [math]::Round((docker info --format "{{.MemTotal}}") / 1GB, 1)
if ($Full -and $memGb -lt 5.5) {
    Write-Warning "Docker has only $memGb GB. The full profile may be OOM-killed; raise memory in .wslconfig or drop -Full."
} elseif ($memGb -lt 3.4) {
    Write-Warning "Docker has only $memGb GB. The light profile needs about 3.5 GB."
}

# --- Qdrant (outside the cluster, so the RAG index survives cluster rebuilds) ---------------
$qdrant = docker ps -a --filter "name=^qdrant$" --format "{{.Status}}"
if (-not $qdrant) {
    Write-Host "Starting Qdrant..." -ForegroundColor Cyan
    docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "could not start Qdrant (is port 6333 in use?)" }
} elseif ($qdrant -notlike "Up*") {
    docker start qdrant | Out-Null
}

# --- Cluster + workloads -------------------------------------------------------------------
$tfArgs = @("-chdir=$tf")
terraform @tfArgs init -input=false 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "terraform init failed" }

$applyArgs = @("apply", "-auto-approve", "-input=false")
if ($Full) {
    $applyArgs += @("-var", "enable_adservice=true", "-var", "enable_loadgenerator=true", "-var", "enable_grafana=true")
}
Write-Host "Applying Terraform (first run pulls about 1.5 GB of images; this takes several minutes)..." -ForegroundColor Cyan
terraform @tfArgs @applyArgs
if ($LASTEXITCODE -ne 0) { throw "terraform apply failed (see the output above; 'terraform apply' is safe to re-run)" }

# The Kubernetes MCP server reads the default kubeconfig, so merge the new context into it.
kind export kubeconfig --name $clusterName 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "kind export kubeconfig failed" }
kubectl config use-context "kind-$clusterName" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "could not switch to context kind-$clusterName" }

if ($LoadRag) {
    $py = Join-Path $repo ".venv\Scripts\python.exe"
    foreach ($step in "chunk_runbooks", "embed_runbooks", "load_qdrant") {
        Write-Host "RAG: $step" -ForegroundColor Cyan
        & $py (Join-Path $repo "src\rag\$step.py")
        if ($LASTEXITCODE -ne 0) { throw "RAG step $step failed" }
    }
}

Write-Host ""
Write-Host "AI-SRE lab is up." -ForegroundColor Green
Write-Host "  kubectl context : kind-$clusterName"
Write-Host "  Storefront      : http://localhost:8081"
Write-Host "  Prometheus      : http://localhost:9090   (PROMETHEUS_URL in .env)"
if ($Full) { Write-Host "  Grafana         : http://localhost:3300   (admin / admin)" }
Write-Host "  Qdrant          : http://localhost:6333/dashboard"
Write-Host ""
Write-Host "Next: kubectl apply -f infra/scenarios/scenarios.yaml   (broken workloads in namespace sre-lab)"
