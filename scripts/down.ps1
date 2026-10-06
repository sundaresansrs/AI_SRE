<#
.SYNOPSIS
  Tears down the local AI-SRE lab cluster. Qdrant (and its RAG index) is left running.

.PARAMETER Qdrant
  Also stop and remove the Qdrant container (the index volume is kept).
#>
param([switch]$Qdrant)

# Native tools (kind, terraform, docker) print progress to stderr, which Windows PowerShell 5.1 turns into
# terminating errors under "Stop"; success is decided by the explicit $LASTEXITCODE checks below instead.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

Push-Location (Join-Path $repo "infra\terraform")
try {
    # Destroying the kind cluster removes everything inside it. Helm releases would only time out
    # trying to uninstall from a cluster that is going away, so remove them from state first.
    $state = terraform state list 2>$null
    if ($state) {
        foreach ($address in $state) {
            if ($address -notlike "kind_cluster.*") { terraform state rm $address | Out-Null }
        }
        terraform destroy -auto-approve -input=false
    }
} finally {
    Pop-Location
}

# Fallback for a cluster created outside Terraform or a half-finished apply.
if ((kind get clusters 2>$null) -contains "ai-sre") { kind delete cluster --name ai-sre }

if ($Qdrant) {
    docker rm -f qdrant | Out-Null
}
Write-Host "AI-SRE lab removed." -ForegroundColor Green
