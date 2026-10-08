output "kubeconfig_path" {
  description = "Kubeconfig for the cluster (scripts/up.ps1 merges it into ~/.kube/config)."
  value       = local.kubeconfig_path
}

output "context" {
  description = "kubectl context name."
  value       = "kind-${var.cluster_name}"
}

output "storefront_url" {
  value = "http://localhost:${var.frontend_host_port}"
}

output "prometheus_url" {
  description = "Set PROMETHEUS_URL to this for the observability MCP server."
  value       = "http://localhost:${var.prometheus_host_port}"
}

output "grafana_url" {
  value = var.enable_grafana ? "http://localhost:${var.grafana_host_port} (admin / see grafana_admin_password)" : "disabled (enable_grafana = false)"
}
