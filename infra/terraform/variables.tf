variable "cluster_name" {
  description = "Name of the kind cluster (the kubectl context becomes kind-<name>)."
  type        = string
  default     = "ai-sre"
}

variable "kubeconfig_path" {
  description = "Where Terraform writes the cluster's kubeconfig. A dedicated file, so an existing ~/.kube/config is never overwritten; scripts/up.ps1 merges it into the default config."
  type        = string
  default     = ""
}

variable "frontend_host_port" {
  description = "Host port that exposes the Online Boutique storefront."
  type        = number
  default     = 8081
}

variable "prometheus_host_port" {
  description = "Host port that exposes Prometheus (matches PROMETHEUS_URL for the observability MCP server)."
  type        = number
  default     = 9090
}

variable "grafana_host_port" {
  description = "Host port that exposes Grafana when enable_grafana is true."
  type        = number
  default     = 3300
}

variable "enable_chaos_mesh" {
  description = "Install Chaos Mesh so infra/chaos-mesh-experiments can inject faults."
  type        = bool
  default     = true
}

variable "enable_adservice" {
  description = "Run the Java ad service (about 300 MiB). Off in the light profile; the storefront works without ads."
  type        = bool
  default     = false
}

variable "enable_loadgenerator" {
  description = "Run the Locust load generator (about 500 MiB) so the shop has steady traffic and metrics."
  type        = bool
  default     = false
}

variable "enable_grafana" {
  description = "Install Grafana with a Prometheus datasource."
  type        = bool
  default     = false
}

variable "grafana_admin_password" {
  description = "Grafana admin password (local lab cluster only)."
  type        = string
  default     = "admin"
  sensitive   = true
}
