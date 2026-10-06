locals {
  kubeconfig_path = var.kubeconfig_path != "" ? var.kubeconfig_path : "${path.module}/.kube/${var.cluster_name}.kubeconfig"

  # NodePorts the cluster exposes on the host through kind's port mappings.
  frontend_node_port   = 30080
  prometheus_node_port = 30090
  grafana_node_port    = 30300
}

# --- Cluster ---------------------------------------------------------------------------------

resource "kind_cluster" "this" {
  name            = var.cluster_name
  kubeconfig_path = local.kubeconfig_path
  wait_for_ready  = true

  kind_config {
    kind        = "Cluster"
    api_version = "kind.x-k8s.io/v1alpha4"

    # One node keeps memory low enough for a laptop running Docker Desktop.
    node {
      role = "control-plane"

      extra_port_mappings {
        container_port = local.frontend_node_port
        host_port      = var.frontend_host_port
      }
      extra_port_mappings {
        container_port = local.prometheus_node_port
        host_port      = var.prometheus_host_port
      }
      extra_port_mappings {
        container_port = local.grafana_node_port
        host_port      = var.grafana_host_port
      }
    }
  }
}

provider "kubernetes" {
  host                   = kind_cluster.this.endpoint
  cluster_ca_certificate = kind_cluster.this.cluster_ca_certificate
  client_certificate     = kind_cluster.this.client_certificate
  client_key             = kind_cluster.this.client_key
}

provider "helm" {
  kubernetes {
    host                   = kind_cluster.this.endpoint
    cluster_ca_certificate = kind_cluster.this.cluster_ca_certificate
    client_certificate     = kind_cluster.this.client_certificate
    client_key             = kind_cluster.this.client_key
  }
}

# --- Metrics: Prometheus (what the observability MCP server queries) ------------------------

resource "helm_release" "prometheus" {
  name             = "prometheus"
  namespace        = "monitoring"
  create_namespace = true
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "prometheus"
  timeout          = 900

  values = [
    file("${path.module}/../helm/values/prometheus.yaml"),
    yamlencode({ server = { service = { nodePort = local.prometheus_node_port } } }),
  ]
}

resource "helm_release" "grafana" {
  count = var.enable_grafana ? 1 : 0

  name       = "grafana"
  namespace  = "monitoring"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  timeout    = 900

  values = [
    file("${path.module}/../helm/values/grafana.yaml"),
    yamlencode({
      adminPassword = var.grafana_admin_password
      service       = { nodePort = local.grafana_node_port }
    }),
  ]

  depends_on = [helm_release.prometheus]
}

# --- Fault injection ---------------------------------------------------------------------------

resource "helm_release" "chaos_mesh" {
  count = var.enable_chaos_mesh ? 1 : 0

  name             = "chaos-mesh"
  namespace        = "chaos-mesh"
  create_namespace = true
  repository       = "https://charts.chaos-mesh.org"
  chart            = "chaos-mesh"
  timeout          = 900

  values = [file("${path.module}/../helm/values/chaos-mesh.yaml")]
}

# --- Workload under test: Online Boutique --------------------------------------------------

resource "helm_release" "online_boutique" {
  name             = "onlineboutique"
  namespace        = "online-boutique"
  create_namespace = true
  repository       = "oci://us-docker.pkg.dev/online-boutique-ci/charts"
  chart            = "onlineboutique"
  version          = "0.10.7"
  timeout          = 900

  values = [
    file("${path.module}/../helm/values/online-boutique.yaml"),
    yamlencode({
      adService     = { create = var.enable_adservice }
      loadGenerator = { create = var.enable_loadgenerator }
    }),
  ]

  # Prometheus first so the shop's first minutes of metrics are captured.
  depends_on = [helm_release.prometheus]
}

# kind has no cloud load balancer, so expose the storefront through a NodePort mapped to the host.
resource "kubernetes_service_v1" "frontend_nodeport" {
  metadata {
    name      = "frontend-nodeport"
    namespace = "online-boutique"
  }

  spec {
    type     = "NodePort"
    selector = { app = "frontend" }

    port {
      port        = 80
      target_port = 8080
      node_port   = local.frontend_node_port
    }
  }

  depends_on = [helm_release.online_boutique]
}

# Namespace for the deliberately broken deployments in infra/scenarios (kept apart from the shop).
resource "kubernetes_namespace_v1" "sre_lab" {
  metadata {
    name = "sre-lab"
  }

  depends_on = [kind_cluster.this]
}
