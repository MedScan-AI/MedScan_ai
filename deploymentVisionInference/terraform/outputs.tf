output "service_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_service.vision_inference_api.status[0].url
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_service.vision_inference_api.name
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}"
}

output "docker_image_url" {
  description = "Docker image URL for the service"
  value       = local.image_url
}

output "health_endpoint" {
  description = "Health check endpoint"
  value       = "${google_cloud_run_service.vision_inference_api.status[0].url}/health"
}

output "api_docs_endpoint" {
  description = "API documentation endpoint"
  value       = "${google_cloud_run_service.vision_inference_api.status[0].url}/docs"
}

output "tb_prediction_endpoint" {
  description = "TB prediction endpoint"
  value       = "${google_cloud_run_service.vision_inference_api.status[0].url}/predict/tb"
}

output "lung_cancer_prediction_endpoint" {
  description = "Lung Cancer prediction endpoint"
  value       = "${google_cloud_run_service.vision_inference_api.status[0].url}/predict/lung_cancer"
}

output "cloudbuild_service_account" {
  description = "Cloud Build service account email"
  value       = local.cloudbuild_sa
}

# Monitoring outputs
output "monitoring_dashboard_url" {
  description = "URL to the monitoring dashboard"
  value = (var.enable_monitoring && local.monitoring_dashboard_id != "")
    ? format("https://console.cloud.google.com/monitoring/dashboards/custom/%s?project=%s", local.monitoring_dashboard_id, var.project_id)
    : "Monitoring disabled"
}

output "monitoring_alerts" {
  description = "List of monitoring alert policies"
  value = var.enable_monitoring ? [
    local.alert_policy_display_names.high_error_rate,
    local.alert_policy_display_names.high_latency,
    local.alert_policy_display_names.service_unavailable,
    local.alert_policy_display_names.high_cpu,
    local.alert_policy_display_names.high_memory
  ] : []
}
