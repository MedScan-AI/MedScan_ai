################################################################################
# RAG Service - Terraform Infrastructure (Monitoring Only)
################################################################################
#
# This Terraform configuration manages ONLY monitoring resources for RAG service:
#
# Resources Managed (will be created/updated/destroyed by Terraform):
#   • Custom log-based metrics for RAG quality monitoring
#   • Monitoring dashboard with Cloud Run + RAG-specific metrics
#   • Alert policies for production issues and quality degradation
#   • Email notification channel for alerts
#
# Resources NOT Managed (will NOT be affected by terraform destroy):
#   • Cloud Run service: rag-service (assumed to exist)
#   • GCS Bucket: medscan-pipeline-medscanai-476500
#   • RAG models and indices in GCS
#   • Other Cloud Run services
#   • Any manually created resources
#
################################################################################

# Enable Monitoring API (optional)
resource "google_project_service" "monitoring_api" {
  count              = var.enable_apis ? 1 : 0
  service            = "monitoring.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "logging_api" {
  count              = var.enable_apis ? 1 : 0
  service            = "logging.googleapis.com"
  disable_on_destroy = false
}

