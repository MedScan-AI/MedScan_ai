################################################################################
# Vision Inference API - Terraform Infrastructure
################################################################################
#
# This Terraform configuration manages ONLY the following Vision Inference resources:
#
# Resources Managed (will be created/updated/destroyed by Terraform):
#   • Cloud Run service: vision-inference-api
#   • Artifact Registry repository: vision-inference
#   • IAM bindings for Cloud Build service account
#   • API enablement (Cloud Run, Artifact Registry, Cloud Build, Storage)
#
# Resources NOT Managed (will NOT be affected by terraform destroy):
#   • GCS Bucket: medscan-pipeline-medscanai-476500
#   • Trained models in GCS
#   • Other Cloud Run services (e.g., rag-service)
#   • Other Artifact Registry repositories
#   • Cloud Build service account itself
#   • Any manually created resources
#
################################################################################

# Enable required APIs (optional - set enable_apis=true if needed)
resource "google_project_service" "run_api" {
  count            = var.enable_apis ? 1 : 0
  service          = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifact_registry_api" {
  count            = var.enable_apis ? 1 : 0
  service          = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild_api" {
  count            = var.enable_apis ? 1 : 0
  service          = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage_api" {
  count            = var.enable_apis ? 1 : 0
  service          = "storage-api.googleapis.com"
  disable_on_destroy = false
}

# Artifact Registry repository is assumed to exist (created manually or via other means)
# Terraform will not create or manage the repository

# Get the Cloud Build service account
data "google_project" "project" {}

locals {
  cloudbuild_sa = "${data.google_project.project.number}-compute@developer.gserviceaccount.com"
  image_url = var.container_image != "" ? var.container_image : "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}/vision-inference:latest"
}

# Grant Cloud Build service account permissions to deploy to Cloud Run
resource "google_project_iam_member" "cloudbuild_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${local.cloudbuild_sa}"

  # depends_on removed - Terraform will infer dependencies
}

# Grant Cloud Build service account permissions to act as service accounts
resource "google_project_iam_member" "cloudbuild_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${local.cloudbuild_sa}"

  # depends_on removed - Terraform will infer dependencies
}

# Grant Cloud Build service account permissions to read from GCS
resource "google_storage_bucket_iam_member" "cloudbuild_gcs_reader" {
  bucket = var.gcs_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${local.cloudbuild_sa}"
}

# Cloud Run service for Vision Inference API
resource "google_cloud_run_service" "vision_inference_api" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      containers {
        image = local.image_url

        resources {
          limits = {
            cpu    = var.cpu
            memory = var.memory
          }
        }

        ports {
          container_port = 5000
        }

        env {
          name  = "USE_GCS_MODELS"
          value = "true"
        }

        env {
          name  = "GCS_BUCKET_NAME"
          value = var.gcs_bucket_name
        }

        env {
          name  = "GCS_MODELS_PREFIX"
          value = var.gcs_models_prefix
        }

        # Startup probe configuration
        startup_probe {
          tcp_socket {
            port = 5000
          }
          initial_delay_seconds = 0
          timeout_seconds       = 240
          period_seconds        = 240
          failure_threshold     = 1
        }
      }

      # Service account for the Cloud Run service
      service_account_name = local.cloudbuild_sa

      # Container concurrency
      container_concurrency = 80

      # Request timeout
      timeout_seconds = var.timeout
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale"     = tostring(var.min_instances)
        "autoscaling.knative.dev/maxScale"     = tostring(var.max_instances)
        "run.googleapis.com/startup-cpu-boost" = "true"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  # No dependencies - Artifact Registry repository is assumed to exist

  # Lifecycle to prevent destruction due to image changes
  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image,
      template[0].metadata[0].annotations["client.knative.dev/user-image"],
      template[0].metadata[0].annotations["run.googleapis.com/client-name"],
      template[0].metadata[0].annotations["run.googleapis.com/client-version"]
    ]
  }
}

# IAM policy to allow unauthenticated access (if enabled)
resource "google_cloud_run_service_iam_member" "allow_unauthenticated" {
  count = var.allow_unauthenticated ? 1 : 0

  service  = google_cloud_run_service.vision_inference_api.name
  location = google_cloud_run_service.vision_inference_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Grant Cloud Run service account permissions to read models from GCS
resource "google_storage_bucket_iam_member" "cloudrun_gcs_reader" {
  bucket = var.gcs_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${local.cloudbuild_sa}"
}
