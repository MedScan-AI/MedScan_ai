variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "medscanai-476500"
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "vision-inference-api"
}

variable "repository_name" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "vision-inference"
}

variable "gcs_bucket_name" {
  description = "GCS bucket name for models"
  type        = string
  default     = "medscan-pipeline-medscanai-476500"
}

variable "gcs_models_prefix" {
  description = "GCS prefix for trained models"
  type        = string
  default     = "vision/trained_models"
}

# Cloud Run configuration
variable "memory" {
  description = "Memory allocation for Cloud Run service"
  type        = string
  default     = "2Gi"
}

variable "cpu" {
  description = "CPU allocation for Cloud Run service"
  type        = string
  default     = "1"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 3
}

variable "timeout" {
  description = "Request timeout in seconds"
  type        = number
  default     = 300
}

variable "container_image" {
  description = "Container image for Cloud Run (leave empty to use latest from Artifact Registry)"
  type        = string
  default     = ""
}

variable "allow_unauthenticated" {
  description = "Allow unauthenticated access to the service"
  type        = bool
  default     = true
}

variable "enable_apis" {
  description = "Enable required GCP APIs (set to false if APIs are already enabled)"
  type        = bool
  default     = false
}

# Monitoring configuration
variable "enable_monitoring" {
  description = "Enable Google Cloud Monitoring (alerts and dashboards)"
  type        = bool
  default     = true
}

variable "monitoring_email" {
  description = "Email address for monitoring alerts (leave empty to disable email notifications)"
  type        = string
  default     = "sriharsha.py@gmail.com"
}

variable "create_notification_channel" {
  description = "Create email notification channel (set to false if you lack monitoring.notificationChannelEditor permission)"
  type        = bool
  default     = true
}
