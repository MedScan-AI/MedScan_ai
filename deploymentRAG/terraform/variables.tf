variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "medscanai-476500"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "rag-service"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "enable_monitoring" {
  description = "Enable monitoring dashboard and alerts"
  type        = bool
  default     = true
}

variable "monitoring_email" {
  description = "Email for alert notifications"
  type        = string
  default     = "" # Set in tfvars
}

variable "create_notification_channel" {
  description = "Create email notification channel"
  type        = bool
  default     = true
}

variable "enable_apis" {
  description = "Enable required GCP APIs (set to false if APIs are already enabled)"
  type        = bool
  default     = false
}

