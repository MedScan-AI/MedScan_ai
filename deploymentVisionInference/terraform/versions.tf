terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state backend (uncomment to use persistent state in GCS)
  # backend "gcs" {
  #   bucket = "medscan-pipeline-medscanai-476500"
  #   prefix = "terraform/vision-inference/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
