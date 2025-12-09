#!/bin/bash
# Import existing Vision Inference resources into Terraform state

set -e

PROJECT_ID="medscanai-476500"
REGION="us-central1"
SERVICE_NAME="vision-inference-api"
REPO_NAME="vision-inference"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Importing Existing Resources into Terraform"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Initialize Terraform
echo "1. Initializing Terraform..."
terraform init
echo ""

# Import Cloud Run Service
echo "2. Importing Cloud Run service: ${SERVICE_NAME}..."
terraform import google_cloud_run_service.vision_inference_api \
  "locations/${REGION}/namespaces/${PROJECT_ID}/services/${SERVICE_NAME}" || echo "  ⚠️  Service may not exist or already imported"
echo ""

# Import Artifact Registry Repository
echo "3. Importing Artifact Registry repository: ${REPO_NAME}..."
terraform import google_artifact_registry_repository.vision_inference \
  "projects/${PROJECT_ID}/locations/${REGION}/repositories/${REPO_NAME}" || echo "  ⚠️  Repository may not exist or already imported"
echo ""

# Import IAM bindings
echo "4. Importing IAM bindings..."

# Cloud Build Run Admin
terraform import google_project_iam_member.cloudbuild_run_admin \
  "${PROJECT_ID} roles/run.admin serviceAccount:246542889931-compute@developer.gserviceaccount.com" || echo "  ⚠️  IAM binding may not exist or already imported"

# Cloud Build Service Account User
terraform import google_project_iam_member.cloudbuild_sa_user \
  "${PROJECT_ID} roles/iam.serviceAccountUser serviceAccount:246542889931-compute@developer.gserviceaccount.com" || echo "  ⚠️  IAM binding may not exist or already imported"

# GCS IAM bindings
terraform import google_storage_bucket_iam_member.cloudbuild_gcs_reader \
  "b/medscan-pipeline-medscanai-476500 roles/storage.objectViewer serviceAccount:246542889931-compute@developer.gserviceaccount.com" || echo "  ⚠️  GCS IAM may not exist or already imported"

terraform import google_storage_bucket_iam_member.cloudrun_gcs_reader \
  "b/medscan-pipeline-medscanai-476500 roles/storage.objectViewer serviceAccount:246542889931-compute@developer.gserviceaccount.com" || echo "  ⚠️  GCS IAM may not exist or already imported"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Import complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Run: terraform plan"
echo "  2. Review changes (should show 'No changes')"
echo "  3. Now you can run: terraform destroy (to actually delete resources)"
echo ""
