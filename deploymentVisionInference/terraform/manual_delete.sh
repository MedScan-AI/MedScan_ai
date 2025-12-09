#!/bin/bash
# Manually delete Vision Inference resources using gcloud

set -e

PROJECT_ID="medscanai-476500"
REGION="us-central1"
SERVICE_NAME="vision-inference-api"
REPO_NAME="vision-inference"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  MANUAL DELETION - VISION INFERENCE ONLY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will DELETE:"
echo "  ❌ Cloud Run service: ${SERVICE_NAME}"
echo "  ❌ Artifact Registry: ${REPO_NAME}"
echo ""
echo "This will NOT affect:"
echo "  ✅ GCS bucket and models"
echo "  ✅ Other Cloud Run services"
echo "  ✅ IAM permissions (will remain)"
echo ""
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
  echo "Cancelled."
  exit 0
fi

echo ""
echo "1. Deleting Cloud Run service: ${SERVICE_NAME}..."
gcloud run services delete ${SERVICE_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --quiet || echo "  ⚠️  Service may not exist"

echo ""
echo "2. Deleting Artifact Registry repository: ${REPO_NAME}..."
gcloud artifacts repositories delete ${REPO_NAME} \
  --location=${REGION} \
  --project=${PROJECT_ID} \
  --quiet || echo "  ⚠️  Repository may not exist"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Manual deletion complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Note: IAM bindings for Cloud Build were NOT removed."
echo "They are harmless and can stay."
echo ""
