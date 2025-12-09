#!/bin/bash
# Cleanup Vision Inference resources - handles both Terraform and manual resources

set -e

PROJECT_ID="${PROJECT_ID:-medscanai-476500}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-vision-inference-api}"
REPO_NAME="${REPO_NAME:-vision-inference}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🗑️  Vision Inference Resource Cleanup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# Function to check if resource exists
resource_exists() {
  local resource_type=$1
  local resource_name=$2
  
  case $resource_type in
    "cloudrun")
      gcloud run services describe "$resource_name" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(metadata.name)" 2>/dev/null || return 1
      ;;
    "artifactregistry")
      gcloud artifacts repositories describe "$resource_name" \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(name)" 2>/dev/null || return 1
      ;;
  esac
}

# Step 1: Try Terraform destroy first
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Attempting Terraform destroy..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if terraform destroy -auto-approve 2>&1 | tee /tmp/terraform_destroy.log; then
  echo "✅ Terraform destroy completed"
  
  # Check if it actually destroyed anything
  if grep -q "Destroy complete! Resources: 0 destroyed" /tmp/terraform_destroy.log; then
    echo "⚠️  Terraform destroyed 0 resources - resources may have been created manually"
    TERRAFORM_DESTROYED=false
  else
    echo "✅ Terraform successfully destroyed resources"
    TERRAFORM_DESTROYED=true
  fi
else
  echo "⚠️  Terraform destroy failed or had issues"
  TERRAFORM_DESTROYED=false
fi

echo ""

# Step 2: Manual cleanup of any remaining resources
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Checking for manually created resources..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

MANUAL_CLEANUP_NEEDED=false

# Check Cloud Run Service
echo "Checking Cloud Run service: ${SERVICE_NAME}..."
if resource_exists "cloudrun" "$SERVICE_NAME"; then
  echo "  ⚠️  Service still exists - will delete manually"
  MANUAL_CLEANUP_NEEDED=true
  
  echo "  Deleting Cloud Run service..."
  gcloud run services delete "$SERVICE_NAME" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --quiet && echo "  ✅ Service deleted" || echo "  ❌ Failed to delete service"
else
  echo "  ✅ Service does not exist (already deleted)"
fi

echo ""

# Check Artifact Registry
echo "Checking Artifact Registry: ${REPO_NAME}..."
if resource_exists "artifactregistry" "$REPO_NAME"; then
  echo "  ⚠️  Repository still exists - will delete manually"
  MANUAL_CLEANUP_NEEDED=true
  
  # List images if any
  IMAGE_COUNT=$(gcloud artifacts docker images list \
    "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}" \
    --format="value(package)" 2>/dev/null | wc -l || echo "0")
  
  if [ "$IMAGE_COUNT" -gt 0 ]; then
    echo "  📦 Repository contains ${IMAGE_COUNT} image(s) - will be deleted with repository"
  fi
  
  echo "  Deleting Artifact Registry repository (force delete if needed)..."
  if gcloud artifacts repositories delete "$REPO_NAME" \
    --location="$REGION" \
    --project="$PROJECT_ID" \
    --quiet 2>&1; then
    echo "  ✅ Repository deleted"
  else
    echo "  ⚠️  Standard delete failed, trying alternative method..."
    # Try with explicit project format
    gcloud artifacts repositories delete \
      "projects/${PROJECT_ID}/locations/${REGION}/repositories/${REPO_NAME}" \
      --quiet 2>&1 && echo "  ✅ Repository deleted" || echo "  ❌ Failed to delete repository - check permissions"
  fi
else
  echo "  ✅ Repository does not exist (already deleted)"
fi

echo ""

# Step 3: Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Cleanup Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$MANUAL_CLEANUP_NEEDED" = true ]; then
  echo "⚠️  Manual cleanup was needed"
  echo ""
  echo "This means resources were created outside of Terraform."
  echo "For future deployments, use Terraform to create resources:"
  echo "  Actions → Vision Inference - Terraform Setup → action=apply"
  echo ""
else
  echo "✅ All resources cleaned up successfully"
  echo ""
fi

echo "Resources deleted:"
echo "  • Cloud Run service: ${SERVICE_NAME}"
echo "  • Artifact Registry: ${REPO_NAME}"
echo ""
echo "Resources NOT affected (safe):"
echo "  ✓ GCS bucket: medscan-pipeline-${PROJECT_ID}"
echo "  ✓ Trained models in GCS"
echo "  ✓ Other Cloud Run services"
echo "  ✓ IAM permissions (remain for future use)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify deletion
echo ""
echo "Verifying deletion..."
if ! resource_exists "cloudrun" "$SERVICE_NAME" && ! resource_exists "artifactregistry" "$REPO_NAME"; then
  echo "✅ All Vision Inference resources successfully removed"
  exit 0
else
  echo "⚠️  Some resources may still exist - check GCP Console"
  exit 1
fi
