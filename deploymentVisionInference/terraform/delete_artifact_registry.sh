#!/bin/bash
# Quick script to force delete the artifact registry

PROJECT_ID="${PROJECT_ID:-medscanai-476500}"
REGION="${REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-vision-inference}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🗑️  Force Delete Artifact Registry"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Repository: ${REPO_NAME}"
echo "Location: ${REGION}"
echo "Project: ${PROJECT_ID}"
echo ""

# Check if repository exists
echo "Checking if repository exists..."
if gcloud artifacts repositories describe "$REPO_NAME" \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  --format="value(name)" 2>/dev/null; then
  
  echo "✅ Repository found"
  echo ""
  
  # List images in the repository
  echo "Listing images in repository..."
  IMAGE_COUNT=$(gcloud artifacts docker images list \
    "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}" \
    --format="value(package)" 2>/dev/null | wc -l)
  
  echo "Found ${IMAGE_COUNT} image(s) in repository"
  echo ""
  
  # Force delete the repository
  echo "Force deleting repository (including all images)..."
  if gcloud artifacts repositories delete "$REPO_NAME" \
    --location="$REGION" \
    --project="$PROJECT_ID" \
    --quiet; then
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Repository deleted successfully"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
  else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ Failed to delete repository"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
  fi
else
  echo "✅ Repository does not exist (already deleted)"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✅ Nothing to delete"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
fi
