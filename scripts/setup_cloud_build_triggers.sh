#!/bin/bash
# Setup Cloud Build Triggers for GitHub Integration

set -e

# Load from airflow/.env
ENV_FILE="airflow/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "✗ airflow/.env not found!"
    exit 1
fi

PROJECT_ID="${GCP_PROJECT_ID}"
REPO_OWNER="rjaditya-2702"
REPO_NAME="MedScan_ai"
BRANCH="main"

if [ -z "$PROJECT_ID" ]; then
    echo "✗ GCP_PROJECT_ID not found in $ENV_FILE"
    exit 1
fi

echo "================================================"
echo "Cloud Build Triggers Setup"
echo "================================================"
echo "Project: $PROJECT_ID"
echo "Repository: $REPO_OWNER/$REPO_NAME"
echo "Branch: $BRANCH"
echo ""

# Check if GitHub connection exists
echo "Checking GitHub connection..."
CONNECTED_REPOS=$(gcloud builds connections list --region=us-central1 --project="$PROJECT_ID" --format="value(name)" 2>/dev/null || echo "")

if [ -z "$CONNECTED_REPOS" ]; then
    echo ""
    echo "⚠ GitHub is not connected to Cloud Build yet."
    echo ""
    echo "To connect GitHub:"
    echo "  1. Go to: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"
    echo "  2. Click 'Connect Repository'"
    echo "  3. Select 'GitHub (Cloud Build GitHub App)'"
    echo "  4. Authenticate and select repository: $REPO_OWNER/$REPO_NAME"
    echo ""
    echo "OR run this command (it will open a browser for authorization):"
    echo "  gcloud builds connections create github \\"
    echo "    --connector-installation-name=projects/$PROJECT_ID/locations/us-central1/githubConnections/github-connection \\"
    echo "    --region=us-central1 \\"
    echo "    --project=$PROJECT_ID"
    echo ""
    read -p "Have you connected GitHub? (y/n): " connected
    if [ "$connected" != "y" ] && [ "$connected" != "Y" ]; then
        echo "Please connect GitHub first, then run this script again."
        exit 0
    fi
else
    echo "✓ GitHub connection exists"
fi

# Get connection name
CONNECTION_NAME=$(gcloud builds connections list --region=us-central1 --project="$PROJECT_ID" --format="value(name)" 2>/dev/null | head -1)

if [ -z "$CONNECTION_NAME" ]; then
    echo "⚠ Could not find GitHub connection"
    echo "Please connect GitHub manually at: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"
    exit 1
fi

echo "Using connection: $CONNECTION_NAME"
echo ""

# Check if triggers already exist
EXISTING_TRIGGERS=$(gcloud builds triggers list --project="$PROJECT_ID" --format="value(name)" 2>/dev/null || echo "")

# Create Vision model trigger
if echo "$EXISTING_TRIGGERS" | grep -q "vision-model-training"; then
    echo "⚠ Cloud Build trigger already exists: vision-model-training"
    read -p "Delete and recreate? (y/n): " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        gcloud builds triggers delete vision-model-training --project="$PROJECT_ID" --quiet
        echo "Deleted existing trigger"
    else
        echo "Keeping existing trigger"
    fi
fi

if ! echo "$EXISTING_TRIGGERS" | grep -q "vision-model-training"; then
    echo "Creating vision-model-training trigger..."
    gcloud builds triggers create github \
        --name="vision-model-training" \
        --repo-name="$REPO_NAME" \
        --repo-owner="$REPO_OWNER" \
        --branch-pattern="^${BRANCH}$" \
        --included-files="ModelDevelopment/Vision/**,cloudbuild/vision-training.yaml" \
        --build-config=cloudbuild/vision-training.yaml \
        --region=us-central1 \
        --project="$PROJECT_ID" \
        --description="Trigger Vision model training on push to $BRANCH" || {
        echo "⚠ Failed to create trigger. You may need to connect GitHub first."
        echo "   Go to: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"
        exit 1
    }
    echo "✓ Created vision-model-training trigger"
else
    echo "✓ vision-model-training trigger exists"
fi

echo ""

# Create RAG model trigger
if echo "$EXISTING_TRIGGERS" | grep -q "rag-model-training"; then
    echo "⚠ Cloud Build trigger already exists: rag-model-training"
    read -p "Delete and recreate? (y/n): " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        gcloud builds triggers delete rag-model-training --project="$PROJECT_ID" --quiet
        echo "Deleted existing trigger"
    else
        echo "Keeping existing trigger"
    fi
fi

if ! echo "$EXISTING_TRIGGERS" | grep -q "rag-model-training"; then
    echo "Creating rag-model-training trigger..."
    gcloud builds triggers create github \
        --name="rag-model-training" \
        --repo-name="$REPO_NAME" \
        --repo-owner="$REPO_OWNER" \
        --branch-pattern="^${BRANCH}$" \
        --included-files="ModelDevelopment/RAG/**,cloudbuild/rag-training.yaml" \
        --build-config=cloudbuild/rag-training.yaml \
        --region=us-central1 \
        --project="$PROJECT_ID" \
        --description="Trigger RAG model training on push to $BRANCH" || {
        echo "⚠ Failed to create trigger. You may need to connect GitHub first."
        echo "   Go to: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"
        exit 1
    }
    echo "✓ Created rag-model-training trigger"
else
    echo "✓ rag-model-training trigger exists"
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Verification:"
gcloud builds triggers list --project="$PROJECT_ID" --format="table(name,branch,filename)" | grep -E "(NAME|vision|rag)" || echo "No triggers found"

echo ""
echo "View triggers:"
echo "  https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"

