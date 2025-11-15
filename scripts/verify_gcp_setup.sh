#!/bin/bash
# GCP Setup Verification Script
# Verifies all GCP configurations are correct

set -e

echo "================================================"
echo "GCP Setup Verification"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load values from airflow/.env file (lines 41-42 based on user's note)
ENV_FILE="airflow/.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading configuration from $ENV_FILE..."
    # Source the .env file to get variables
    set -a
    source "$ENV_FILE"
    set +a
    echo "✓ Environment file loaded"
else
    echo -e "${YELLOW}⚠${NC} $ENV_FILE not found, using environment variables or defaults"
fi

# Get project ID and bucket from .env (now loaded) or environment
# Note: No defaults - must come from .env file
PROJECT_ID="${GCP_PROJECT_ID}"
BUCKET_NAME="${GCS_BUCKET_NAME}"

echo ""
echo "Using Project ID: $PROJECT_ID"
echo "Using Bucket: $BUCKET_NAME"
echo "  (loaded from: $ENV_FILE or environment variables)"
echo ""

# Track errors
ERRORS=0

# Function to check status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        ERRORS=$((ERRORS + 1))
    fi
}

# Function to check if value exists
check_value() {
    if [ -z "$2" ]; then
        echo -e "${YELLOW}⚠${NC} $1: Not set"
        return 1
    else
        echo -e "${GREEN}✓${NC} $1: $2"
        return 0
    fi
}

# ============================================
# Step 1: Verify GCP Project
# ============================================
echo "1. Verifying GCP Project..."
echo "----------------------------------------"

# Check if gcloud is authenticated
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    check_status "GCloud authenticated"
    ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)
    echo "   Active account: $ACTIVE_ACCOUNT"
else
    echo -e "${RED}✗${NC} GCloud not authenticated"
    echo "   Run: gcloud auth login"
    ERRORS=$((ERRORS + 1))
fi

# Check if project is set
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$CURRENT_PROJECT" = "$PROJECT_ID" ]; then
    check_status "Project set correctly: $PROJECT_ID"
else
    echo -e "${YELLOW}⚠${NC} Current project: $CURRENT_PROJECT"
    echo "   Run: gcloud config set project $PROJECT_ID"
fi

echo ""

# ============================================
# Step 2: Verify Environment Variables
# ============================================
echo "2. Verifying Environment Variables..."
echo "----------------------------------------"

check_value "GCP_PROJECT_ID" "$GCP_PROJECT_ID"
check_value "GCS_BUCKET_NAME" "$GCS_BUCKET_NAME"

# Check credentials file
if [ -f ~/gcp-service-account.json ]; then
    check_status "Service account key exists: ~/gcp-service-account.json"
    
    # Try to extract project from key
    KEY_PROJECT=$(cat ~/gcp-service-account.json | grep -o '"project_id": "[^"]*"' | cut -d'"' -f4)
    if [ "$KEY_PROJECT" = "$PROJECT_ID" ]; then
        check_status "Service account project matches: $PROJECT_ID"
    else
        echo -e "${YELLOW}⚠${NC} Service account project: $KEY_PROJECT (expected: $PROJECT_ID)"
    fi
else
    echo -e "${RED}✗${NC} Service account key not found: ~/gcp-service-account.json"
    ERRORS=$((ERRORS + 1))
fi

# Check airflow/.env file exists and has correct values
if [ -f "$ENV_FILE" ]; then
    check_status "airflow/.env file exists"
    
    # Get values from .env file directly (lines 41-42 approximately)
    ENV_PROJECT=$(grep "^GCP_PROJECT_ID=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    ENV_BUCKET=$(grep "^GCS_BUCKET_NAME=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    
    if [ -n "$ENV_PROJECT" ]; then
        check_status "airflow/.env GCP_PROJECT_ID found: $ENV_PROJECT"
        # Use the value from .env file as the source of truth
        PROJECT_ID="$ENV_PROJECT"
    else
        echo -e "${RED}✗${NC} GCP_PROJECT_ID not found in airflow/.env"
        ERRORS=$((ERRORS + 1))
    fi
    
    if [ -n "$ENV_BUCKET" ]; then
        check_status "airflow/.env GCS_BUCKET_NAME found: $ENV_BUCKET"
        # Use the value from .env file as the source of truth
        BUCKET_NAME="$ENV_BUCKET"
    else
        echo -e "${RED}✗${NC} GCS_BUCKET_NAME not found in airflow/.env"
        ERRORS=$((ERRORS + 1))
    fi
    
    echo ""
    echo "Using values from airflow/.env:"
    echo "  GCP_PROJECT_ID: $PROJECT_ID"
    echo "  GCS_BUCKET_NAME: $BUCKET_NAME"
    echo ""
else
    echo -e "${RED}✗${NC} airflow/.env file not found"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ============================================
# Step 3: Verify GCP APIs
# ============================================
echo "3. Verifying GCP APIs..."
echo "----------------------------------------"

REQUIRED_APIS=(
    "aiplatform.googleapis.com"
    "cloudbuild.googleapis.com"
    "storage-component.googleapis.com"
    "monitoring.googleapis.com"
    "pubsub.googleapis.com"
    "secretmanager.googleapis.com"
    "cloudscheduler.googleapis.com"
)

# Get list of enabled APIs once (more efficient and avoids timeout issues)
ENABLED_APIS=$(gcloud services list --enabled --project="$PROJECT_ID" --format="value(config.name)" 2>/dev/null)

for API in "${REQUIRED_APIS[@]}"; do
    if echo "$ENABLED_APIS" | grep -q "^${API}$"; then
        check_status "API enabled: $API"
    else
        echo -e "${RED}✗${NC} API not enabled: $API"
        echo "   Run: gcloud services enable $API --project=$PROJECT_ID"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# ============================================
# Step 4: Verify GCS Bucket
# ============================================
echo "4. Verifying GCS Bucket..."
echo "----------------------------------------"

if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null 2>&1; then
    check_status "Bucket exists: gs://$BUCKET_NAME"
    
    # Check bucket contents
    if gsutil ls "gs://$BUCKET_NAME/vision/" &>/dev/null 2>&1; then
        check_status "Vision data directory exists"
    else
        echo -e "${YELLOW}⚠${NC} Vision data directory not found (may need to run DataPipeline)"
    fi
    
    if gsutil ls "gs://$BUCKET_NAME/RAG/" &>/dev/null 2>&1; then
        check_status "RAG data directory exists"
    else
        echo -e "${YELLOW}⚠${NC} RAG data directory not found (may need to run DataPipeline)"
    fi
else
    echo -e "${RED}✗${NC} Bucket not found: gs://$BUCKET_NAME"
    echo "   Run: gsutil mb -l us-central1 gs://$BUCKET_NAME"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ============================================
# Step 5: Verify Secret Manager
# ============================================
echo "5. Verifying Secret Manager..."
echo "----------------------------------------"

# First check if Secret Manager API is enabled (required for this check)
if ! gcloud services list --enabled --project="$PROJECT_ID" --format="value(config.name)" 2>/dev/null | grep -q "^secretmanager.googleapis.com$"; then
    echo -e "${YELLOW}⚠${NC} Secret Manager API not enabled - skipping secret checks"
    echo "   Enable with: gcloud services enable secretmanager.googleapis.com --project=$PROJECT_ID"
    echo ""
else
    # Check secrets (with error handling)
    if gcloud secrets describe smtp-username --project="$PROJECT_ID" &>/dev/null 2>&1; then
        check_status "Secret exists: smtp-username"
    else
        echo -e "${YELLOW}⚠${NC} Secret not found: smtp-username"
        echo "   Run: echo 'your-email@gmail.com' | gcloud secrets create smtp-username --data-file=- --project=$PROJECT_ID"
    fi
    
    if gcloud secrets describe smtp-password --project="$PROJECT_ID" &>/dev/null 2>&1; then
        check_status "Secret exists: smtp-password"
    else
        echo -e "${YELLOW}⚠${NC} Secret not found: smtp-password"
        echo "   Run: echo 'your-app-password' | gcloud secrets create smtp-password --data-file=- --project=$PROJECT_ID"
    fi
fi

echo ""

# ============================================
# Step 6: Verify Cloud Build Triggers
# ============================================
echo "6. Verifying Cloud Build Triggers..."
echo "----------------------------------------"

# Get triggers list once (more efficient)
BUILD_TRIGGERS=$(gcloud builds triggers list --project="$PROJECT_ID" --format="value(name)" 2>/dev/null || echo "")

if echo "$BUILD_TRIGGERS" | grep -q "vision-model-training"; then
    check_status "Cloud Build trigger exists: vision-model-training"
else
    echo -e "${YELLOW}⚠${NC} Cloud Build trigger not found: vision-model-training"
    echo "   This is OK if not using GitHub integration yet"
fi

if echo "$BUILD_TRIGGERS" | grep -q "rag-model-training"; then
    check_status "Cloud Build trigger exists: rag-model-training"
else
    echo -e "${YELLOW}⚠${NC} Cloud Build trigger not found: rag-model-training"
    echo "   This is OK if not using GitHub integration yet"
fi

echo ""

# ============================================
# Step 7: Verify Pub/Sub Topics
# ============================================
echo "7. Verifying Pub/Sub Topics..."
echo "----------------------------------------"

if gcloud pubsub topics describe model-alerts --project="$PROJECT_ID" &>/dev/null 2>&1; then
    check_status "Pub/Sub topic exists: model-alerts"
else
    echo -e "${YELLOW}⚠${NC} Pub/Sub topic not found: model-alerts"
    echo "   Run: gcloud pubsub topics create model-alerts --project=$PROJECT_ID"
fi

if gcloud pubsub topics describe model-retraining-check --project="$PROJECT_ID" &>/dev/null 2>&1; then
    check_status "Pub/Sub topic exists: model-retraining-check"
else
    echo -e "${YELLOW}⚠${NC} Pub/Sub topic not found: model-retraining-check"
    echo "   Run: gcloud pubsub topics create model-retraining-check --project=$PROJECT_ID"
fi

echo ""

# ============================================
# Step 8: Check Python Files for Hardcoded Values
# ============================================
echo "8. Checking Python Files for Hardcoded Values..."
echo "----------------------------------------"

# Check if any Python files still have hardcoded project IDs (not using env vars)
HARDCODED=$(grep -r "medscanai-476203" --include="*.py" . 2>/dev/null | grep -v "\.env" | grep -v "__pycache__" | grep -v "getenv\|os.getenv" | head -5 || true)

if [ -z "$HARDCODED" ]; then
    check_status "No hardcoded project IDs found (all using env vars)"
else
    echo -e "${YELLOW}⚠${NC} Found potential hardcoded values:"
    echo "$HARDCODED" | head -3
    echo "   (This may be OK if in comments or default values)"
fi

echo ""

# ============================================
# Summary
# ============================================
echo "================================================"
echo "Verification Summary"
echo "================================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run DataPipeline if data is missing from GCS"
    echo "2. Test Cloud Build: gcloud builds submit --config=cloudbuild/vision-training.yaml"
    echo "3. Check monitoring: https://console.cloud.google.com/monitoring?project=$PROJECT_ID"
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS critical issue(s)${NC}"
    echo ""
    echo "Please fix the issues marked with ✗ above before proceeding."
    exit 1
fi

