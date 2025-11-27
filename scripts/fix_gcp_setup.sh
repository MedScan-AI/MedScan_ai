#!/bin/bash
# Quick Fix Script for GCP Setup Issues
# Reads values from airflow/.env automatically

set -e

echo "================================================"
echo "GCP Setup - Quick Fixes"
echo "================================================"
echo ""

# Load from airflow/.env
ENV_FILE="airflow/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "✓ Loaded configuration from $ENV_FILE"
else
    echo "✗ airflow/.env not found!"
    exit 1
fi

PROJECT_ID="${GCP_PROJECT_ID}"
BUCKET_NAME="${GCS_BUCKET_NAME}"

if [ -z "$PROJECT_ID" ] || [ -z "$BUCKET_NAME" ]; then
    echo "✗ GCP_PROJECT_ID or GCS_BUCKET_NAME not found in $ENV_FILE"
    exit 1
fi

echo "Project ID: $PROJECT_ID"
echo "Bucket Name: $BUCKET_NAME"
echo ""

# Fix 1: Set gcloud project
echo "1. Setting gcloud project..."
gcloud config set project "$PROJECT_ID"
echo "✓ Project set to: $PROJECT_ID"
echo ""

# Fix 2: Enable missing APIs
echo "2. Enabling missing APIs..."
gcloud services enable secretmanager.googleapis.com --project="$PROJECT_ID" || echo "Already enabled"
gcloud services enable cloudscheduler.googleapis.com --project="$PROJECT_ID" || echo "Already enabled"
echo "✓ APIs enabled"
echo ""

# Fix 3: Create SMTP secrets (if not exist)
echo "3. Setting up SMTP secrets..."
echo "Do you want to create SMTP secrets now? (y/n)"
read -r answer

if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    echo "Enter SMTP username (email):"
    read -r smtp_user
    
    echo "Enter SMTP password (Gmail app password):"
    read -s smtp_pass
    echo ""
    
    # Create secrets
    echo "$smtp_user" | gcloud secrets create smtp-username --data-file=- --project="$PROJECT_ID" 2>/dev/null || \
        echo "$smtp_user" | gcloud secrets versions add smtp-username --data-file=- --project="$PROJECT_ID"
    
    echo "$smtp_pass" | gcloud secrets create smtp-password --data-file=- --project="$PROJECT_ID" 2>/dev/null || \
        echo "$smtp_pass" | gcloud secrets versions add smtp-password --data-file=- --project="$PROJECT_ID"
    
    echo "✓ SMTP secrets created"
else
    echo "⏭ Skipping SMTP secrets setup"
    echo "  Run manually:"
    echo "    echo 'email' | gcloud secrets create smtp-username --data-file=- --project=$PROJECT_ID"
    echo "    echo 'password' | gcloud secrets create smtp-password --data-file=- --project=$PROJECT_ID"
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Run verification again:"
echo "  ./scripts/verify_gcp_setup.sh"

