#!/bin/bash
# setup_gcp.sh - Setup GCP resources for MedScan AI
# Email-based notifications only

set -e

# Get from environment variables or use defaults
PROJECT_ID="${GCP_PROJECT_ID:-medscanai-476203}"
REGION="${GCP_REGION:-us-central1}"
BUCKET_NAME="${GCS_BUCKET_NAME:-medscan-data}"

# Validate required variables
if [ -z "$PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID environment variable is not set"
    exit 1
fi

if [ -z "$BUCKET_NAME" ]; then
    echo "Error: GCS_BUCKET_NAME environment variable is not set"
    exit 1
fi

echo "Using GCP Project ID: $PROJECT_ID"
echo "Using GCS Bucket: $BUCKET_NAME"
echo ""

echo "Setting up GCP resources for MedScan AI"
echo ""

# Enable APIs
echo "Enabling required GCP APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    monitoring.googleapis.com \
    pubsub.googleapis.com \
    cloudscheduler.googleapis.com \
    --project=$PROJECT_ID

echo "APIs enabled"
echo ""

# Create Pub/Sub topics for internal messaging
echo "Creating Pub/Sub topics..."
gcloud pubsub topics create model-alerts --project=$PROJECT_ID 2>/dev/null || echo "Topic 'model-alerts' already exists"
gcloud pubsub topics create model-retraining-check --project=$PROJECT_ID 2>/dev/null || echo "Topic 'model-retraining-check' already exists"

echo "Pub/Sub topics created"
echo ""

# Create email notification channels
echo "Creating email notification channels..."

# Primary email
gcloud alpha monitoring channels create \
    --display-name="MedScan Primary Alert" \
    --type=email \
    --channel-labels=email_address=harshitha8.shekar@gmail.com \
    --project=$PROJECT_ID 2>/dev/null || echo "Channel for harshitha8.shekar@gmail.com already exists"

# Secondary email
gcloud alpha monitoring channels create \
    --display-name="MedScan Secondary Alert" \
    --type=email \
    --channel-labels=email_address=kothari.sau@northeastern.edu \
    --project=$PROJECT_ID 2>/dev/null || echo "Channel for kothari.sau@northeastern.edu already exists"

echo "Email notification channels created"
echo ""

# Create Cloud Build triggers
echo "Creating Cloud Build triggers..."

# Vision model trigger
gcloud builds triggers create github \
    --name="vision-model-training" \
    --repo-name=MedScan_ai \
    --repo-owner=rjaditya-2702 \
    --branch-pattern="^main$" \
    --included-files="ModelDevelopment/Vision/**,cloudbuild/vision-training.yaml" \
    --build-config=cloudbuild/vision-training.yaml \
    --project=$PROJECT_ID 2>/dev/null || echo "Vision trigger already exists"

# RAG model trigger
gcloud builds triggers create github \
    --name="rag-model-training" \
    --repo-name=MedScan_ai \
    --repo-owner=rjaditya-2702 \
    --branch-pattern="^main$" \
    --included-files="ModelDevelopment/RAG/**,cloudbuild/rag-training.yaml" \
    --build-config=cloudbuild/rag-training.yaml \
    --project=$PROJECT_ID 2>/dev/null || echo "RAG trigger already exists"

echo "Cloud Build triggers created"
echo ""

# Create alert policies
echo "Creating alert policies..."

# Vision model accuracy alert
cat > /tmp/vision-accuracy-alert.yaml << EOF
displayName: "Vision Model Accuracy Drop"
conditions:
  - displayName: "Accuracy below threshold"
    conditionThreshold:
      filter: 'metric.type="custom.googleapis.com/vision/model_accuracy" AND resource.type="global"'
      comparison: COMPARISON_LT
      thresholdValue: 0.70
      duration: 300s
notificationChannels: []
alertStrategy:
  autoClose: 604800s
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/vision-accuracy-alert.yaml --project=$PROJECT_ID 2>/dev/null || echo "Vision accuracy alert already exists"

# RAG model performance alert
cat > /tmp/rag-performance-alert.yaml << EOF
displayName: "RAG Model Performance Degradation"
conditions:
  - displayName: "Composite score below threshold"
    conditionThreshold:
      filter: 'metric.type="custom.googleapis.com/rag/composite_score" AND resource.type="global"'
      comparison: COMPARISON_LT
      thresholdValue: 0.60
      duration: 600s
notificationChannels: []
alertStrategy:
  autoClose: 604800s
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/rag-performance-alert.yaml --project=$PROJECT_ID 2>/dev/null || echo "RAG performance alert already exists"

# Bias detection alert
cat > /tmp/bias-alert.yaml << EOF
displayName: "Model Bias Detected"
conditions:
  - displayName: "Bias disparity above threshold"
    conditionThreshold:
      filter: 'metric.type="custom.googleapis.com/model/bias_disparity" AND resource.type="global"'
      comparison: COMPARISON_GT
      thresholdValue: 0.10
      duration: 0s
notificationChannels: []
alertStrategy:
  autoClose: 86400s
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/bias-alert.yaml --project=$PROJECT_ID 2>/dev/null || echo "Bias alert already exists"

echo "Alert policies created"
echo ""

# Cleanup temp files
rm -f /tmp/*-alert.yaml

echo "GCP Setup Complete!"
echo ""
echo "Summary:"
echo "  - Project: $PROJECT_ID"
echo "  - Region: $REGION"
echo "  - Bucket: gs://$BUCKET_NAME"
echo "  - Email alerts: harshitha8.shekar@gmail.com"
echo ""
echo "Next steps:"
echo "  1. Add GitHub secrets: GCP_SA_KEY, SMTP_USER, SMTP_PASSWORD"
echo "  2. Push code to trigger builds"
echo "  3. Monitor at: https://console.cloud.google.com/cloud-build"
echo ""