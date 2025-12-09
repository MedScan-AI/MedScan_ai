#!/bin/bash
# Quick test script for RAG deployment
# Usage: ./scripts/test-rag-deployment.sh

set -e

PROJECT_ID="medscanai-476500"
REGION="us-central1"
SERVICE_NAME="rag-service"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Testing RAG Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not found. Installing jq..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    else
        echo "Please install jq manually: https://stedolan.github.io/jq/download/"
        exit 1
    fi
fi

# Check if service exists
echo "1️⃣  Checking if Cloud Run service exists..."
if gcloud run services describe $SERVICE_NAME \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format="value(status.url)" &>/dev/null; then
    echo "✅ Service exists"
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --format="value(status.url)")
    echo "   URL: $SERVICE_URL"
else
    echo "❌ Service not found. Please deploy first using rag-complete-setup.yaml"
    exit 1
fi

echo ""
echo "2️⃣  Testing Health Endpoint..."
HEALTH_RESPONSE=$(curl -s -f "$SERVICE_URL/health" || echo '{"ready":false}')
READY=$(echo $HEALTH_RESPONSE | jq -r '.ready // false')

if [ "$READY" = "true" ]; then
    echo "✅ Health check passed"
    echo "$HEALTH_RESPONSE" | jq
else
    echo "❌ Health check failed"
    echo "$HEALTH_RESPONSE" | jq
    exit 1
fi

echo ""
echo "3️⃣  Testing Config Endpoint..."
CONFIG_RESPONSE=$(curl -s "$SERVICE_URL/config")
if echo "$CONFIG_RESPONSE" | jq . &>/dev/null; then
    echo "✅ Config endpoint accessible"
    echo "$CONFIG_RESPONSE" | jq '.model_name, .model_type, .k' 2>/dev/null || echo "$CONFIG_RESPONSE"
else
    echo "⚠️  Config endpoint returned non-JSON response"
    echo "$CONFIG_RESPONSE"
fi

echo ""
echo "4️⃣  Testing Prediction Endpoint..."
PREDICTION_RESPONSE=$(curl -s -X POST "$SERVICE_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"instances":[{"query":"What is tuberculosis?"}]}')

SUCCESS=$(echo $PREDICTION_RESPONSE | jq -r '.predictions[0].success // false' 2>/dev/null || echo "false")

if [ "$SUCCESS" = "true" ]; then
    echo "✅ Prediction test passed"
    echo "$PREDICTION_RESPONSE" | jq '.predictions[0] | {success, latency, composite_score, num_retrieved_docs}' 2>/dev/null || echo "Response received"
else
    echo "⚠️  Prediction test returned success=false or error"
    echo "$PREDICTION_RESPONSE" | jq . 2>/dev/null || echo "$PREDICTION_RESPONSE"
fi

echo ""
echo "5️⃣  Checking Monitoring Resources..."

# Check custom metrics
echo "   Checking custom metrics..."
METRIC_COUNT=$(gcloud logging metrics list --project=$PROJECT_ID --format="value(name)" 2>/dev/null | grep -c "rag_" || echo "0")
if [ "$METRIC_COUNT" -gt 0 ]; then
    echo "   ✅ Found $METRIC_COUNT custom metrics"
    gcloud logging metrics list --project=$PROJECT_ID --format="table(name,description)" 2>/dev/null | grep rag_ | head -5
else
    echo "   ⚠️  No custom metrics found (monitoring may not be set up)"
fi

# Check alert policies
echo ""
echo "   Checking alert policies..."
ALERT_COUNT=$(gcloud alpha monitoring policies list --project=$PROJECT_ID --format="value(displayName)" 2>/dev/null | grep -c "RAG Service" || echo "0")
if [ "$ALERT_COUNT" -gt 0 ]; then
    echo "   ✅ Found $ALERT_COUNT alert policies"
    gcloud alpha monitoring policies list --project=$PROJECT_ID --format="table(displayName,enabled)" 2>/dev/null | grep "RAG Service" | head -5
else
    echo "   ⚠️  No alert policies found (monitoring may not be set up)"
fi

# Check dashboard
echo ""
echo "   Checking dashboard..."
DASHBOARD_EXISTS=$(gcloud monitoring dashboards list --project=$PROJECT_ID --format="value(displayName)" 2>/dev/null | grep -c "RAG Service" || echo "0")
if [ "$DASHBOARD_EXISTS" -gt 0 ]; then
    echo "   ✅ Dashboard exists"
    echo "   View at: https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
else
    echo "   ⚠️  Dashboard not found (monitoring may not be set up)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Testing Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Service URL: $SERVICE_URL"
echo "🔗 Monitoring Dashboard: https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
echo "🔗 Cloud Run Console: https://console.cloud.google.com/run?project=$PROJECT_ID"
echo ""

