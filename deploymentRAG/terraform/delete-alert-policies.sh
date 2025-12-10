#!/bin/bash
# Script to delete RAG Service alert policies before metrics are destroyed
# This prevents "metric still used in alerting policy" errors

set -e

PROJECT_ID="${TF_VAR_project_id:-medscanai-476500}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Deleting RAG Service alert policies before metrics..."
echo "Project: $PROJECT_ID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get all RAG Service alert policies
POLICIES=$(gcloud alpha monitoring policies list \
  --project="$PROJECT_ID" \
  --filter="displayName~'RAG Service'" \
  --format="value(name)" 2>/dev/null || echo "")

if [ -n "$POLICIES" ]; then
  echo ""
  echo "Found alert policies to delete:"
  echo "$POLICIES" | while read POLICY; do
    if [ -n "$POLICY" ]; then
      echo "  - $POLICY"
    fi
  done
  echo ""
  
  echo "$POLICIES" | while read POLICY; do
    if [ -n "$POLICY" ]; then
      echo "Deleting alert policy: $POLICY"
      gcloud alpha monitoring policies delete "$POLICY" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || echo "  (may have already been deleted or not found)"
    fi
  done
  
  echo ""
  echo "Alert policies deleted. Waiting 15 seconds for GCP to process deletions..."
  sleep 15
  echo "Ready to delete metrics"
else
  echo "No RAG Service alert policies found to delete"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

