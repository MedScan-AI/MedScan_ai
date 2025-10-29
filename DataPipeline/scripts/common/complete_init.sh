#!/bin/bash
# Complete Initialization Script for MedScan AI
# Runs GCS setup and DVC initialization

echo "================================================"
echo "MedScan AI - Complete Initialization"
echo "================================================"
echo ""

# Check for GCP credentials
if [ ! -f /opt/airflow/gcp-service-account.json ]; then
    echo "WARNING: GCP credentials not found at /opt/airflow/gcp-service-account.json"
    echo "Skipping GCS and DVC setup"
    exit 0
fi

echo "Step 1: GCS Bucket Setup"
echo "------------------------"
cd /opt/airflow/DataPipeline
python scripts/common/auto_setup_gcs.py || {
    echo "WARNING: GCS setup failed (non-critical)"
}
echo ""

echo "Step 2: DVC Initialization"
echo "-------------------------"
python scripts/common/init_dvc.py || {
    echo "WARNING: DVC init failed (non-critical)"
}
echo ""

echo "================================================"
echo "Initialization Complete"
echo "================================================"
echo ""