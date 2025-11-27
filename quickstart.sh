#!/bin/bash
# Quick Start Script for MedScan AI

set -e

echo "================================================"
echo "MedScan AI - Quick Start"
echo "================================================"
echo ""

# Check credentials
if [ ! -f ~/gcp-service-account.json ]; then
    echo "ERROR: GCP credentials not found"
    echo ""
    echo "Please download service account JSON from GCP Console and save as:"
    echo "  ~/gcp-service-account.json"
    echo ""
    exit 1
fi

echo "GCP credentials found"
echo ""

# Check .env file
if [ ! -f airflow/.env ]; then
    echo "Creating .env file from template..."
    
    # Generate Fernet key
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    
    cat > airflow/.env << EOF
# Airflow Configuration
AIRFLOW__CORE__FERNET_KEY=${FERNET_KEY}
AIRFLOW__WEBSERVER__SECRET_KEY=${SECRET_KEY}
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow

# Airflow User
AIRFLOW_USERNAME=airflow
AIRFLOW_PASSWORD=airflow123
AIRFLOW_FIRSTNAME=Admin
AIRFLOW_LASTNAME=User
AIRFLOW_EMAIL=admin@medscan.ai

# GCP Configuration (use environment variables if set, otherwise use defaults)
GCP_PROJECT_ID=${GCP_PROJECT_ID:-medscanai-476203}
GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-medscan-data}

# Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_MAIL_FROM=
ALERTS_ENABLED=true
ALERT_EMAIL_RECIPIENTS=harshitha8.shekar@gmail.com,kothari.sau@northeastern.edu

# Pipeline Thresholds
VISION_MAX_ANOMALY_PCT=25.0
RAG_SCRAPING_MIN_SUCCESS=0.7
EOF
    
    echo ".env file created"
    echo ""
fi

# Build and start
echo "Building and starting containers..."
echo ""
cd airflow
docker-compose down -v 2>/dev/null || true
docker-compose build
docker-compose up -d

echo ""
echo "Waiting for initialization to complete..."
sleep 10

# Follow logs
echo ""
echo "================================================"
echo "Initialization Logs"
echo "================================================"
docker-compose logs airflow-init

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Access Airflow UI:"
echo "  URL: http://localhost:8080"
echo "  Username: airflow"
echo "  Password: airflow123"
echo ""
echo "Available DAGs:"
echo "  - medscan_vision_pipeline_dvc"
echo "  - rag_data_pipeline_dvc"
echo ""
echo "Verify setup:"
echo "  docker-compose exec webserver bash"
echo "  cd /opt/airflow/DataPipeline"
echo "  dvc status"
echo ""