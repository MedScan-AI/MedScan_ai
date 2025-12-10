# MedScan AI

AI-powered radiological assistant for CT scan analysis with explainable AI and patient engagement RAG (Retrieval-Augmented Generation) to answer questions about medical report content.

## Table of Contents

- [About MedScan AI](#about-medscan-ai)
- [Features](#features)
- [Project Stages](#project-stages)
- [Architecture](#architecture)
- [Setup Instructions](#setup-instructions)
- [CI/CD Pipeline](#cicd-pipeline)
- [Manual Execution](#manual-execution)
- [Monitoring & Alerts](#monitoring--alerts)
- [Documentation](#documentation)

---

## About MedScan AI

MedScan AI is an end-to-end MLOps platform for medical image analysis and patient engagement. It combines:

- **Vision Models**: Deep learning models for medical image classification (Tuberculosis detection, Lung Cancer detection)
- **RAG System**: Retrieval-Augmented Generation for answering patient questions about medical reports
- **Data Pipeline**: Automated data acquisition, preprocessing, validation, and bias detection
- **Model Training**: Automated model selection, hyperparameter optimization, and deployment
- **Monitoring**: Real-time performance monitoring with automated retraining triggers

**Project Scope**: See [docs/Scoping.pdf](docs/Scoping.pdf)

---

## Features

### Data Pipeline Capabilities

- ✅ **Automated Data Acquisition**: Fetches datasets from Kaggle with date partitioning
- ✅ **Image Preprocessing**: Standardizes images to 224x224 JPEG format
- ✅ **Synthetic Metadata Generation**: Creates realistic patient demographics using Faker
- ✅ **Comprehensive Validation**: Schema validation, drift detection, anomaly detection
- ✅ **Bias Detection & Mitigation**: Advanced fairness analysis using Fairlearn, SliceFinder, TFMA
- ✅ **Data Versioning**: DVC integration with GCS for reproducibility
- ✅ **RAG Knowledge Base**: Medical article scraping, chunking, embedding, and FAISS indexing
- ✅ **Email Alerts**: Automated notifications for anomalies, drift, and bias

### Model Training & Deployment

- ✅ **Automated Model Training**: Cloud Build pipelines for Vision and RAG models
- ✅ **Model Selection**: Hyperparameter optimization (Optuna) and multi-architecture comparison
- ✅ **Model Validation**: Performance threshold checks before deployment
- ✅ **Bias Detection**: Comprehensive bias checks during model training
- ✅ **Endpoint Deployment**: Automatic deployment to Vertex AI endpoints (Vision) and Cloud Run (RAG)
- ✅ **MLflow Tracking**: Experiment tracking and model versioning
- ✅ **Email Notifications**: Training completion, validation failures, bias violations

### Monitoring & Observability

- ✅ **Real-time Monitoring**: Performance metrics, latency tracking, quality scores
- ✅ **Automated Retraining**: Intelligent retraining triggers based on performance degradation
- ✅ **Monitoring Dashboards**: GCP Monitoring dashboards with custom metrics
- ✅ **Alert Policies**: 11 alert policies for production and quality issues
- ✅ **Custom Metrics**: 7 log-based metrics for RAG quality tracking

---

## Project Stages

The MedScan AI pipeline consists of **5 main stages**:

### Stage 1: Data Pipeline

**Purpose**: Acquire, preprocess, validate, and prepare data for training

**Components**:
- **Vision Pipeline**: Medical image data (TB X-rays, Lung Cancer CT scans)
- **RAG Pipeline**: Medical knowledge base (scraped articles, research papers)

**Outputs**:
- Preprocessed images (224x224 JPEG)
- Synthetic patient metadata
- Validated datasets with bias mitigation
- RAG knowledge base (embeddings, FAISS index)

**Location**: `DataPipeline/`

### Stage 2: Model Training

**Purpose**: Train and select best models for Vision and RAG

**Components**:
- **Vision Training**: ResNet, ViT, Custom CNN architectures
- **RAG Training**: Model selection with 7 LLM models, 4 retrieval strategies

**Outputs**:
- Trained models in Vertex AI Model Registry
- Model performance metrics
- Bias detection reports
- MLflow experiment tracking

**Location**: `ModelDevelopment/`

### Stage 3: Model Deployment

**Purpose**: Deploy models to production endpoints

**Components**:
- **Vision Inference**: Vertex AI endpoint deployment
- **RAG Service**: Cloud Run service with GPU support

**Outputs**:
- Production endpoints (Vertex AI, Cloud Run)
- Service URLs and health checks
- Deployment configurations

**Location**: `deployment/`, `deploymentVisionInference/`, `deploymentRAG/`

### Stage 4: Monitoring Setup

**Purpose**: Set up monitoring infrastructure and dashboards

**Components**:
- Custom log-based metrics
- Monitoring dashboards
- Alert policies
- Notification channels

**Outputs**:
- GCP Monitoring dashboard
- 11 alert policies
- 7 custom metrics

**Location**: `deploymentRAG/terraform/`, `Monitoring/`

### Stage 5: Continuous Monitoring

**Purpose**: Monitor production models and trigger retraining

**Components**:
- Performance metric collection
- Drift detection
- Quality score tracking
- Automated retraining triggers

**Outputs**:
- Monitoring reports
- Retraining decisions
- Performance analytics

**Location**: `Monitoring/RAG/`

---

## Architecture

![High-Level Architecture](assets/high_level_architecture.png)

### Key Components

1. **Data Pipeline** (Airflow): Automated data processing and validation
2. **Model Training** (Cloud Build): Automated model training and selection
3. **Model Deployment** (Vertex AI, Cloud Run): Production inference endpoints
4. **Monitoring** (GCP Monitoring): Real-time performance tracking
5. **CI/CD** (GitHub Actions): Automated workflows for training and deployment

---

## Setup Instructions

> **Note**: Naming conventions (project IDs, bucket names, service names, etc.) can be configured at the time of setup. Throughout this documentation, placeholders like `<YOUR-GCP_PROJECT_ID>` indicate values that should be customized for your environment.

### Prerequisites

- **Python 3.10+**
- **Docker Desktop**
- **Google Cloud Platform Account**
- **Kaggle Account** (for dataset access)
- **Git**
- **HF token with access to the models**

**Note**: Naming conventions (project IDs, bucket names, service names) can be configured at the time of setup. See configuration sections below for details.

### 1. Clone Repository

```bash
git clone https://github.com/MedScan-AI/MedScan_ai.git
cd MedScan_ai
```

### 2. Setup GCP Credentials

1. Go to **IAM & Admin → Service Accounts** in GCP Console
2. Select your service account (or create one)
3. Click **Keys → Add Key → Create New Key → JSON**
4. Save the file as `~/gcp-service-account.json`

**Required GCP Permissions**:
- Storage Admin
- Storage Object Admin
- Vertex AI User
- Cloud Run Admin
- Monitoring Admin
- Secret Manager Secret Accessor

**Note**: Naming conventions (project IDs, bucket names, service names) can be configured at the time of setup. Default values in this documentation use placeholders like `<YOUR-GCP_PROJECT_ID>` - replace these with your actual values.

### 3. Setup Kaggle Credentials

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Download kaggle.json from kaggle.com/settings
# Place it in ~/.kaggle/
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Configure GitHub Secrets

Go to **GitHub Repository → Settings → Secrets and variables → Actions** and add:

- `GCP_SA_KEY`: Service account JSON key (for GitHub Actions)
- `SMTP_USER`: Email for notifications (optional)
- `SMTP_PASSWORD`: Email app password (optional)

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create `airflow/.env` file:

```bash
cd airflow

# Generate Fernet key
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Create .env file
cat > .env << 'EOF'
# Airflow Configuration
AIRFLOW__CORE__FERNET_KEY=<your-generated-fernet-key>
AIRFLOW__WEBSERVER__SECRET_KEY=<random-secret-key>
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
AIRFLOW_UID=50000
AIRFLOW_GID=0
AIRFLOW_USERNAME=airflow
AIRFLOW_PASSWORD=airflow123

# GCP Configuration
# Note: Naming conventions can be configured at setup time
GCP_PROJECT_ID=<YOUR-GCP_PROJECT_ID>
GCS_BUCKET_NAME=medscan-pipeline-<YOUR-GCP_PROJECT_ID>  # Can be customized
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/gcp-service-account.json

# Email Alerts (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_RECIPIENTS=your-email@example.com
EOF
```

### 7. Initialize Docker Environment

```bash
# Make scripts executable
chmod +x quickstart.sh

# Run quickstart (builds images and initializes everything)
./quickstart.sh
```

**OR manually**:

```bash
cd airflow
docker-compose build
docker-compose up -d
docker-compose logs -f airflow-init
```

### 8. Verify Setup

```bash
# Check containers are running
docker-compose ps

# Verify GCS bucket
gsutil ls gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/

# Access Airflow UI
open http://localhost:8080
# Login: airflow / airflow123
```

---

## CI/CD Pipeline

The project uses **GitHub Actions** for automated CI/CD. All workflows are located in `.github/workflows/`.

### Automated Workflows

#### 1. **RAG Data Pipeline** (`rag-data-pipeline.yaml`)

**Triggers**:
- Manual dispatch
- Push to `main` (when `DataPipeline/scripts/RAG/**` changes)

**What it does**:
- Processes RAG data (scraping, chunking, embedding, indexing)
- Validates data quality
- Uploads to GCS

**Status**: ✅ Automated

#### 2. **RAG Model Training** (`rag-training.yaml`)

**Triggers**:
- Push to `main` (when `ModelDevelopment/RAG/**` or `cloudbuild/rag-training.yaml` changes)
- Pull requests (validation only)
- Manual dispatch

**What it does**:
- Verifies RAG data exists in GCS
- Triggers Cloud Build for model selection
- Runs hyperparameter optimization
- Selects best model
- Sends email notifications

**Status**: ✅ Automated

#### 3. **RAG Complete Setup** (`rag-complete-setup.yaml`)

**Triggers**:
- Manual dispatch (recommended)
- Push to `main` (when deployment files change)

**What it does**:
- Deploys Cloud Run service (`rag-service`)
- Sets up monitoring infrastructure (dashboard, alerts, metrics)
- Tests service endpoints

**Status**: ✅ Automated (one-time setup)

#### 4. **RAG Deployment** (`rag-deploy.yaml`)

**Triggers**:
- After `rag-training.yaml` completes successfully
- Manual dispatch
- Push to `rag_deployment_monitoring` branch

**What it does**:
- Deploys/updates Cloud Run service
- Tests health, config, and prediction endpoints
- Sets IAM permissions

**Status**: ✅ Automated

#### 5. **RAG Monitoring** (`rag-monitoring.yaml`)

**Triggers**:
- Scheduled (every 6 hours)
- Manual dispatch

**What it does**:
- Runs monitoring checks (Python script)
- Collects metrics from Cloud Logging
- Analyzes performance and quality
- Optionally triggers retraining

**Status**: ✅ Automated (scheduled)

#### 6. **RAG Retraining** (`rag-retraining.yaml`)

**Triggers**:
- After `rag-data-pipeline.yaml` completes
- Manual dispatch

**What it does**:
- Retrains RAG models
- Can use existing data or trigger data pipeline first

**Status**: ✅ Automated

#### 7. **Vision Model Training** (`vision-training.yaml`)

**Triggers**:
- Push to `main` (when `ModelDevelopment/Vision/**` or `cloudbuild/vision-training.yaml` changes)
- Pull requests (validation only)
- Manual dispatch

**What it does**:
- Triggers Cloud Build for vision model training
- Trains multiple architectures (ResNet, ViT, Custom CNN)
- Runs hyperparameter optimization
- Deploys to Vertex AI
- Sends email notifications

**Status**: ✅ Automated

#### 8. **Vision Inference Deployment** (`vision-inference-deploy.yaml`)

**Triggers**:
- After `vision-training.yaml` completes successfully
- Manual dispatch

**What it does**:
- Deploys vision inference API to Cloud Run
- Tests endpoints

**Status**: ✅ Automated

#### 9. **Vision Inference Terraform Setup** (`vision-inference-terraform-setup.yaml`)

**Triggers**:
- Manual dispatch only

**What it does**:
- Sets up infrastructure via Terraform
- Creates Cloud Run service, Artifact Registry, IAM bindings

**Status**: ⚠️ Manual (one-time setup)

### Workflow Dependencies

```
Data Pipeline → Model Training → Deployment → Monitoring
     ↓              ↓                ↓            ↓
rag-data-pipeline  rag-training  rag-deploy  rag-monitoring
     ↓              ↓                ↓            ↓
rag-retraining  (auto-triggers)  (auto-triggers)  (scheduled)
```

### Workflow Alerts

All workflows send email notifications for:
- ✅ **Success**: Training/deployment completed successfully
- ❌ **Failure**: Pipeline failures, validation errors
- ⚠️ **Warnings**: Bias violations, performance degradation

**Email Recipients**: Configured in GitHub Secrets (`SMTP_USER`, `SMTP_PASSWORD`)

---

## Manual Execution

If you need to run stages manually (bypassing CI/CD), follow these instructions:

### Stage 1: Data Pipeline

#### Vision Data Pipeline

**Via Airflow UI**:
1. Navigate to http://localhost:8080
2. Find DAG: `medscan_vision_pipeline_dvc`
3. Toggle **ON**
4. Click **Trigger DAG**

**Via CLI**:
```bash
docker-compose exec webserver airflow dags trigger medscan_vision_pipeline_dvc
```

**Manual Script Execution**:
```bash
cd DataPipeline

# 1. Download data
python scripts/data_acquisition/fetch_data.py --config config/vision_pipeline.yml

# 2. Preprocess images
python scripts/data_preprocessing/process_tb.py --config config/vision_pipeline.yml
python scripts/data_preprocessing/process_lungcancer.py --config config/vision_pipeline.yml

# 3. Generate synthetic metadata
python scripts/data_preprocessing/baseline_synthetic_data_generator.py --config config/synthetic_data.yml

# 4. Validate data
python scripts/data_preprocessing/schema_statistics.py --config config/metadata.yml
```

#### RAG Data Pipeline

**Via Airflow UI**:
1. Navigate to http://localhost:8080
2. Find DAG: `rag_data_pipeline_dvc`
3. Toggle **ON**
4. Click **Trigger DAG**

**Via CLI**:
```bash
docker-compose exec webserver airflow dags trigger rag_data_pipeline_dvc
```

**Manual Script Execution**:
```bash
cd DataPipeline
python scripts/RAG/main.py
```

### Stage 2: Model Training

#### Vision Model Training

**Via Cloud Build**:
```bash
# Full training run - Tuberculosis
gcloud builds submit \
  --config=cloudbuild/vision-training.yaml \
  --substitutions=_DATASET=tb,_EPOCHS=50 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --region=us-central1

# Full training run - Lung Cancer
gcloud builds submit \
  --config=cloudbuild/vision-training.yaml \
  --substitutions=_DATASET=lung_cancer_ct_scan,_EPOCHS=50 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --region=us-central1
```

**Local Training**:
```bash
cd ModelDevelopment/Vision

# Train ResNet
python train_resnet.py --config ../config/vision_training.yml

# Train ViT
python train_vit.py --config ../config/vision_training.yml

# Train Custom CNN
python train_custom_cnn.py --config ../config/vision_training.yml
```

#### RAG Model Training

**Via Cloud Build**:
```bash
gcloud builds submit \
  --config=cloudbuild/rag-training.yaml \
  --project=<YOUR-GCP_PROJECT_ID> \
  --region=us-central1
```

**Local Training**:
```bash
cd ModelDevelopment/RAG

# Run experiments
python ModelSelection/experiment.py

# Select best model
python ModelSelection/select_best_model.py
```

### Stage 3: Model Deployment

#### RAG Service Deployment

**One-Stop Setup** (Recommended):
```bash
# Via GitHub Actions UI:
# 1. Go to Actions → "RAG Complete Setup - Cloud Run + Monitoring"
# 2. Click "Run workflow"
# 3. Configure inputs (monitoring_email, auto_approve_terraform)
# 4. Click "Run workflow"
```

**Manual Deployment**:
```bash
# Deploy Cloud Run service
gcloud builds submit \
  --config=cloudbuild/rag-deploy.yaml \
  --project=<<YOUR-GCP_PROJECT_ID>>

# Setup monitoring (Terraform)
cd deploymentRAG/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings
terraform init
terraform plan
terraform apply
```

#### Vision Inference Deployment

**Via GitHub Actions**:
1. Go to Actions → "Vision Inference - Terraform Setup"
2. Click "Run workflow" → action: `apply` → auto_approve: `true`
3. Then run "Vision Inference Deployment" workflow

**Manual Deployment**:
```bash
# Setup infrastructure
cd deploymentVisionInference/terraform
terraform init
terraform apply

# Deploy service
gcloud builds submit \
  --config=ModelDevelopment/VisionInference/cloudbuild.yaml \
  --project=<YOUR-GCP_PROJECT_ID>
```

### Stage 4: Monitoring Setup

**Automated** (via `rag-complete-setup.yaml`):
- Creates monitoring dashboard
- Sets up 11 alert policies
- Creates 7 custom metrics
- Configures email notifications

**Manual Setup**:
```bash
cd deploymentRAG/terraform
terraform init
terraform apply
```

### Stage 5: Continuous Monitoring

**Automated** (scheduled every 6 hours):
- Runs via `rag-monitoring.yaml` workflow
- Collects metrics from Cloud Logging
- Analyzes performance
- Triggers retraining if needed

**Manual Execution**:
```bash
cd Monitoring/RAG
python run_monitoring.py --hours=24

# With retraining trigger
python run_monitoring.py --hours=24 --trigger-retrain
```

---

## Monitoring & Alerts

### Monitoring Dashboard

**Location**: GCP Console → Monitoring → Dashboards → "RAG Service - Monitoring Dashboard"

**URL**: https://console.cloud.google.com/monitoring/dashboards?project=<YOUR-GCP_PROJECT_ID>

**Widgets**:
- Request Rate, Error Rate, Latency (P50/P95/P99)
- CPU/Memory Utilization, Instance Count
- Composite Score Distribution
- Answer Groundedness Trend
- Retrieval Score Trend
- Low Quality Predictions Count
- Tokens Usage

### Alert Policies

**11 Alert Policies** configured:

**Production Alerts (5)**:
1. High Error Rate (>15% for 5 min)
2. High Latency (P95 >10s for 5 min)
3. Service Unavailable (no requests in 5h)
4. High CPU (>80% for 5 min)
5. High Memory (>85% for 5 min)

**Quality Alerts (5)**:
6. Low Composite Score (avg <0.3 for 15 min)
7. Critical Composite Score (min <0.25 for 5 min)
8. Low Hallucination Score (avg <0.2 for 15 min)
9. Low Retrieval Quality (avg <0.6 for 15 min)
10. Low Quality Prediction Spike (>10 in 5 min)

**Retraining Alert (1)**:
11. Retraining Triggered (immediate notification)

### Custom Metrics

**7 Custom Log-Based Metrics**:
1. `rag_composite_score` - Composite quality score (0-1)
2. `rag_hallucination_score` - Hallucination score (0-1, higher is better)
3. `rag_retrieval_score` - Average retrieval score (0-1)
4. `rag_low_composite_score` - Count of low-quality predictions
5. `rag_tokens_used` - Total tokens consumed
6. `rag_docs_retrieved` - Number of documents retrieved
7. `rag_retraining_triggered` - Retraining event count

### Email Notifications

Alerts are sent to email addresses configured in:
- **Terraform**: `deploymentRAG/terraform/terraform.tfvars` → `monitoring_email`
- **GitHub Actions**: Workflow inputs → `monitoring_email`

**Alert Types**:
- Data pipeline failures
- Validation issues (anomalies, drift, bias)
- Model training completion/failures
- Deployment status
- Monitoring alerts (error rate, latency, quality degradation)
- Retraining triggers

---

## Documentation

All detailed documentation is located in the `docs/` folder:

- **[Project Structure](docs/PROJECT_STRUCTURE.md)**: Detailed project structure and file organization
- **[Model Monitoring Audit](docs/MODEL_MONITORING_AUDIT_REPORT.md)**: Monitoring and retraining readiness audit
- **[DVC Documentation](docs/DVC.md)**: Data versioning with DVC
- **[Run Pipeline Guide](docs/RUN_PIPELINE.md)**: Step-by-step pipeline execution guide
- **[Vision Inference README](docs/README-vision-inference.md)**: Vision inference deployment guide
- **[Testing Guide](docs/TESTING-GUIDE.md)**: Testing guide for RAG complete setup
- **[GCP Setup Steps](docs/verify_gcp_setup_STEPS.md)**: GCP setup verification steps
- **[Scoping Document](docs/Scoping.pdf)**: Project scoping and requirements

### Component-Specific Documentation

- **Data Pipeline**: `DataPipeline/README.md`
- **RAG Model Development**: `ModelDevelopment/RAG/README.md`
- **Vision Model Development**: `ModelDevelopment/Vision/README.md`
- **Airflow Setup**: `airflow/README.md`
- **RAG Terraform**: `deploymentRAG/terraform/README.md`
- **Vision Terraform**: `deploymentVisionInference/terraform/README.md`

---

## Quick Reference

### Key URLs

- **Airflow UI**: http://localhost:8080 (local) or GCP Composer (production)
- **MLflow UI**: http://localhost:5000 (local)
- **GCP Console**: https://console.cloud.google.com/?project=<YOUR-GCP_PROJECT_ID>
- **Monitoring Dashboard**: https://console.cloud.google.com/monitoring/dashboards?project=<YOUR-GCP_PROJECT_ID>
- **Cloud Build**: https://console.cloud.google.com/cloud-build?project=<YOUR-GCP_PROJECT_ID>
- **Vertex AI Models**: https://console.cloud.google.com/vertex-ai/models?project=<YOUR-GCP_PROJECT_ID>
- **Cloud Run**: https://console.cloud.google.com/run?project=<YOUR-GCP_PROJECT_ID>

### Key Commands

```bash
# Start Airflow
cd airflow && docker-compose up -d

# Trigger Vision Pipeline
docker-compose exec webserver airflow dags trigger medscan_vision_pipeline_dvc

# Trigger RAG Pipeline
docker-compose exec webserver airflow dags trigger rag_data_pipeline_dvc

# Test RAG Deployment
./scripts/test-rag-deployment.sh

# Run Monitoring
cd Monitoring/RAG && python run_monitoring.py --hours=24
```

### Project Structure

```
MedScan_ai/
├── README.md                    # This file
├── DataPipeline/               # Data processing and validation
├── ModelDevelopment/           # Model training and selection
├── deployment/                  # Deployment configurations
├── deploymentRAG/               # RAG deployment (Terraform)
├── deploymentVisionInference/  # Vision deployment (Terraform)
├── Monitoring/                  # Monitoring scripts
├── airflow/                     # Airflow orchestration
├── cloudbuild/                  # Cloud Build configurations
├── scripts/                     # Utility scripts
└── docs/                        # Documentation files
```

---

## Support

For issues or questions:
1. Check component-specific README files
2. Review documentation in `docs/` folder
3. Check GitHub Actions workflow logs
4. Review GCP Console logs and monitoring

---

## License

TBD

---

**Last Updated**: December 2024  
**Maintained By**: MedScan AI Development Team
