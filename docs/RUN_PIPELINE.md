# How to Run MedScan AI Pipelines

This comprehensive guide covers all pipeline stages for both **Vision** and **RAG** components of MedScan AI. It provides step-by-step instructions for data processing, model training, deployment, monitoring setup, and continuous monitoring.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Stage 1: Data Pipeline](#stage-1-data-pipeline)
   - [Vision Data Pipeline](#vision-data-pipeline)
   - [RAG Data Pipeline](#rag-data-pipeline)
3. [Stage 2: Model Training](#stage-2-model-training)
   - [Vision Model Training](#vision-model-training)
   - [RAG Model Training](#rag-model-training)
4. [Stage 3: Model Deployment](#stage-3-model-deployment)
   - [Vision Inference Deployment](#vision-inference-deployment)
   - [RAG Service Deployment](#rag-service-deployment)
5. [Stage 4: Monitoring Setup](#stage-4-monitoring-setup)
   - [RAG Monitoring Infrastructure](#rag-monitoring-infrastructure)
   - [Vision Monitoring Infrastructure](#vision-monitoring-infrastructure)
6. [Stage 5: Continuous Monitoring](#stage-5-continuous-monitoring)
7. [Verification and Testing](#verification-and-testing)
8. [Troubleshooting](#troubleshooting)
9. [Cleanup](#cleanup)

---

## Prerequisites

### GCP Setup

- **Google Cloud CLI** installed and authenticated
- **Project configured**: `gcloud config set project <YOUR-GCP_PROJECT_ID>`
- **Region**: `us-central1` (default)
- **Bucket**: `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/`
- **APIs enabled**:
  - Cloud Build
  - Vertex AI
  - Cloud Storage
  - Artifact Registry
  - Secret Manager
  - Cloud Run
  - Monitoring API
  - Logging API

### GitHub Actions (Optional)

If using automated workflows via GitHub Actions:

- **GitHub Secrets configured**:
  - `GCP_SA_KEY`: Service account JSON key
  - `SMTP_USER`: Email for notifications (optional)
  - `SMTP_PASSWORD`: Email app password (optional)
- **Repository access**: Push permissions to trigger workflows

### Component-Specific Requirements

**Vision Component**:
- Preprocessed image data in GCS: `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/vision/`
- Kaggle credentials configured (for data acquisition)

**RAG Component**:
- HuggingFace token in Secret Manager: `huggingface-token`
- RAG data pipeline completed (embeddings and index files in GCS)

### Verify GCP Setup

Run the verification script to ensure everything is configured correctly:

```bash
bash scripts/verify_gcp_setup.sh
```

Confirm that:
- Project and bucket resolve correctly
- Required APIs are enabled
- You have access to Storage and Vertex AI
- Service account has required permissions

---

## Stage 1: Data Pipeline

### Vision Data Pipeline

The Vision data pipeline processes medical image data (TB X-rays, Lung Cancer CT scans) through acquisition, preprocessing, validation, and bias detection.

#### Method 1: Cloud Build (Recommended for CI/CD)

```bash
gcloud builds submit \
  --config=cloudbuild/vision-data-pipeline.yaml \
  --project=<YOUR-GCP_PROJECT_ID> \
  --region=us-central1
```

**Note**: If `vision-data-pipeline.yaml` doesn't exist, use Airflow or manual execution methods below.

#### Method 2: GitHub Actions Workflow

1. Navigate to **Actions** → **Vision Data Pipeline**
2. Click **Run workflow**
3. Select branch: `main`
4. Click **Run workflow**

**Workflow**: `.github/workflows/vision-data-pipeline.yaml` (if exists)

#### Method 3: Airflow DAG

**Via Airflow UI**:
1. Navigate to http://localhost:8080
2. Find DAG: `medscan_vision_pipeline_dvc`
3. Toggle **ON**
4. Click **Trigger DAG**

**Via CLI**:
```bash
docker-compose exec webserver airflow dags trigger medscan_vision_pipeline_dvc
```

#### Method 4: Manual Script Execution

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

#### Verification

After pipeline completion, verify:

```bash
# Check preprocessed data in GCS
gsutil ls gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/vision/preprocessed/

# Check DVC data
cd DataPipeline
dvc status
dvc list data/preprocessed/
```

**Expected outputs**:
- Preprocessed images (224x224 JPEG) in `data/preprocessed/`
- Synthetic metadata files
- Validation reports in GCS

---

### RAG Data Pipeline

The RAG data pipeline scrapes medical articles, processes them into chunks, generates embeddings, and creates a FAISS index for retrieval.

#### Method 1: GitHub Actions Workflow

1. Navigate to **Actions** → **RAG Data Pipeline**
2. Click **Run workflow** → **workflow_dispatch**
3. Select branch: `main`
4. Click **Run workflow**

**Workflow**: `.github/workflows/rag-data-pipeline.yaml`

**Triggers**:
- Manual dispatch (`workflow_dispatch`)
- Push to `main` (when `DataPipeline/scripts/RAG/**` changes)

#### Method 2: Airflow DAG

**Via Airflow UI**:
1. Navigate to http://localhost:8080
2. Find DAG: `rag_data_pipeline_dvc`
3. Toggle **ON**
4. Click **Trigger DAG**

**Via CLI**:
```bash
docker-compose exec webserver airflow dags trigger rag_data_pipeline_dvc
```

#### Method 3: Manual Script Execution

```bash
cd DataPipeline
python scripts/RAG/main.py
```

This script executes:
- URL scraping from GCS
- Article content extraction
- Text chunking
- Embedding generation
- FAISS index creation
- Upload to GCS

#### Verification

After pipeline completion, verify:

```bash
# Check RAG data in GCS
gsutil ls gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/RAG/

# Expected files:
# - embeddings/ (embedding vectors)
# - index/ (FAISS index files)
# - chunks/ (processed text chunks)
# - metadata/ (pipeline metadata)
```

**Expected outputs**:
- Embedding files in `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/RAG/embeddings/`
- FAISS index in `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/RAG/index/`
- Chunked documents in `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/RAG/chunks/`

---

## Stage 2: Model Training

### Vision Model Training

Train multiple vision architectures (ResNet, ViT, Custom CNN) with hyperparameter optimization and bias detection.

#### Method 1: Cloud Build (Recommended)

**Full training run - Tuberculosis**:
```bash
gcloud builds submit \
  --config=cloudbuild/vision-training.yaml \
  --substitutions=_DATASET=tb,_EPOCHS=50 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --region=us-central1
```

**Full training run - Lung Cancer**:
```bash
gcloud builds submit \
  --config=cloudbuild/vision-training.yaml \
  --substitutions=_DATASET=lung_cancer_ct_scan,_EPOCHS=50 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --region=us-central1
```

**Quick test run** (for iteration):
```bash
gcloud builds submit \
  --config=cloudbuild/vision-training.yaml \
  --substitutions=_DATASET=tb,_EPOCHS=3 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --region=us-central1
```

#### Method 2: GitHub Actions Workflow

1. Navigate to **Actions** → **Vision Model Training**
2. Click **Run workflow**
3. Configure inputs:
   - `dataset`: `tb` or `lung_cancer_ct_scan`
   - `epochs`: `50` (or `3` for quick test)
4. Click **Run workflow**

**Workflow**: `.github/workflows/vision-training.yaml`

**Triggers**:
- Push to `main` (when `ModelDevelopment/Vision/**` or `cloudbuild/vision-training.yaml` changes)
- Pull requests (validation only)
- Manual dispatch

#### Method 3: Local Training

```bash
cd ModelDevelopment/Vision

# Train ResNet
python train_resnet.py --config ../config/vision_training.yml

# Train ViT
python train_vit.py --config ../config/vision_training.yml

# Train Custom CNN
python train_custom_cnn.py --config ../config/vision_training.yml
```

#### Monitoring Training

**Via Cloud Build Console**:
- Navigate to: https://console.cloud.google.com/cloud-build?project=<YOUR-GCP_PROJECT_ID>
- Select the running build to stream logs

**Via CLI**:
```bash
# List latest builds
gcloud builds list --project=<YOUR-GCP_PROJECT_ID> --region=us-central1 --limit=1

# Stream logs
gcloud builds log <BUILD_ID> --project=<YOUR-GCP_PROJECT_ID> --region=us-central1
```

#### Artifacts and Outputs

**Trained models**:
- Location: `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/vision/trained_models/${BUILD_ID}/`
- Format: TensorFlow SavedModel

**Validation results**:
- Location: `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/vision/validation/${BUILD_ID}/validation_results.json`

**Vertex AI Model Registry**:
- Models: https://console.cloud.google.com/vertex-ai/models?project=<YOUR-GCP_PROJECT_ID>
- Endpoints: https://console.cloud.google.com/vertex-ai/endpoints?project=<YOUR-GCP_PROJECT_ID>

---

### RAG Model Training

Train and select the best RAG model from 7 LLM models and 4 retrieval strategies using hyperparameter optimization.

#### Method 1: Cloud Build

```bash
gcloud builds submit \
  --config=cloudbuild/rag-training.yaml \
  --project=<YOUR-GCP_PROJECT_ID> \
  --region=us-central1
```

#### Method 2: GitHub Actions Workflow

1. Navigate to **Actions** → **RAG Model Training**
2. Click **Run workflow**
3. Select branch: `main`
4. Click **Run workflow**

**Workflow**: `.github/workflows/rag-training.yaml`

**Triggers**:
- Push to `main` (when `ModelDevelopment/RAG/**` or `cloudbuild/rag-training.yaml` changes)
- Pull requests (validation only)
- Manual dispatch

**What it does**:
- Verifies RAG data exists in GCS
- Triggers Cloud Build for model selection
- Runs hyperparameter optimization (Optuna)
- Selects best model based on performance metrics
- Sends email notifications

#### Method 3: Local Training

```bash
cd ModelDevelopment/RAG

# Run experiments
python ModelSelection/experiment.py

# Select best model
python ModelSelection/select_best_model.py
```

#### Model Selection Process

The training pipeline:
1. Tests 7 LLM models (various sizes and architectures)
2. Tests 4 retrieval strategies (dense, sparse, hybrid, reranking)
3. Runs hyperparameter optimization for each combination
4. Evaluates on validation set
5. Selects best model based on composite score (accuracy, latency, cost)

#### Verification

After training, verify:

```bash
# Check model files in GCS
gsutil ls gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/RAG/models/

# Check MLflow experiments (if using)
# Navigate to MLflow UI: http://localhost:5000
```

**Expected outputs**:
- Best model configuration in `ModelDevelopment/RAG/ModelSelection/best_model.json`
- Model artifacts in GCS
- Training metrics and logs

---

## Stage 3: Model Deployment

### Vision Inference Deployment

Deploy the Vision inference API to Cloud Run with Grad-CAM visualization support.

#### Method 1: GitHub Actions Workflow (Recommended)

**Step 1: Terraform Setup** (one-time):
1. Navigate to **Actions** → **Vision Inference - Terraform Setup**
2. Click **Run workflow**
3. Configure:
   - `action`: `apply`
   - `auto_approve`: `true`
4. Click **Run workflow**

**Workflow**: `.github/workflows/vision-inference-terraform-setup.yaml`

**Step 2: Deploy Service**:
1. Navigate to **Actions** → **Vision Inference API - Deploy**
2. Click **Run workflow**
3. Optionally check **Force deployment**
4. Click **Run workflow**

**Workflow**: `.github/workflows/vision-inference-deploy.yaml`

**Triggers**:
- After `vision-training.yaml` completes successfully
- Manual dispatch
- Push to `main` (when `ModelDevelopment/VisionInference/**` changes)

#### Method 2: Manual Deployment

**Setup Infrastructure**:
```bash
cd deploymentVisionInference/terraform

# Copy example config
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your settings
# Then initialize and apply
terraform init
terraform plan
terraform apply
```

**Deploy Service**:
```bash
gcloud builds submit \
  --config=ModelDevelopment/VisionInference/cloudbuild.yaml \
  --project=<YOUR-GCP_PROJECT_ID>
```

#### Verification

After deployment, verify:

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe vision-inference-api \
  --region=us-central1 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --format="value(status.url)")

# Test health endpoint
curl $SERVICE_URL/health

# Test TB endpoint (will return 422 without file, but endpoint is accessible)
curl -X POST "$SERVICE_URL/predict/tb" \
  -H "Content-Type: multipart/form-data"

# Test Lung Cancer endpoint
curl -X POST "$SERVICE_URL/predict/lung_cancer" \
  -H "Content-Type: multipart/form-data"
```

**Expected outputs**:
- Cloud Run service: `vision-inference-api`
- Service URL accessible
- Health endpoint returns 200 OK
- Prediction endpoints functional

---

### RAG Service Deployment

Deploy the RAG service to Cloud Run with GPU support for LLM inference.

#### Method 1: One-Stop Setup (Recommended)

**Via GitHub Actions UI**:
1. Navigate to **Actions** → **RAG Complete Setup - Cloud Run + Monitoring**
2. Click **Run workflow**
3. Configure inputs:
   - `monitoring_email`: Your email (e.g., `your-email@example.com`)
   - `enable_monitoring`: `true` (to set up monitoring infrastructure)
   - `auto_approve_terraform`: `true` (for testing, set to `false` in production)
4. Click **Run workflow**

**Workflow**: `.github/workflows/rag-complete-setup.yaml`

**What it does**:
- Deploys Cloud Run service (`rag-service`)
- Sets up monitoring infrastructure (dashboard, alerts, metrics)
- Tests service endpoints
- Configures IAM permissions

#### Method 2: Separate Deployment Workflow

If you only want to deploy the service (without monitoring):

1. Navigate to **Actions** → **RAG Deployment**
2. Click **Run workflow**
3. Click **Run workflow**

**Workflow**: `.github/workflows/rag-deploy.yaml`

**Triggers**:
- After `rag-training.yaml` completes successfully
- Manual dispatch
- Push to `rag_deployment_monitoring` branch

#### Method 3: Manual Deployment

**Deploy Cloud Run Service**:
```bash
gcloud builds submit \
  --config=cloudbuild/rag-deploy.yaml \
  --project=<YOUR-GCP_PROJECT_ID>
```

**Setup Monitoring** (if not using one-stop setup):
```bash
cd deploymentRAG/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings
terraform init
terraform plan
terraform apply
```

#### Verification

After deployment, verify:

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe rag-service \
  --region=us-central1 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --format="value(status.url)")

# Test health endpoint
curl $SERVICE_URL/health | jq

# Test config endpoint
curl $SERVICE_URL/config | jq

# Test prediction endpoint
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"query":"What is tuberculosis?"}]}' | jq
```

**Expected outputs**:
- Cloud Run service: `rag-service`
- Service URL accessible
- Health endpoint returns `{"ready": true}`
- Config endpoint returns model configuration
- Prediction endpoint returns answers

---

## Stage 4: Monitoring Setup

### RAG Monitoring Infrastructure

Set up comprehensive monitoring for the RAG service with 11 alert policies, 7 custom metrics, and a monitoring dashboard.

#### Automated Setup (Recommended)

**Via `rag-complete-setup.yaml`**:
- Monitoring is automatically set up when you run the complete setup workflow with `enable_monitoring=true`
- Creates all resources in one go

**Resources Created**:
- **7 Custom Log-Based Metrics**:
  1. `rag_composite_score` - Composite quality score (0-1)
  2. `rag_hallucination_score` - Hallucination score (0-1, higher is better)
  3. `rag_retrieval_score` - Average retrieval score (0-1)
  4. `rag_low_composite_score` - Count of low-quality predictions
  5. `rag_tokens_used` - Total tokens consumed
  6. `rag_docs_retrieved` - Number of documents retrieved
  7. `rag_retraining_triggered` - Retraining event count

- **11 Alert Policies**:
  - **Production Alerts (5)**:
    1. High Error Rate (>15% for 5 min)
    2. High Latency (P95 >10s for 5 min)
    3. Service Unavailable (no requests in 5h)
    4. High CPU (>80% for 5 min)
    5. High Memory (>85% for 5 min)
  - **Quality Alerts (5)**:
    6. Low Composite Score (avg <0.3 for 15 min)
    7. Critical Composite Score (min <0.25 for 5 min)
    8. Low Hallucination Score (avg <0.2 for 15 min)
    9. Low Retrieval Quality (avg <0.6 for 15 min)
    10. Low Quality Prediction Spike (>10 in 5 min)
  - **Retraining Alert (1)**:
    11. Retraining Triggered (immediate notification)

- **1 Monitoring Dashboard**: 11 widgets showing production and quality metrics

#### Manual Setup

```bash
cd deploymentRAG/terraform

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_id = "<YOUR-GCP_PROJECT_ID>"
region = "us-central1"
service_name = "rag-service"
enable_monitoring = true
monitoring_email = "your-email@example.com"
create_notification_channel = true
enable_apis = false
EOF

# Initialize and apply
terraform init
terraform plan
terraform apply
```

#### Verification

```bash
# List custom metrics
gcloud logging metrics list --project=<YOUR-GCP_PROJECT_ID> | grep rag_

# List alert policies
gcloud alpha monitoring policies list --project=<YOUR-GCP_PROJECT_ID> | grep "RAG Service"

# List dashboards
gcloud monitoring dashboards list --project=<YOUR-GCP_PROJECT_ID> | grep "RAG Service"

# View dashboard in browser
open "https://console.cloud.google.com/monitoring/dashboards?project=<YOUR-GCP_PROJECT_ID>"
```

**Dashboard URL**: https://console.cloud.google.com/monitoring/dashboards?project=<YOUR-GCP_PROJECT_ID>

---

### Vision Monitoring Infrastructure

Set up monitoring for the Vision inference service with 6 alert policies, 1 custom metric, and a monitoring dashboard.

#### Automated Setup (Recommended)

**Via GitHub Actions**:
1. Navigate to **Actions** → **Vision Inference - Terraform Setup**
2. Click **Run workflow**
3. Configure:
   - `action`: `apply`
   - `auto_approve`: `true`
4. Click **Run workflow**

**Workflow**: `.github/workflows/vision-inference-terraform-setup.yaml`

**Resources Created**:
- **1 Custom Log-Based Metric**:
  - `low_confidence_predictions` - Counts predictions with confidence below threshold

- **6 Alert Policies**:
  1. High Error Rate (>5% for 5 min)
  2. High Latency (P95 >5s for 5 min)
  3. Service Unavailable (no requests in 5h)
  4. High CPU (>80% for 5 min)
  5. High Memory (>85% for 5 min)
  6. Low Confidence Streak (>=3 in 30s)

- **1 Monitoring Dashboard**: 6 widgets showing production metrics

#### Manual Setup

```bash
cd deploymentVisionInference/terraform

# Copy example config
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your settings:
# - project_id
# - region
# - service_name
# - monitoring_email

# Initialize and apply
terraform init
terraform plan
terraform apply
```

#### Verification

```bash
# List custom metrics
gcloud logging metrics list --project=<YOUR-GCP_PROJECT_ID> | grep "low_confidence"

# List alert policies
gcloud alpha monitoring policies list --project=<YOUR-GCP_PROJECT_ID> | grep "Vision Inference"

# List dashboards
gcloud monitoring dashboards list --project=<YOUR-GCP_PROJECT_ID> | grep "Vision Inference"

# View dashboard in browser
open "https://console.cloud.google.com/monitoring/dashboards?project=<YOUR-GCP_PROJECT_ID>"
```

**Dashboard URL**: https://console.cloud.google.com/monitoring/dashboards?project=<YOUR-GCP_PROJECT_ID>

---

## Stage 5: Continuous Monitoring

Set up continuous monitoring to track production performance and trigger retraining when needed.

### RAG Monitoring

**Automated (Scheduled)**:
- Workflow: `.github/workflows/rag-monitoring.yaml`
- Schedule: Runs every 6 hours via cron
- **Triggers**: 
  - Scheduled (cron: `0 */6 * * *`)
  - Manual dispatch

**What it does**:
- Runs monitoring checks (Python script)
- Collects metrics from Cloud Logging
- Analyzes performance and quality scores
- Optionally triggers retraining if thresholds exceeded

**Manual Execution**:
```bash
cd Monitoring/RAG

# Run monitoring for last 24 hours
python run_monitoring.py --hours=24

# With retraining trigger
python run_monitoring.py --hours=24 --trigger-retrain
```

**Retraining Trigger Options**:
- Automatic: Set `trigger_retrain=true` in workflow
- Manual: Run monitoring script with `--trigger-retrain` flag
- Via script: `python scripts/trigger_retraining.py`

---

### Vision Monitoring

**Automated (Scheduled)**:
- Workflow: `.github/workflows/vision-inference-retrain-decoy.yaml`
- Schedule: Runs daily via cron
- **Triggers**:
  - Scheduled (cron: `0 0 * * *`)
  - Manual dispatch

**What it does**:
- Monitors low-confidence predictions from Cloud Logging
- Checks if threshold exceeded (default: 50 low-confidence predictions in 24h)
- Sends alerts if threshold exceeded
- Can trigger retraining workflow

**Manual Execution**:
```bash
# Via GitHub Actions UI:
# 1. Navigate to Actions → "Vision Inference Retraining Threshold Monitor"
# 2. Click "Run workflow"
# 3. Configure:
#    - threshold: 50 (default)
#    - hours_lookback: 24 (default)
# 4. Click "Run workflow"
```

**Retraining Trigger**:
- After monitoring detects degradation, manually trigger:
  ```bash
  # Trigger retraining workflow
  # Navigate to Actions → "Vision Model Training"
  # Click "Run workflow" with appropriate dataset
  ```

---

## Verification and Testing

### Verification Checklist

#### ✅ Vision Data Pipeline
- [ ] Preprocessed images exist in `data/preprocessed/`
- [ ] Synthetic metadata generated
- [ ] Validation reports in GCS
- [ ] DVC tracking updated

#### ✅ RAG Data Pipeline
- [ ] Embeddings in `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/RAG/embeddings/`
- [ ] FAISS index in `gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/RAG/index/`
- [ ] Chunks processed and stored
- [ ] Pipeline metadata available

#### ✅ Vision Model Training
- [ ] Models trained and saved to GCS
- [ ] Validation results generated
- [ ] Models registered in Vertex AI
- [ ] Best model selected

#### ✅ RAG Model Training
- [ ] Model selection completed
- [ ] Best model configuration saved
- [ ] Model artifacts in GCS
- [ ] Training metrics logged

#### ✅ Vision Inference Deployment
- [ ] Cloud Run service deployed
- [ ] Health endpoint returns 200 OK
- [ ] TB prediction endpoint accessible
- [ ] Lung Cancer prediction endpoint accessible

#### ✅ RAG Service Deployment
- [ ] Cloud Run service deployed
- [ ] Health endpoint returns `{"ready": true}`
- [ ] Config endpoint returns model info
- [ ] Prediction endpoint functional

#### ✅ RAG Monitoring Infrastructure
- [ ] Dashboard exists: "RAG Service - Monitoring Dashboard"
- [ ] 7 custom metrics created
- [ ] 11 alert policies created
- [ ] Email notification channel configured

#### ✅ Vision Monitoring Infrastructure
- [ ] Dashboard exists: "Vision Inference API - Monitoring Dashboard"
- [ ] 1 custom metric created (`low_confidence_predictions`)
- [ ] 6 alert policies created
- [ ] Email notification channel configured

### Testing Commands

**Test RAG Service**:
```bash
SERVICE_URL=$(gcloud run services describe rag-service \
  --region=us-central1 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --format="value(status.url)")

curl $SERVICE_URL/health | jq
curl $SERVICE_URL/config | jq
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"query":"What is tuberculosis?"}]}' | jq
```

**Test Vision Service**:
```bash
SERVICE_URL=$(gcloud run services describe vision-inference-api \
  --region=us-central1 \
  --project=<YOUR-GCP_PROJECT_ID> \
  --format="value(status.url)")

curl $SERVICE_URL/health
```

**Use Test Scripts**:
```bash
# RAG test script
./scripts/test-rag-deployment.sh
```

For comprehensive testing instructions, see [docs/TESTING-GUIDE.md](TESTING-GUIDE.md).

---

## Troubleshooting

### Data Pipeline Issues

**Problem**: Data pipeline fails during preprocessing
- **Solution**: Check DVC configuration and GCS permissions
- **Debug**: Review Airflow logs or Cloud Build logs
- **Fix**: Ensure data exists in source location and GCS bucket is accessible

**Problem**: RAG pipeline fails during embedding generation
- **Solution**: Verify HuggingFace token in Secret Manager
- **Debug**: Check logs for authentication errors
- **Fix**: Update token: `gcloud secrets versions add huggingface-token --data-file=-`

### Training Issues

**Problem**: Vertex AI registration error: "no files in directory"
- **Solution**: Ensure SavedModel format is correct
- **Fix**: Pipeline should export SavedModel, not `.keras` file. Verify `artifact_uri` points to directory.

**Problem**: Permission error when registering/deploying
- **Solution**: Grant Vertex service agent Storage read permission
- **Fix**:
    ```bash
  gcloud storage buckets add-iam-policy-binding gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID> \
    --member="serviceAccount:service-<PROJECT_NUMBER>@gcp-sa-aiplatform.iam.gserviceaccount.com" \
      --role="roles/storage.objectViewer" \
    --project=<YOUR-GCP_PROJECT_ID>
  ```

**Problem**: Deployment step appears stuck
- **Solution**: First deploys can take 30-60+ minutes. This is normal.
- **Debug**: Check operation status:
  ```bash
  gcloud ai operations describe <OPERATION_ID> \
    --project=<YOUR-GCP_PROJECT_ID> --region=us-central1
  ```

**Problem**: Validation or test data not found
- **Solution**: Ensure DVC step pulled preprocessed data
- **Fix**: Rerun data pipeline if data is missing

### Deployment Issues

**Problem**: Cloud Run service deployment fails
- **Solution**: Check Cloud Build logs
- **Debug**: Verify model files exist in GCS
- **Fix**: Ensure training completed successfully before deployment

**Problem**: Service deployed but returns 403/404
- **Solution**: Check IAM permissions
- **Debug**: Verify service is running and publicly accessible
- **Fix**: Check IAM policy: `gcloud run services get-iam-policy <SERVICE_NAME> --region=us-central1`

**Problem**: Service fails to load model
- **Solution**: Verify model files in GCS
- **Debug**: Check service logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=<SERVICE_NAME>" --limit=50`
- **Fix**: Ensure model path in service configuration matches GCS location

### Monitoring Issues

**Problem**: Dashboard not showing data
- **Solution**: Wait 5-10 minutes for metrics to appear
- **Debug**: Verify service is logging predictions
- **Fix**: Make test predictions to generate logs

**Problem**: Alert policies not triggering
- **Solution**: Verify email notification channel is verified
- **Debug**: Check alert policy is enabled and thresholds are correct
- **Fix**: Test notification channel in GCP Console

**Problem**: Terraform apply fails
- **Solution**: Check Terraform logs
- **Debug**: Verify service account has monitoring permissions
- **Fix**: Ensure APIs are enabled and permissions are granted

### General Issues

**Problem**: Workflow fails at authentication
- **Solution**: Verify `GCP_SA_KEY` secret is valid
- **Fix**: Regenerate service account key and update secret

**Problem**: Costs are high
- **Solution**: Cloud Run with GPU can be expensive
- **Fix**: Set up billing alerts and clean up test resources

For more detailed troubleshooting, see:
- [docs/TESTING-GUIDE.md](TESTING-GUIDE.md) - Comprehensive testing and troubleshooting
- Component-specific READMEs in `ModelDevelopment/`, `deploymentRAG/`, `deploymentVisionInference/`

---

## Cleanup

### RAG Cleanup

**Delete Cloud Run Service**:
```bash
gcloud run services delete rag-service \
  --region=us-central1 \
  --project=<YOUR-GCP_PROJECT_ID>
```

**Delete Monitoring Resources**:
```bash
cd deploymentRAG/terraform
terraform destroy
```

**Delete Alert Policies** (if Terraform destroy doesn't work):
```bash
cd deploymentRAG/terraform
bash delete-alert-policies.sh
```

**Delete GCS Data** (optional):
```bash
gsutil -m rm -r gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/RAG/
```

---

### Vision Cleanup

**Delete Cloud Run Service**:
```bash
gcloud run services delete vision-inference-api \
  --region=us-central1 \
  --project=<YOUR-GCP_PROJECT_ID>
```

**Delete Monitoring Resources**:
```bash
cd deploymentVisionInference/terraform
terraform destroy
```

**Delete Vertex AI Endpoints**:
```bash
# List endpoints
gcloud ai endpoints list --project=<YOUR-GCP_PROJECT_ID> --region=us-central1

# Delete endpoint
gcloud ai endpoints delete <ENDPOINT_ID> \
  --project=<YOUR-GCP_PROJECT_ID> --region=us-central1
```

**Delete Vertex AI Models**:
```bash
# List models
gcloud ai models list --project=<YOUR-GCP_PROJECT_ID> --region=us-central1

# Delete model
gcloud ai models delete <MODEL_ID> \
  --project=<YOUR-GCP_PROJECT_ID> --region=us-central1
```

**Delete GCS Artifacts** (optional):
```bash
gsutil -m rm -r gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/vision/trained_models/
gsutil -m rm -r gs://medscan-pipeline-<YOUR-GCP_PROJECT_ID>/vision/validation/
```

**Note**: Be careful with cleanup - make sure you're not deleting production resources!

---

## Quick Reference

### Key URLs

- **Airflow UI**: http://localhost:8080 (local) or GCP Composer (production)
- **GCP Console**: https://console.cloud.google.com/?project=<YOUR-GCP_PROJECT_ID>
- **Monitoring Dashboards**: https://console.cloud.google.com/monitoring/dashboards?project=<YOUR-GCP_PROJECT_ID>
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

### Workflow Summary

**RAG Workflows**:
- `rag-data-pipeline.yaml` - Data pipeline
- `rag-training.yaml` - Model training
- `rag-complete-setup.yaml` - Deployment + monitoring (recommended)
- `rag-deploy.yaml` - Deployment only
- `rag-monitoring.yaml` - Continuous monitoring (scheduled)

**Vision Workflows**:
- `vision-training.yaml` - Model training
- `vision-inference-deploy.yaml` - Service deployment
- `vision-inference-terraform-setup.yaml` - Monitoring setup
- `vision-inference-retrain-decoy.yaml` - Retraining monitor (scheduled)

---

## Related Documentation

- **[README.md](../README.md)**: Main project documentation
- **[TESTING-GUIDE.md](TESTING-GUIDE.md)**: Comprehensive testing guide
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Project structure details
- **[DVC.md](DVC.md)**: Data versioning with DVC
- **[MODEL_MONITORING_AUDIT_REPORT.md](MODEL_MONITORING_AUDIT_REPORT.md)**: Monitoring audit
- **[README-vision-inference.md](README-vision-inference.md)**: Vision inference details

### Component-Specific Documentation

- **Data Pipeline**: `DataPipeline/README.md`
- **RAG Model Development**: `ModelDevelopment/RAG/README.md`
- **Vision Model Development**: `ModelDevelopment/Vision/README.md`
- **RAG Terraform**: `deploymentRAG/terraform/README.md`
- **Vision Terraform**: `deploymentVisionInference/terraform/README.md`

---

**Last Updated**: December 2024  
**Maintained By**: MedScan AI Development Team
