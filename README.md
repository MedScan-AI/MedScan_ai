# MedScan AI

AI-powered radiological assistant for CT scan analysis with explainable AI and patient engagement RAG to answer questions about their report's content.

Note: This project is in the initial development phase. Repository structure, naming conventions, technology choices, and implementation details are subject to change based on ongoing technical discussions and requirements refinement.

## About MedScan AI

Scope - Click [here](docs/Scoping.pdf) 

### High Level Architecture

![Architecture](assets/high_level_architecture.png)

## Getting Started

### Prerequisites

- **Python 3.10**
- **Docker Desktop**
- **Google Cloud Platform Account**
- **Kaggle Account** (for dataset access)
- **Git**

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/rjaditya-2702/MedScan_ai.git
cd MedScan_ai
```

#### 2. Setup GCP Credentials

Download your service account JSON key from GCP Console:

1. Go to **IAM & Admin → Service Accounts**
2. Select your service account
3. Click **Keys → Add Key → Create New Key → JSON**
4. Save the file as `~/gcp-service-account.json`

```bash
# Verify credentials file
ls -la ~/gcp-service-account.json
```

Required GCP Permissions:

- Storage Admin
- Storage Object Admin

#### 3. Setup Kaggle Credentials

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Download kaggle.json from kaggle.com/settings
# Place it in ~/.kaggle/
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Configure Environment Variables

**Environment Variable Management:**

The project uses **two `.env` files** for different purposes:

1. **Root `.env`** (optional for local ModelDevelopment): For running ModelDevelopment scripts locally
2. **`airflow/.env`** (required for Airflow): For Docker Compose and Airflow execution

**Option 1: Single Root `.env` (Recommended for Local Development)**

Create a `.env` file at the project root:
```bash
# Copy the template
cp .env.example .env

# Edit and fill in your values
nano .env
```

**Option 2: System Environment Variables**

Set environment variables in your shell:
```bash
# Add to ~/.zshrc or ~/.bashrc
export GCP_PROJECT_ID="your-gcp-project-id"
export GCS_BUCKET_NAME="your-gcs-bucket-name"
```

**Required Environment Variables:**
```bash
GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET_NAME=your-gcs-bucket-name
```

**Where Environment Variables Are Used:**

| Context | Source | Notes |
|---------|--------|-------|
| **Local ModelDevelopment** | System env vars or root `.env` | Scripts use `os.getenv()` directly |
| **Airflow/Docker** | `airflow/.env` | Loaded by Docker Compose |
| **Cloud Build** | Substitution variables | Set in Cloud Build config |

**Note:** If environment variables are not set, some scripts may fall back to default values (for backward compatibility), but this is **not recommended for production use**.

**Create `airflow/.env` file (for Docker/Airflow):**

The `airflow/.env` file is specifically for Airflow Docker Compose. It should include both Airflow-specific settings AND the shared GCP configuration.

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

# Airflow User
AIRFLOW__CORE__FERNET_KEY=your-existing-fernet-key
AIRFLOW__WEBSERVER__SECRET_KEY=your-existing-secret-key
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow

AIRFLOW_UID=50000
AIRFLOW_GID=0
AIRFLOW_USERNAME=airflow
AIRFLOW_PASSWORD=airflow123
AIRFLOW_FIRSTNAME=Admin
AIRFLOW_LASTNAME=User
AIRFLOW_EMAIL=admin@example.com

# GCP Configuration (REQUIRED - Set these values)
GCP_PROJECT_ID=medscanai-476203
GCS_BUCKET_NAME=medscan-data
# Note: All Python scripts and components now use these environment variables
# instead of hardcoded values. Make sure these match your actual GCP project.
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/gcp-service-account.json

# Email Alerts (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_MAIL_FROM=your-email@gmail.com
ALERTS_ENABLED=true
ALERT_EMAIL_RECIPIENTS=harshitha8.shekar@gmail.com,kothari.sau@northeastern.edu

VISION_MAX_ANOMALY_PCT=25.0
VISION_MAX_ANOMALIES=10
VISION_MAX_DRIFT_FEATURES=3
VISION_MIN_COMPLETENESS=0.95
VISION_EXPECTED_TB_IMAGES=700
VISION_EXPECTED_LC_IMAGES=1000
VISION_VARIANCE_TOLERANCE=0.1
VISION_MIN_IMAGE_COUNT=500
VISION_PREPROCESS_MIN_SUCCESS=0.95
VISION_BIAS_MAX_VARIANCE=0.3
VISION_BIAS_MIN_REPRESENTATION=0.05

RAG_SCRAPING_MIN_SUCCESS=0.7
RAG_MAX_ANOMALY_PCT=25.0
RAG_MAX_DRIFT_FEATURES=3
RAG_EMBEDDING_MIN_SUCCESS=0.95
RAG_MIN_VECTORS=100

AIRFLOW_URL=http://localhost:8080
MLFLOW_URL=http://localhost:5000
SKIP_CONFIG_VALIDATION=false
DEBUG=false
EOF
```

#### 6. Initialize Docker Environment

```bash
# Make scripts executable
chmod +x quickstart.sh
chmod +x DataPipeline/scripts/common/complete_init.sh

# Run quickstart (builds images and initializes everything)
./quickstart.sh
```

**OR manually:**

```bash
cd airflow

# Build Docker images
docker-compose build

# Start services
docker-compose up -d

# Monitor initialization
docker-compose logs -f airflow-init
```

**Expected initialization output:**

```
✓ PostgreSQL ready
✓ Airflow database initialized
✓ Admin user created
✓ MLflow directories ready
✓ GCS bucket created/verified
✓ Folder structure created
✓ URLs file uploaded
✓ Git initialized
✓ DVC initialized
✓ DVC remotes configured
```

#### 7. Verify Setup

```bash
# Check containers are running
docker-compose ps

# Verify GCS bucket
gsutil ls gs://medscan-data/

# Verify DVC
docker-compose exec webserver bash
cd /opt/airflow/DataPipeline
dvc remote list
# Should show:
# vision	gs://medscan-data/dvc-storage/vision
# rag	gs://medscan-data/dvc-storage/rag
exit

# Access Airflow UI
open http://localhost:8080
# Login: airflow / airflow123
```
## Usage

### Running Pipelines

#### **Vision Pipeline** (Medical Image Processing)

**Via Airflow UI:**

1. Navigate to http://localhost:8080
2. Find DAG: `medscan_vision_pipeline_dvc`
3. Toggle **ON**
4. Click **Trigger DAG**

**Via CLI:**

```bash
docker-compose exec webserver airflow dags trigger medscan_vision_pipeline_dvc
```

**What it does:**

1. Downloads TB and Lung Cancer datasets from Kaggle
2. Preprocesses images (resize to 224x224, JPEG standardization)
3. Generates synthetic patient metadata using Faker
4. Runs comprehensive validation (schema, drift, bias detection)
5. Applies bias mitigation strategies
6. Tracks all data with DVC
7. Uploads validation reports to GCS
8. Sends email alerts if issues detected

**Pipeline Tasks:**

```
download_kaggle_data → preprocess_images → generate_metadata →
validate_and_upload_reports → track_all_with_dvc →
check_validation_results → check_drift_results → check_bias_results →
send_alerts (if needed) → cleanup
```

#### **RAG Pipeline** (Medical Knowledge Base)

**Via Airflow UI:**

1. Navigate to http://localhost:8080
2. Find DAG: `rag_data_pipeline_dvc`
3. Toggle **ON**
4. Click **Trigger DAG**

**Via CLI:**

```bash
docker-compose exec webserver airflow dags trigger rag_data_pipeline_dvc
```

**What it does:**

1. Scrapes 40+ medical URLs (CDC, WHO, Mayo Clinic, research papers)
2. Creates baseline or validates against existing baseline
3. Chunks documents for optimal retrieval
4. Generates embeddings using llm-embedder
5. Builds FAISS vector index
6. Tracks all data with DVC
7. Uploads validation reports to GCS

**Pipeline Tasks:**

```
scrape_data → check_baseline → [create_baseline OR validate_data] →
chunk_data → generate_embeddings → create_index → track_all_with_dvc
```
### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## Dataset Information

### **Medical Image Datasets**

**Time Period**: Public datasets updated through 2024

**Size**:

- Tuberculosis X-rays: ~700 images
- Lung Cancer CT scans: ~1,000 images

#### **Data Types**

**Vision Pipeline (Medical Images)**:

- **Images**: Chest X-rays (TB), CT scans (Lung Cancer)
- **Metadata**: Patient demographics, symptoms, medical history
- **Numerical**: Age, weight, height
- **Categorical**: Gender, diagnosis class, urgency level
- **Text**: Presenting symptoms, medications, surgical history

**RAG Pipeline (Medical Knowledge)**:

- **Text**: Medical articles, research papers, treatment guidelines
- **Metadata**: Authors, publication dates, source types
- **Topics**: Treatment methods, disease information, clinical guidelines

### **Data Sources**

1. **Medical Images**: Kaggle public datasets

   - [Tuberculosis Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
   - [Lung Cancer CT Scan Dataset](https://www.kaggle.com/datasets/dishantrathi20/ct-scan-images-for-lung-cancer)

2. **Medical Knowledge**: Trusted medical sources

## Key Features

### Data Pipeline Capabilities

-  **Automated Data Acquisition**: Fetches datasets from Kaggle with partitioning
-  **Image Preprocessing**: Standardizes images to 224x224 JPEG format
-  **Synthetic Metadata Generation**: Creates realistic patient demographics using Faker
-  **Comprehensive Validation**: Schema validation, drift detection, anomaly detection
-  **Bias Detection & Mitigation**: Advanced fairness analysis using Fairlearn, SliceFinder, TFMA
-  **Data Versioning**: DVC integration with GCS for reproducibility
-  **RAG Knowledge Base**: Medical article scraping, chunking, embedding, and FAISS indexing
-  **Email Alerts**: Automated notifications for anomalies, drift, and bias

### Advanced Quality Checks

**Validation Framework**:

- Schema consistency validation
- Statistical drift detection (Kolmogorov-Smirnov, Chi-square tests)
- Bias detection across demographic slices
- Exploratory Data Analysis (EDA) with outlier detection
- Fairness metrics (demographic parity, equalized odds)

**Bias Mitigation Strategies**:

- Resampling underrepresented groups
- Class weight computation
- Stratified split recommendations
- Fairlearn post-processing techniques

### Data Versioning with DVC

#### Check DVC Status

```bash
docker-compose exec webserver bash
cd /opt/airflow/DataPipeline

# Overall status
dvc status

# Remote status
dvc status -r vision
dvc status -r rag
```
### Alerts

Email alerts are configured for:

- **Data Acquisition Failures**: Kaggle download errors
- **Preprocessing Failures**: Image processing errors
- **Validation Issues**: Anomalies exceed threshold (>25%)
- **Drift Detection**: More than 3 features show drift
- **Bias Detection**: Fairness violations detected
- **DVC Operations**: Push/pull failures
