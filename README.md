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
   - CDC, WHO, Mayo Clinic
   - NIH/PubMed research papers
   - Cancer.org, NCI treatment guidelines
   - Clinical journals and research databases

## Key Features

### Data Pipeline Capabilities

- ✅ **Automated Data Acquisition**: Fetches datasets from Kaggle with partitioning
- ✅ **Image Preprocessing**: Standardizes images to 224x224 JPEG format
- ✅ **Synthetic Metadata Generation**: Creates realistic patient demographics using Faker
- ✅ **Comprehensive Validation**: Schema validation, drift detection, anomaly detection
- ✅ **Bias Detection & Mitigation**: Advanced fairness analysis using Fairlearn, SliceFinder, TFMA
- ✅ **Data Versioning**: DVC integration with GCS for reproducibility
- ✅ **RAG Knowledge Base**: Medical article scraping, chunking, embedding, and FAISS indexing
- ✅ **Email Alerts**: Automated notifications for anomalies, drift, and bias
- ✅ **MLflow Tracking**: Experiment tracking and artifact logging

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
git clone https://github.com/your-username/MedScan_ai.git
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

#### 5. Configure Environment

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

# Airflow User
AIRFLOW_USERNAME=airflow
AIRFLOW_PASSWORD=airflow123
AIRFLOW_FIRSTNAME=Admin
AIRFLOW_LASTNAME=User
AIRFLOW_EMAIL=admin@medscan.ai

# GCP Configuration
GCP_PROJECT_ID=medscanai-476203
GCS_BUCKET_NAME=medscan-data

# Email Alerts (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_MAIL_FROM=your-email@gmail.com
ALERTS_ENABLED=true
ALERT_EMAIL_RECIPIENTS=harshitha8.shekar@gmail.com,kothari.sau@northeastern.edu
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

### Managing Medical URLs

The RAG pipeline uses a versioned URLs list stored in GCS.

#### View Current URLs

```bash
docker-compose exec webserver bash
cd /opt/airflow/DataPipeline
python scripts/common/manage_urls.py list
```

#### Add New Medical Resources

```bash
# Add URLs
python scripts/common/manage_urls.py add \
  "https://www.nejm.org/doi/full/10.1056/NEJMoa1234567" \
  "https://pubmed.ncbi.nlm.nih.gov/12345678/"
```

#### Download and Edit

```bash
# Download current list
python scripts/common/manage_urls.py download -o urls.txt

# Edit
nano urls.txt

# Upload updated list (creates automatic backup)
python scripts/common/manage_urls.py upload urls.txt
```

#### View Version History

```bash
# List all versions
python scripts/common/manage_urls.py list

# Restore previous version
python scripts/common/manage_urls.py restore RAG/config/versions/urls_20250129_120000.txt
```

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

#### Pull Data from DVC

```bash
# Pull all data
dvc pull

# Pull specific pipeline
dvc pull -r vision
dvc pull -r rag

# Pull specific dataset
dvc pull data/raw.dvc
```

#### View Tracked Files

```bash
# List DVC tracked files
ls -la *.dvc
ls -la data/*.dvc

# View DVC file content
cat data/raw.dvc
```

## Data Quality & Validation

### Validation Pipeline

The pipeline performs comprehensive data quality checks:

**Schema Validation**:

- Validates against predefined schemas
- Detects missing/extra columns
- Checks data type consistency
- Tracks schema changes over time

**Drift Detection**:

- Kolmogorov-Smirnov test for numerical features
- Chi-square test for categorical features
- Temporal distribution monitoring
- Alert if drift exceeds thresholds

**Bias Detection**:

- **SliceFinder**: Automatic discovery of problematic data slices
- **Fairlearn**: Industry-standard fairness metrics
- **TFMA**: Model performance across demographic groups
- Detects disparate impact, unequal representation
- Age, gender, diagnosis class analysis

**Bias Mitigation**:

- Resampling underrepresented groups
- Class weight computation
- Stratified sampling recommendations
- Fairlearn post-processing techniques

### Alerts

Email alerts are configured for:

- **Data Acquisition Failures**: Kaggle download errors
- **Preprocessing Failures**: Image processing errors
- **Validation Issues**: Anomalies exceed threshold (>25%)
- **Drift Detection**: More than 3 features show drift
- **Bias Detection**: Fairness violations detected
- **DVC Operations**: Push/pull failures

Alert thresholds (configurable in `.env`):

```bash
VISION_MAX_ANOMALY_PCT=25.0
VISION_MAX_DRIFT_FEATURES=3
RAG_SCRAPING_MIN_SUCCESS=0.7
RAG_EMBEDDING_MIN_SUCCESS=0.95
```

## Monitoring

### Airflow UI

Monitor pipeline execution:

```
http://localhost:8080
Username: airflow
Password: airflow123
```

Features:

- DAG run history
- Task logs
- Task duration metrics
- Retry tracking
- Email alert history

### MLflow Tracking

View experiment metrics:

```bash
docker-compose exec webserver bash
cd /opt/airflow/DataPipeline
mlflow ui --backend-store-uri file:///opt/airflow/mlflow/mlruns

# Access at: http://localhost:5000
```

Tracked metrics:

- Dataset statistics
- Validation results
- Drift metrics
- Bias detection results
- Schema changes

### GCS Console

View stored data and reports:

```
https://console.cloud.google.com/storage/browser/medscan-data
```

Structure:

- `vision/ge_outputs/` - Validation reports and HTML visualizations
- `vision/mlflow/` - MLflow artifacts
- `RAG/validation/` - RAG validation reports
- `dvc-storage/` - Versioned data storage
