# MedScan AI - Project Structure Documentation

This document provides a comprehensive overview of the MedScan AI project structure, file organization, and detailed descriptions of what each file contains.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Root Directory Structure](#root-directory-structure)
3. [Directory Details](#directory-details)
   - [DataPipeline/](#datapipeline)
   - [ModelDevelopment/](#modeldevelopment)
   - [airflow/](#airflow)
   - [deployment/](#deployment)
   - [deploymentRAG/](#deploymentrag)
   - [deploymentVisionInference/](#deploymentvisioninference)
   - [cloudbuild/](#cloudbuild)
   - [scripts/](#scripts)
   - [docs/](#docs)
   - [assets/](#assets)
4. [Key Configuration Files](#key-configuration-files)
5. [Data Organization](#data-organization)
6. [Workflow Overview](#workflow-overview)

---

## Project Overview

**MedScan AI** is an AI-powered radiological assistant for CT scan analysis with explainable AI and patient engagement RAG (Retrieval-Augmented Generation) to answer questions about medical report content.

### Key Components

1. **Data Pipeline**: Automated data acquisition, preprocessing, validation, and bias detection
2. **Vision Models**: Medical image classification models (TB detection, Lung Cancer detection)
3. **RAG System**: Medical knowledge base with retrieval and question-answering capabilities
4. **Orchestration**: Apache Airflow for pipeline automation
5. **Deployment**: Cloud Build configurations for CI/CD

---

## Root Directory Structure

```
MedScan_ai/
├── README.md                      # Main project README with setup instructions
├── PROJECT_STRUCTURE.md          # This file - detailed project structure documentation
├── requirements.txt              # Root-level Python dependencies
├── report.txt                    # Technical report on vision pipeline
├── quickstart.sh                 # Quick start script for project initialization
│
├── DataPipeline/                 # Data acquisition, preprocessing, and validation
├── ModelDevelopment/             # Model training and inference code
├── airflow/                      # Apache Airflow orchestration setup
├── deployment/                   # Shared deployment configurations
├── deploymentRAG/               # RAG service deployment and monitoring (Terraform)
├── deploymentVisionInference/    # Vision inference deployment and monitoring (Terraform)
├── cloudbuild/                   # Google Cloud Build configurations
├── scripts/                      # Utility scripts
├── docs/                         # Documentation files
└── assets/                       # Static assets (images, diagrams)
```

---

## Directory Details

### DataPipeline/

**Purpose**: Handles all data-related operations including acquisition, preprocessing, validation, bias detection, and RAG knowledge base creation.

```
DataPipeline/
├── README.md                     # Data pipeline documentation
├── setup_gcs_bucket.sh          # GCS bucket setup script
│
├── config/                       # Configuration files
│   ├── gcp_config.py            # GCP configuration helper
│   ├── metadata.yml             # Main metadata validation config
│   ├── synthetic_data.yml       # Synthetic data generation config
│   ├── vision_pipeline.yml      # Vision pipeline configuration
│   ├── rag_pipeline.yml         # RAG pipeline configuration
│   ├── dev/                     # Development environment configs
│   │   ├── metadata.yml
│   │   ├── synthetic_data.yml
│   │   └── vision_pipeline.yml
│   └── prod/                    # Production environment configs
│       ├── metadata.yml
│       ├── synthetic_data.yml
│       └── vision_pipeline.yml
│
├── scripts/                      # Pipeline execution scripts
│   ├── common/                  # Shared utilities
│   │   ├── __init__.py
│   │   ├── auto_setup_gcs.py   # GCS bucket auto-setup
│   │   ├── complete_init.sh    # Complete initialization script
│   │   ├── dvc_helper.py       # DVC (Data Version Control) utilities
│   │   ├── gcs_manager.py      # Google Cloud Storage manager
│   │   ├── init_dvc.py         # DVC initialization
│   │   └── manage_urls.py      # URL management for RAG pipeline
│   │
│   ├── data_acquisition/        # Data acquisition from external sources
│   │   └── fetch_data.py       # Kaggle dataset fetcher
│   │
│   ├── data_preprocessing/      # Data preprocessing and validation
│   │   ├── alert_utils.py      # Email alert utilities
│   │   ├── baseline_synthetic_data_generator.py  # Synthetic metadata generator
│   │   ├── process_lungcancer.py  # Lung cancer dataset processor
│   │   ├── process_tb.py       # Tuberculosis dataset processor
│   │   └── schema_statistics.py  # Schema validation and statistics
│   │
│   └── RAG/                     # RAG pipeline scripts
│       ├── __init__.py
│       ├── alert_utils.py      # RAG-specific alert utilities
│       ├── chunking.py         # Document chunking for RAG
│       ├── create_urls_file.py # Creates URLs file for scraping
│       ├── embedding.py        # Text embedding generation
│       ├── indexing.py         # FAISS index creation
│       ├── main.py             # Main RAG pipeline orchestrator
│       ├── scraper.py          # Web scraper for medical content
│       ├── url_manager.py      # URL management and validation
│       └── analysis/           # Data quality analysis
│           ├── anomalies_and_bias_detection.py  # Anomaly and bias detection
│           ├── drift.py        # Data drift detection
│           ├── main.py         # Analysis pipeline orchestrator
│           └── validator.py    # Data validator
│
├── data/                        # Data storage (versioned with DVC)
│   ├── raw.dvc                 # Raw data tracking
│   ├── preprocessed.dvc        # Preprocessed data tracking
│   ├── synthetic_metadata.dvc  # Synthetic metadata tracking
│   ├── synthetic_metadata_mitigated.dvc  # Bias-mitigated data tracking
│   ├── RAG/                    # RAG-specific data
│   │   ├── chunks.dvc         # Document chunks tracking
│   │   ├── embeddings.dvc     # Embeddings tracking
│   │   ├── index.dvc          # FAISS index tracking
│   │   └── raw_data/          # Raw scraped data
│   │       ├── baseline.dvc
│   │       └── incremental.dvc
│   └── [actual data directories]  # Raw, preprocessed, etc.
│
├── docs/                        # Documentation
│   └── schema_statistics.md    # Schema statistics documentation
│
├── experiment/                  # Experimental notebooks
│   └── md_modification.ipynb   # Markdown modification experiments
│
└── tests/                       # Unit tests
    ├── data_acquisition/
    │   └── fetch_data_test.py
    ├── data_preprocessing/
    │   ├── baseline_synthetic_data_generator_test.py
    │   ├── preprocess_lung_cancer_test.py
    │   ├── preprocess_tb_test.py
    │   └── schema_statistics_test.py
    ├── test_chunking.py
    ├── test_embedding.py
    ├── test_indexer.py
    ├── test_rag_analysis_integration.py
    ├── test_rag_anomalies_and_bias_detection.py
    ├── test_rag_drift.py
    ├── test_rag_validator.py
    └── test_scraper.py
```

#### Key Files in DataPipeline/

- **`config/metadata.yml`**: Main configuration for data validation, schema checks, drift detection, and bias analysis
- **`config/vision_pipeline.yml`**: Configuration for vision pipeline (data acquisition, preprocessing)
- **`config/rag_pipeline.yml`**: Configuration for RAG pipeline (scraping, chunking, embedding)
- **`scripts/data_preprocessing/schema_statistics.py`**: Core validation engine with schema validation, drift detection, bias detection, and mitigation
- **`scripts/RAG/scraper.py`**: Web scraper that fetches medical content from 40+ trusted sources
- **`scripts/RAG/chunking.py`**: Intelligent document chunking for optimal retrieval
- **`scripts/RAG/embedding.py`**: Generates embeddings using llm-embedder models
- **`scripts/RAG/indexing.py`**: Creates FAISS vector index for similarity search

---

### ModelDevelopment/

**Purpose**: Contains model training, selection, and inference code for both Vision and RAG models.

```
ModelDevelopment/
├── Vision/                       # Vision model development
│   ├── README.md                # Vision model documentation
│   ├── requirements.txt         # Vision-specific dependencies
│   ├── Dockerfile              # Docker image for vision training
│   ├── train_resnet.py         # ResNet50 training script
│   ├── train_vit.py            # Vision Transformer training script
│   ├── train_custom_cnn.py     # Custom CNN training script
│   ├── validate.py             # Model validation script
│   ├── deploy.py               # Model deployment script
│   └── bias_check.py           # Bias checking for vision models
│
├── RAG/                         # RAG model development
│   ├── README.md               # RAG model documentation
│   ├── requirements.txt        # RAG-specific dependencies
│   ├── deploy.py               # RAG deployment script
│   │
│   ├── ModelSelection/         # Model selection and experimentation
│   │   ├── experiment.py       # Experiment runner
│   │   ├── prompts.py          # Prompt templates
│   │   ├── qa.json            # Question-answer pairs for evaluation
│   │   ├── rag_bias_adapter.py # Bias adaptation for RAG
│   │   ├── rag_bias_check.py   # Bias checking for RAG
│   │   ├── retreival_methods.py # Different retrieval strategies
│   │   └── select_best_model.py # Model selection logic
│   │
│   ├── ModelInference/         # Inference implementation
│   │   ├── __init__.py
│   │   ├── guardrails.py       # Safety guardrails for responses
│   │   └── RAG_inference.py    # Main RAG inference engine
│   │
│   ├── models/                 # Model definitions
│   │   └── models.py          # Model architecture definitions
│   │
│   ├── utils/                  # Utilities
│   │   └── RAG_config.json    # RAG configuration
│   │
│   └── test/                   # Tests
│       └── inference_test.py  # Inference testing
│
├── common/                      # Shared utilities
│   ├── email_notifier.py      # Email notification utilities
│   ├── gcp_utils.py           # GCP helper functions
│   └── monitoring.py          # Model monitoring utilities
│
└── config/                      # Model training configurations
    └── vision_training.yml     # Vision model training config
```

#### Key Files in ModelDevelopment/

**Vision Models:**
- **`Vision/train_resnet.py`**: Trains ResNet50 architecture for medical image classification
- **`Vision/train_vit.py`**: Trains Vision Transformer (ViT) model
- **`Vision/train_custom_cnn.py`**: Trains custom CNN architecture
- **`Vision/validate.py`**: Validates trained models on test sets
- **`Vision/bias_check.py`**: Checks for bias in vision model predictions

**RAG System:**
- **`RAG/ModelSelection/select_best_model.py`**: Selects best RAG model based on evaluation metrics
- **`RAG/ModelInference/RAG_inference.py`**: Main inference engine that combines retrieval and generation
- **`RAG/ModelInference/guardrails.py`**: Safety checks and content filtering for medical responses
- **`RAG/ModelSelection/prompts.py`**: Prompt templates for different medical query types

---

### airflow/

**Purpose**: Apache Airflow orchestration setup for automated pipeline execution.

```
airflow/
├── README.md                    # Airflow setup documentation
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Custom Airflow Docker image
├── .env                        # Environment variables (create this)
│
├── dags/                       # Airflow DAG definitions
│   ├── medscan_vision_pipeline_gcs.py  # Vision pipeline DAG
│   └── rag_data_pipeline.py   # RAG pipeline DAG
│
├── logs/                       # Airflow execution logs
│   └── scheduler/
│       └── [dated folders]
│
└── plugins/                    # Airflow plugins (empty by default)
```

#### Key Files in airflow/

- **`dags/medscan_vision_pipeline_gcs.py`**: DAG that orchestrates the entire vision pipeline:
  - Data acquisition from Kaggle
  - Image preprocessing
  - Synthetic metadata generation
  - Schema validation, drift detection, bias detection
  - Email alerts for issues
  - DVC tracking

- **`dags/rag_data_pipeline.py`**: DAG that orchestrates the RAG pipeline:
  - Web scraping of medical content
  - Baseline creation/validation
  - Document chunking
  - Embedding generation
  - FAISS index creation
  - DVC tracking

- **`docker-compose.yml`**: Defines Airflow services (webserver, scheduler, postgres, mlflow)
- **`Dockerfile`**: Custom Airflow image with project dependencies

---

### deployment/

**Purpose**: Shared deployment configurations and utilities.

```
deployment/
├── monitoring_config.yaml      # Model monitoring configuration
├── RAG_serve.py                # RAG service deployment script
└── requirements-extra.txt      # Additional deployment dependencies
```

---

### deploymentRAG/

**Purpose**: RAG service deployment and monitoring infrastructure using Terraform.

```
deploymentRAG/
└── terraform/                  # Terraform configuration for RAG monitoring
    ├── README.md               # RAG monitoring setup documentation
    ├── main.tf                 # Main Terraform configuration
    ├── monitoring.tf            # Monitoring resources (7 metrics, 11 alerts, dashboard)
    ├── variables.tf             # Input variables
    ├── outputs.tf               # Output values
    ├── versions.tf              # Provider version constraints
    ├── terraform.tfvars.example # Example configuration
    └── delete-alert-policies.sh # Cleanup script for alert policies
```

#### Key Files in deploymentRAG/

- **`terraform/monitoring.tf`**: Creates comprehensive monitoring infrastructure:
  - 7 custom log-based metrics (composite score, hallucination score, retrieval score, etc.)
  - 11 alert policies (5 production + 5 quality + 1 retraining)
  - Monitoring dashboard with 11 widgets
  - Email notification channel

- **`terraform/main.tf`**: Core infrastructure setup and provider configuration

- **`terraform/README.md`**: Complete guide for setting up RAG monitoring infrastructure

---

### deploymentVisionInference/

**Purpose**: Vision inference API deployment and monitoring infrastructure using Terraform.

```
deploymentVisionInference/
└── terraform/                  # Terraform configuration for Vision infrastructure
    ├── README.md               # Main Terraform documentation
    ├── README-terraform.md     # Terraform-specific guide
    ├── README-monitoring-setup.md  # Monitoring setup guide
    ├── README-CLEANUP.md       # Cleanup procedures
    ├── main.tf                 # Main Terraform configuration
    ├── monitoring.tf            # Monitoring resources (1 metric, 6 alerts, dashboard)
    ├── variables.tf             # Input variables
    ├── outputs.tf               # Output values
    ├── versions.tf              # Provider version constraints
    ├── terraform.tfvars.example # Example configuration
    ├── Makefile                 # Build automation
    ├── cleanup_resources.sh     # Resource cleanup script
    ├── delete_artifact_registry.sh  # Artifact Registry cleanup
    ├── import_existing.sh       # Import existing resources
    └── manual_delete.sh         # Manual deletion script
```

#### Key Files in deploymentVisionInference/

- **`terraform/monitoring.tf`**: Creates monitoring infrastructure:
  - 1 custom log-based metric (`low_confidence_predictions`)
  - 6 alert policies (error rate, latency, service unavailable, CPU, memory, low confidence streak)
  - Monitoring dashboard with 6 widgets
  - Email notification channel

- **`terraform/main.tf`**: Core infrastructure setup including:
  - Cloud Run service configuration
  - Artifact Registry setup
  - IAM bindings
  - Service account configuration

- **`terraform/README.md`**: Complete deployment and monitoring guide

- **`terraform/README-CLEANUP.md`**: Detailed cleanup procedures for all resources

---

### cloudbuild/

**Purpose**: Google Cloud Build configurations for CI/CD and automated retraining.

```
cloudbuild/
├── vision-training.yaml        # Cloud Build config for vision model training
├── rag-training.yaml          # Cloud Build config for RAG model training
└── send-email-notification.sh  # Email notification script
```

---

### scripts/

**Purpose**: Root-level utility scripts.

```
scripts/
├── setup_gcp.sh               # GCP setup script
└── trigger_retraining.py      # Script to trigger model retraining based on metrics
```

#### Key Files in scripts/

- **`trigger_retraining.py`**: Monitors model performance metrics and triggers Cloud Build jobs for retraining when thresholds are breached

---

### docs/

**Purpose**: Additional documentation files.

```
docs/
├── DVC.md                     # DVC (Data Version Control) documentation
└── Scoping.pdf               # Project scoping document
```

---

### assets/

**Purpose**: Static assets like images and diagrams.

```
assets/
└── high_level_architecture.png  # High-level system architecture diagram
```

---

## Key Configuration Files

### Configuration Files Overview

| File | Purpose | Key Settings |
|------|---------|--------------|
| `DataPipeline/config/metadata.yml` | Data validation and bias detection | Schema definitions, drift thresholds, bias detection parameters |
| `DataPipeline/config/vision_pipeline.yml` | Vision pipeline configuration | Kaggle datasets, preprocessing parameters |
| `DataPipeline/config/rag_pipeline.yml` | RAG pipeline configuration | URLs to scrape, chunking parameters |
| `ModelDevelopment/config/vision_training.yml` | Vision model training | Epochs, batch size, model architectures, hyperparameters |
| `ModelDevelopment/RAG/utils/RAG_config.json` | RAG inference configuration | Retrieval parameters, model selection |
| `airflow/.env` | Airflow environment variables | GCP credentials, email settings, pipeline thresholds |

### Configuration Details

#### `DataPipeline/config/metadata.yml`
- **Schema Validation**: Field types, constraints, value ranges
- **Drift Detection**: Statistical test thresholds (KS test, Chi-square)
- **Bias Detection**: Slicing features, statistical significance thresholds
- **Mitigation**: Resampling strategies, class weight computation

#### `ModelDevelopment/config/vision_training.yml`
- **Training Parameters**: Epochs, batch size, image size, validation split
- **Data Augmentation**: Rotation, shifts, flips, zoom ranges
- **Model Architectures**: ResNet50, ViT, Custom CNN configurations
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Hyperparameter Tuning**: Enable/disable, tuner type, search space

---

## Data Organization

### Data Directory Structure

```
DataPipeline/data/
├── raw/                         # Raw Kaggle datasets
│   ├── tb/                      # Tuberculosis dataset
│   └── lung_cancer_ct_scan/    # Lung cancer dataset
│
├── preprocessed/                # Preprocessed images (224x224, JPEG)
│   ├── tb/
│   │   └── YYYY/MM/DD/         # Partitioned by date
│   │       ├── train/
│   │       │   ├── Normal/
│   │       │   └── Tuberculosis/
│   │       └── test/
│   └── lung_cancer_ct_scan/
│       └── YYYY/MM/DD/
│           ├── train/
│           │   ├── adenocarcinoma/
│           │   ├── benign/
│           │   ├── large_cell_carcinoma/
│           │   ├── malignant/
│           │   ├── normal/
│           │   └── squamous_cell_carcinoma/
│           └── test/
│
├── synthetic_metadata/          # Generated patient metadata
│   └── YYYY/MM/DD/
│       ├── tb_metadata.csv
│       └── lung_cancer_metadata.csv
│
├── synthetic_metadata_mitigated/  # Bias-mitigated metadata
│   └── YYYY/MM/DD/
│
├── ge_outputs/                  # Great Expectations outputs
│   ├── baseline/                # Baseline statistics
│   ├── drift/                   # Drift detection results
│   ├── bias_analysis/           # Bias detection reports
│   ├── eda/                     # Exploratory data analysis
│   ├── reports/                 # HTML visualization reports
│   ├── schemas/                 # Schema definitions
│   └── validations/             # Validation results
│
└── RAG/                         # RAG pipeline data
    ├── raw_data/                # Raw scraped content
    │   ├── baseline/            # Baseline scraped data
    │   └── incremental/         # Incremental updates
    ├── chunks/                  # Document chunks
    ├── embeddings/              # Text embeddings
    └── index/                   # FAISS vector index
```

### Data Version Control (DVC)

All data is versioned using DVC and stored in Google Cloud Storage:

- **Remote Storage**: `gs://medscan-data/dvc-storage/`
- **Remotes**: 
  - `vision`: Vision pipeline data
  - `rag`: RAG pipeline data
- **Tracking Files**: `.dvc` files track data versions and point to GCS storage

---

## Workflow Overview

### Vision Pipeline Workflow

1. **Data Acquisition** (`fetch_data.py`)
   - Downloads TB and Lung Cancer datasets from Kaggle
   - Validates download integrity
   - Organizes data into date-partitioned structure

2. **Preprocessing** (`process_tb.py`, `process_lungcancer.py`)
   - Resizes images to 224x224
   - Converts to JPEG format
   - Organizes into train/test splits
   - Creates class-balanced directories

3. **Synthetic Metadata Generation** (`baseline_synthetic_data_generator.py`)
   - Generates realistic patient demographics using Faker
   - Creates metadata CSV files
   - Includes: age, gender, symptoms, medical history, etc.

4. **Validation** (`schema_statistics.py`)
   - Schema validation (field types, constraints)
   - Statistical drift detection (baseline vs new data)
   - Bias detection across demographic slices
   - Anomaly detection (missing values, outliers)

5. **Bias Mitigation** (if bias detected)
   - Resamples underrepresented groups
   - Generates mitigated metadata CSV
   - Updates data for training

6. **DVC Tracking**
   - Tracks all data versions
   - Pushes to GCS remote storage

7. **Email Alerts** (if issues detected)
   - Sends alerts for anomalies, drift, or bias

### RAG Pipeline Workflow

1. **Web Scraping** (`scraper.py`)
   - Scrapes 40+ medical websites (CDC, WHO, Mayo Clinic, etc.)
   - Extracts content from HTML/PDFs
   - Validates content quality

2. **Baseline Creation/Validation** (`analysis/validator.py`)
   - Creates baseline from first scrape
   - Validates subsequent scrapes against baseline
   - Detects drift in content

3. **Chunking** (`chunking.py`)
   - Splits documents into optimal chunks
   - Preserves context and semantics
   - Creates chunk metadata

4. **Embedding Generation** (`embedding.py`)
   - Generates embeddings using llm-embedder
   - Creates dense vector representations
   - Validates embedding quality

5. **Index Creation** (`indexing.py`)
   - Builds FAISS vector index
   - Enables fast similarity search
   - Optimizes for retrieval performance

6. **DVC Tracking**
   - Tracks chunks, embeddings, and index
   - Version controls knowledge base

### Model Training Workflow

1. **Vision Model Training**
   - Reads preprocessed data from latest partition
   - Trains ResNet50, ViT, and Custom CNN in parallel
   - Evaluates on test set
   - Selects best model based on accuracy
   - Logs to MLflow
   - Saves models to `ModelDevelopment/data/models/`

2. **RAG Model Selection**
   - Evaluates different retrieval methods
   - Tests various prompt templates
   - Measures bias and fairness
   - Selects best combination
   - Deploys for inference

### Inference Workflow

1. **Vision Inference**
   - Loads best trained model
   - Preprocesses input image
   - Generates prediction with confidence scores
   - Applies bias checks

2. **RAG Inference**
   - Receives medical question
   - Retrieves relevant chunks from FAISS index
   - Generates answer using LLM
   - Applies guardrails (safety checks)
   - Returns answer with citations

---

## File Naming Conventions

### Data Files
- **Raw Data**: Organized by dataset name (`tb/`, `lung_cancer_ct_scan/`)
- **Preprocessed Data**: Partitioned by date (`YYYY/MM/DD/`)
- **Metadata**: Named as `{dataset}_metadata.csv`
- **Reports**: Named with timestamp and dataset (`{dataset}_{timestamp}_report.html`)

### Script Files
- **Pipeline Scripts**: Descriptive names (`fetch_data.py`, `process_tb.py`)
- **Analysis Scripts**: Located in `analysis/` subdirectory
- **Utility Scripts**: Located in `common/` subdirectory

### Model Files
- **Saved Models**: `{architecture}_{best|final}.keras`
- **Training Logs**: `{architecture}_training.log`
- **Metadata**: `training_metadata.json`, `training_summary.json`

---

## Testing Structure

Tests mirror the source code structure:

```
DataPipeline/tests/
├── data_acquisition/           # Tests for data fetching
├── data_preprocessing/         # Tests for preprocessing
└── [RAG tests]                # Tests for RAG pipeline

ModelDevelopment/
└── RAG/test/                  # Tests for RAG inference
```

Test files follow naming convention: `*_test.py` or `test_*.py`

---

## Dependencies

### Root `requirements.txt`
- Core data processing: `numpy`, `pandas`, `scipy`, `pyyaml`
- RAG pipeline: `torch`, `sentence-transformers`, `faiss-cpu`, `beautifulsoup4`
- Vision pipeline: `Pillow`, `kaggle`, `kagglehub`
- Validation: `great-expectations`, `mlflow`
- Bias detection: `fairlearn`, `scikit-learn`, `tensorflow`
- Cloud: `google-cloud-storage`, `dvc[gs]`
- Testing: `pytest`, `pytest-cov`, `pytest-mock`

### Module-Specific Requirements
- **`ModelDevelopment/Vision/requirements.txt`**: Vision-specific dependencies
- **`ModelDevelopment/RAG/requirements.txt`**: RAG-specific dependencies

---

## Environment Variables

Key environment variables (set in `airflow/.env`):

| Variable | Purpose |
|----------|---------|
| `GCP_PROJECT_ID` | Google Cloud Project ID |
| `GCS_BUCKET_NAME` | Google Cloud Storage bucket name |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON |
| `AIRFLOW__CORE__FERNET_KEY` | Airflow encryption key |
| `SMTP_*` | Email alert configuration |
| `VISION_MAX_ANOMALY_PCT` | Vision pipeline anomaly threshold |
| `RAG_SCRAPING_MIN_SUCCESS` | RAG scraping success threshold |

---

## Quick Reference

### Key Directories
- **Data Processing**: `DataPipeline/scripts/`
- **Model Training**: `ModelDevelopment/Vision/`, `ModelDevelopment/RAG/`
- **Orchestration**: `airflow/dags/`
- **Configuration**: `DataPipeline/config/`, `ModelDevelopment/config/`
- **Tests**: `DataPipeline/tests/`

### Key Entry Points
- **Vision Pipeline**: `airflow/dags/medscan_vision_pipeline_gcs.py`
- **RAG Pipeline**: `airflow/dags/rag_data_pipeline.py`
- **Vision Training**: `ModelDevelopment/Vision/train_*.py`
- **RAG Inference**: `ModelDevelopment/RAG/ModelInference/RAG_inference.py`

### Key Utilities
- **GCS Management**: `DataPipeline/scripts/common/gcs_manager.py`
- **DVC Helpers**: `DataPipeline/scripts/common/dvc_helper.py`
- **Email Alerts**: `DataPipeline/scripts/data_preprocessing/alert_utils.py`
- **Model Monitoring**: `ModelDevelopment/common/monitoring.py`

---

## Contributing

When adding new files:

1. **Data Scripts**: Add to appropriate subdirectory in `DataPipeline/scripts/`
2. **Model Code**: Add to `ModelDevelopment/Vision/` or `ModelDevelopment/RAG/`
3. **Tests**: Add corresponding test in `tests/` directory
4. **Configs**: Add configuration files in appropriate `config/` directory
5. **Documentation**: Update this file and relevant README files

---

**Last Updated**: Generated from project structure analysis
**Maintained By**: MedScan AI Development Team

