# MedScan AI

AI-powered radiological assistant for CT scan analysis with explainable AI and patient engagement RAG to answer questions about their report's content.

Note: This project is in the initial development phase. Repository structure, naming conventions, technology choices, and implementation details are subject to change based on ongoing technical discussions and requirements refinement.

## About MedScan AI

Scope - Click [here](docs/Scoping.pdf) 

## Requirements

- Python 3.10+
- Google Cloud Platform account
- Docker (optional)
- gcloud CLI

## Installation

```bash
# Clone repository
git clone https://github.com/rjaditya-2702/MedScan_ai.git
cd medscan-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

```bash
# Copy environment template
cp .env.example .env

# Configure GCP credentials
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
```

### Environment Variables

TBD - Environment variables will be documented upon implementation

## Project Structure

```
medscan-ai/
├── src/
│   ├── models/          # TBD
│   ├── services/        # TBD
│   ├── api/            # TBD
│   └── utils/          # TBD
├── data/               # TBD
├── tests/              # TBD
├── configs/            # TBD
├── docker/             # TBD
├── requirements.txt
├── .env.example
└── README.md
```

### High Level Architecture

![Architecture](assets/high_level_architecture.png)

## Usage

### Running the Application

TBD - Application startup commands pending implementation

### API Endpoints

TBD - API documentation pending implementation

### Docker

```bash
# Build image
docker build -t medscan-ai .

# Run container
docker run -p 8080:8080 --env-file .env medscan-ai
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

### Code Style

- PEP 8 compliance
- Black formatter
- 80% minimum test coverage

## Deployment

### GCP Deployment

TBD - Deployment instructions pending infrastructure setup

### Required GCP Services

- Cloud Storage
- Vertex AI
- Cloud Run
- Cloud SQL
- Pub/Sub
- Cloud Composer

## Monitoring

- Cloud Monitoring dashboards
- Cloud Logging for application logs
- Vertex AI Model Monitoring

## Data

### Datasets Used

- CT Scan Lung Cancer Dataset
- Tuberculosis TB Chest X-ray Dataset

### Vision Pipeline

**Setup:**
```bash
cd DataPipeline
pip install -r requirements.txt
```

**Data Acquisition:**
```bash
# Download all datasets
python scripts/data_acquisition/fetch_data.py --config config/vision_pipeline.yml

# Download specific dataset
python scripts/data_acquisition/fetch_data.py --dataset lung_cancer_ct_scan
```

**Data Preprocessing:**
```bash
# Preprocess tuberculosis chest X-rays (reads from latest partition)
python scripts/data_preprocessing/process_tb.py --config config/vision_pipeline.yml

# Preprocess lung cancer CT scans (reads from latest partition)
python scripts/data_preprocessing/process_lungcancer.py --config config/vision_pipeline.yml
```

**Generate Synthetic Patient Metadata:**
```bash
# Generate synthetic patient data for latest preprocessed partition
python scripts/data_preprocessing/baseline_synthetic_data_generator.py --config config/synthetic_data.yml
```

**Data Validation, Schema Tracking & EDA:**
```bash
# Run comprehensive data validation pipeline:
# - Generate statistics
# - Infer and track schema changes
# - Perform exploratory data analysis (EDA)
# - Validate data against expectations
# - Detect drift between partitions
# - Generate HTML reports
python scripts/data_preprocessing/schema_statistics.py --config config/metadata.yml

# View MLflow tracking UI
mlflow ui
# Then open: http://localhost:5000
```

**Run Tests:**
```bash
# Run all tests
pytest tests/

# Data acquisition tests
pytest tests/data_acquisition/fetch_data_test.py -v

# Data preprocessing tests
pytest tests/data_preprocessing/preprocess_tb_test.py -v
pytest tests/data_preprocessing/preprocess_lung_cancer_test.py -v
pytest tests/data_preprocessing/baseline_synthetic_data_generator_test.py -v
pytest tests/data_preprocessing/schema_statistics_test.py -v
```

**Output Structure (Partitioned by Date):**
```
Data-Pipeline/
├── data/
│   ├── raw/                          # Raw downloaded data
│   │   ├── tb/
│   │   │   └── YYYY/MM/DD/           # Partitioned by date
│   │   └── lung_cancer_ct_scan/
│   │       └── YYYY/MM/DD/
│   ├── preprocessed/                 # Preprocessed images (224×224, JPEG)
│   │   ├── tb/
│   │   │   └── YYYY/MM/DD/train/{class}/*.jpg
│   │   └── lung_cancer_ct_scan/
│   │       └── YYYY/MM/DD/Training/{class}/*.jpg
│   ├── synthetic_metadata/           # Patient metadata CSVs
│   │   └── YYYY/MM/DD/
│   │       ├── tb_patients.csv
│   │       └── lung_cancer_ct_scan_patients.csv
│   └── ge_outputs/                   # Validation, EDA, and drift reports
│       ├── baseline/YYYY/MM/DD/      # Baseline statistics
│       ├── schemas/YYYY/MM/DD/       # Schema definitions
│       ├── validations/YYYY/MM/DD/   # Validation results
│       ├── eda/YYYY/MM/DD/           # EDA JSON and HTML reports
│       ├── drift/YYYY/MM/DD/         # Drift detection reports
│       └── reports/YYYY/MM/DD/       # HTML visualization reports
```

### Complete Pipeline Workflow

**Step 1: Download Data**
```bash
cd Data-Pipeline
python scripts/data_acquisition/fetch_data.py --config config/prod/vision_pipeline.yml
```
Output: `data/raw/{dataset}/YYYY/MM/DD/`

**Step 2: Preprocess Images**
```bash
python scripts/data_preprocessing/process_tb.py --config config/prod/vision_pipeline.yml
python scripts/data_preprocessing/process_lungcancer.py --config config/prod/vision_pipeline.yml
```
Output: `data/preprocessed/{dataset}/YYYY/MM/DD/`

**Step 3: Generate Synthetic Metadata**
```bash
python scripts/data_preprocessing/baseline_synthetic_data_generator.py --config config/prod/synthetic_data.yml
```
Output: `data/synthetic_metadata/YYYY/MM/DD/*.csv`

**Step 4: Validate & Analyze Data**
```bash
python scripts/data_preprocessing/schema_statistics.py --config config/prod/metadata.yml
```
Output: `data/ge_outputs/**/YYYY/MM/DD/`

**Step 5: View Results**
```bash
# Open MLflow UI to view tracked experiments
mlflow ui
# Then navigate to: http://localhost:5000

# Or view HTML reports directly:
# - EDA reports: data/ge_outputs/eda/YYYY/MM/DD/*.html
# - Validation reports: data/ge_outputs/reports/YYYY/MM/DD/*.html
```

## Contributing

TBD

## License

TBD

## Support

TBD
