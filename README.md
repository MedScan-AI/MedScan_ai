# MedScan AI

AI-powered radiological assistant for CT scan analysis with explainable AI and patient engagement RAG to answer questions about their report's content.

Note: This project is in the initial development phase. Repository structure, naming conventions, technology choices, and implementation details are subject to change based on ongoing technical discussions and requirements refinement.

## About Ethos AI

Scope - Click [here]() 

## Requirements

- Python 3.10+
- Google Cloud Platform account
- Docker (optional)
- gcloud CLI

## Installation

```bash
# Clone repository
git clone [TBD]
cd ethos-ai

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
ethos-ai/
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
docker build -t ethos-ai .

# Run container
docker run -p 8080:8080 --env-file .env ethos-ai
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

- Brain Cancer MRI Dataset
- CT Scan Lung Cancer Dataset
- Diabetic Retinopathy Dataset

### Data Pipeline

TBD - Pipeline documentation pending implementation

## Contributing

TBD

## License

TBD

## Support

TBD
