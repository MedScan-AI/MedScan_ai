"""GCP configuration for RAG pipeline."""
import os
from pathlib import Path
import yaml

CONFIG_FILE = Path(__file__).parent / "rag_pipeline.yml"

def load_config():
    """Load configuration from YAML."""
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# GCP Settings
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", config['gcs']['project_id'])
GCS_BUCKET_NAME = config['gcs']['bucket_name']
GCS_BUCKET_PATH = f"gs://{GCS_BUCKET_NAME}"

# Service Account (in Docker)
SERVICE_ACCOUNT_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/opt/airflow/gcp-service-account.json"
)

# Paths (Docker environment)
PROJECT_ROOT = Path("/opt/airflow")
DATA_DIR = PROJECT_ROOT / "data" / "RAG"

LOCAL_PATHS = {
    "raw_data": DATA_DIR / "raw_data",
    "merged": DATA_DIR / "merged",
    "validation": DATA_DIR / "validation",
    "chunked_data": DATA_DIR / "chunked_data",
    "index": DATA_DIR / "index",
}

# Create directories
for path in LOCAL_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Getters
def get_scraping_urls():
    return config['scraping']['urls']

def get_embedding_model():
    return config['embedding']['model_name']

def get_validation_config():
    return config['validation']