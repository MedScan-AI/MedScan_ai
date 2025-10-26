"""GCP configuration for RAG pipeline."""
import os
from pathlib import Path
import yaml

# Load RAG pipeline config
CONFIG_FILE = Path(__file__).parent / "rag_pipeline.yml"
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

# GCP Settings
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "medscanai-476203")
GCS_BUCKET_NAME = config['gcs']['bucket_name']
GCS_BUCKET_PATH = f"gs://{GCS_BUCKET_NAME}"

# Service Account
SERVICE_ACCOUNT_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(Path.home() / "gcp-service-account.json")
)

# GCS Paths for RAG
GCS_PATHS = {
    "raw_baseline": f"{GCS_BUCKET_PATH}/RAG/raw_data/baseline",
    "raw_incremental": f"{GCS_BUCKET_PATH}/RAG/raw_data/incremental",
    "merged": f"{GCS_BUCKET_PATH}/RAG/merged",
    "validation": f"{GCS_BUCKET_PATH}/RAG/validation",
    "chunked_data": f"{GCS_BUCKET_PATH}/RAG/chunked_data",
    "index": f"{GCS_BUCKET_PATH}/RAG/index",
}

# Local Paths for RAG
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "RAG"

LOCAL_PATHS = {
    "raw_data": DATA_DIR / "raw_data",
    "merged": DATA_DIR / "merged",
    "validation": DATA_DIR / "validation",
    "chunked_data": DATA_DIR / "chunked_data",
    "index": DATA_DIR / "index",
}

# Create local directories
for path in LOCAL_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)


def get_config():
    """Load full pipeline configuration."""
    return config


def get_scraping_urls():
    """Get scraping URLs from config."""
    return config['scraping']['urls']


def get_embedding_model():
    """Get embedding model name from config."""
    return config['embedding']['model_name']