"""GCP configuration for RAG pipeline"""
import os
from pathlib import Path
import yaml

CONFIG_FILE = Path(__file__).parent / "rag_pipeline.yml"

def load_config():
    """Load configuration."""
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# GCP Settings
GCP_PROJECT_ID = config['gcs']['project_id']
GCS_BUCKET_NAME = config['gcs']['bucket_name']
GCS_BUCKET_PATH = f"gs://{GCS_BUCKET_NAME}"

SERVICE_ACCOUNT_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/opt/airflow/gcp-service-account.json"
)

# Paths (Docker)
PROJECT_ROOT = Path("/opt/airflow")
DATA_DIR = PROJECT_ROOT / "data" / "RAG"

LOCAL_PATHS = {
    "raw_data": DATA_DIR / "raw_data",
    "validation": DATA_DIR / "validation",
    "temp": DATA_DIR / "temp",
    "index": DATA_DIR / "index",
}

# Create directories
for path in LOCAL_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# GCS paths
GCS_PATHS = {
    "urls_file": config['scraping']['urls_file'],
    "raw_baseline": "RAG/raw_data/baseline",
    "raw_incremental": "RAG/raw_data/incremental",
    "validation_baseline": "RAG/validation",
    "validation_reports": "RAG/validation/reports",
    "index": "RAG/index",
}

def get_urls_file_path():
    """Get GCS path for URLs file."""
    return config['scraping']['urls_file']

def get_embedding_model():
    """Get embedding model name."""
    return config['embedding']['model_name']

def get_validation_config():
    """Get validation configuration."""
    return config['validation']

def get_dvc_tracked_items():
    """Get items to track with DVC."""
    return config['dvc']['track_items']

def get_alert_config():
    """Get alert configuration."""
    return config.get('alerts', {})

def get_alert_thresholds():
    """Get alert thresholds."""
    return config.get('alerts', {}).get('thresholds', {})