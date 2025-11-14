"""GCP Configuration for MedScan AI"""
import os
import warnings
from pathlib import Path
from typing import Dict
import yaml


# GCP Core Settings
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "medscanai-476203")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "medscan-data")

# Service account path - auto-detect environment
if os.path.exists("/opt/airflow/gcp-service-account.json"):
    # Running in Docker
    SERVICE_ACCOUNT_PATH = Path("/opt/airflow/gcp-service-account.json")
elif os.path.exists(str(Path.home() / "gcp-service-account.json")):
    # Running locally on Mac/Linux
    SERVICE_ACCOUNT_PATH = Path.home() / "gcp-service-account.json"
else:
    # Fallback to environment variable
    SERVICE_ACCOUNT_PATH = Path(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", 
                 str(Path.home() / "gcp-service-account.json"))
    )


# Project Paths
# Auto-detect if running in Docker or locally
if os.path.exists("/opt/airflow"):
    # Running in Docker
    PROJECT_ROOT = Path("/opt/airflow/DataPipeline")
else:
    # Running locally - use current file's parent directory
    PROJECT_ROOT = Path(__file__).parent.parent

LOCAL_PATHS = {
    'temp': PROJECT_ROOT / "temp",
    'data': PROJECT_ROOT / "data",
    'logs': PROJECT_ROOT / "logs",
    'ge_outputs': PROJECT_ROOT / "data" / "ge_outputs",
    'mlflow_store': PROJECT_ROOT / "data" / "mlflow_store", 
    'validation': PROJECT_ROOT / "data" / "RAG" / "validation",
    'index': PROJECT_ROOT / "data" / "RAG" / "index",
}

# Create directories only if we have permission
try:
    for path in LOCAL_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError) as e:
    pass


# GCS Paths (Bucket Structure)
GCS_PATHS = {
    # RAG paths
    'rag_config': 'RAG/config/',
    'rag_raw_baseline': 'RAG/raw_data/baseline/',
    'rag_raw_incremental': 'RAG/raw_data/incremental/',
    'rag_validation': 'RAG/validation/',
    'rag_validation_reports': 'RAG/validation/reports/',
    'rag_chunks': 'RAG/chunks/',
    'rag_embeddings': 'RAG/embeddings/',
    'rag_index': 'RAG/index/',
    
    # Vision paths
    'vision_raw': 'vision/raw/',
    'vision_preprocessed': 'vision/preprocessed/',
    'vision_metadata': 'vision/metadata/',
    'vision_metadata_mitigated': 'vision/metadata_mitigated/',
    'vision_ge_outputs': 'vision/ge_outputs/',
    'vision_mlflow': 'vision/mlflow/',
}


# Alert Configuration
ALERT_CONFIG = {
    'enabled': os.getenv("ALERTS_ENABLED", "false").lower() == "true",
    'email_recipients': [
        email.strip() 
        for email in os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
        if email.strip()
    ],
    'smtp_settings': {
        'host': os.getenv("SMTP_HOST", "smtp.gmail.com"),
        'port': int(os.getenv("SMTP_PORT", "587")),
        'use_tls': os.getenv("SMTP_USE_TLS", "true").lower() == "true",
        'username': os.getenv("SMTP_USER", ""),
        'password': os.getenv("SMTP_PASSWORD", ""),
    }
}


# Alert Thresholds
# Vision Pipeline Thresholds
VISION_THRESHOLDS = {
    'validation_max_anomaly_pct': float(os.getenv("VISION_MAX_ANOMALY_PCT", "25.0")),
    'validation_max_anomalies': int(os.getenv("VISION_MAX_ANOMALIES", "10")),
    'validation_max_drift_features': int(os.getenv("VISION_MAX_DRIFT_FEATURES", "3")),
    'validation_min_completeness': float(os.getenv("VISION_MIN_COMPLETENESS", "0.95")),
    'expected_tb_images': int(os.getenv("VISION_EXPECTED_TB_IMAGES", "700")),
    'expected_lc_images': int(os.getenv("VISION_EXPECTED_LC_IMAGES", "1000")),
    'variance_tolerance': float(os.getenv("VISION_VARIANCE_TOLERANCE", "0.1")),
    'min_image_count': int(os.getenv("VISION_MIN_IMAGE_COUNT", "500")),
    'preprocessing_min_success_rate': float(os.getenv("VISION_PREPROCESS_MIN_SUCCESS", "0.95")),
    'bias_max_demographic_variance': float(os.getenv("VISION_BIAS_MAX_VARIANCE", "0.3")),
    'bias_min_group_representation': float(os.getenv("VISION_BIAS_MIN_REPRESENTATION", "0.05")),
}

# RAG Pipeline Thresholds
RAG_THRESHOLDS = {
    'scraping_min_success_rate': float(os.getenv("RAG_SCRAPING_MIN_SUCCESS", "0.7")),
    'validation_max_anomaly_pct': float(os.getenv("RAG_MAX_ANOMALY_PCT", "25.0")),
    'validation_max_drift_features': int(os.getenv("RAG_MAX_DRIFT_FEATURES", "3")),
    'embedding_min_success_rate': float(os.getenv("RAG_EMBEDDING_MIN_SUCCESS", "0.95")),
    'indexing_min_vectors': int(os.getenv("RAG_MIN_VECTORS", "100")),
}


# Monitoring URLs
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")


# YAML Config Loaders (for pipeline-specific settings)
def load_rag_pipeline_config() -> Dict:
    """Load RAG pipeline YAML configuration."""
    config_file = PROJECT_ROOT / "config" / "rag_pipeline.yml"
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_vision_pipeline_config() -> Dict:
    """Load Vision pipeline YAML configuration."""
    config_file = PROJECT_ROOT / "config" / "vision_pipeline.yml"
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


# Helper Functions
def get_gcs_path(pipeline: str, data_type: str, dataset: str = None, partition: str = None) -> str:
    """
    Generate GCS path.
    
    Args:
        pipeline: 'vision' or 'rag'
        data_type: Type of data (e.g., 'raw', 'preprocessed', 'index')
        dataset: Optional dataset name (e.g., 'tb', 'lung_cancer')
        partition: Optional date partition 'YYYY/MM/DD'
    
    Returns:
        Full GCS path
    """
    if pipeline == 'vision':
        path = f"vision/{data_type}/"
    elif pipeline == 'rag':
        path = f"RAG/{data_type}/"
    else:
        raise ValueError(f"Invalid pipeline: {pipeline}. Use 'vision' or 'rag'")
    
    if dataset:
        path = f"{path}{dataset}/"
    
    if partition:
        partition = partition.strip('/')
        path = f"{path}{partition}/"
    
    return path


def get_gcs_console_url(gcs_path: str) -> str:
    """Get GCS console URL for a path."""
    gcs_path = gcs_path.strip('/')
    return f"https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/{gcs_path}"


def get_dashboard_url(dag_id: str = None) -> str:
    """Get Airflow dashboard URL."""
    if dag_id:
        return f"{AIRFLOW_URL}/dags/{dag_id}/grid"
    return f"{AIRFLOW_URL}/home"


def get_local_path(path_type: str) -> Path:
    """Get local filesystem path."""
    return LOCAL_PATHS.get(path_type, PROJECT_ROOT / path_type)


def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    if not PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID is not set")
    
    if not BUCKET_NAME:
        raise ValueError("GCS_BUCKET_NAME is not set")
    
    if not SERVICE_ACCOUNT_PATH.exists():
        warnings.warn(
            f"Service account file not found: {SERVICE_ACCOUNT_PATH}\n"
            f"GCS operations will fail. Ensure file exists at:\n"
            f"  Docker: /opt/airflow/gcp-service-account.json\n"
            f"  Local: ~/gcp-service-account.json",
            UserWarning
        )
    
    if ALERT_CONFIG['enabled'] and not ALERT_CONFIG['email_recipients']:
        raise ValueError("No email recipients configured but alerts are enabled")
    
    return True


def print_config_summary():
    """Print configuration summary."""
    print("MedScan AI - GCP Configuration")
    print(f"Environment: {'Docker' if os.path.exists('/opt/airflow') else 'Local'}")
    print(f"GCP Project ID: {PROJECT_ID}")
    print(f"GCS Bucket: {BUCKET_NAME}")
    print(f"  - RAG Path: gs://{BUCKET_NAME}/RAG/")
    print(f"  - Vision Path: gs://{BUCKET_NAME}/vision/")
    print(f"Service Account: {SERVICE_ACCOUNT_PATH}")
    print(f"Service Account Exists: {SERVICE_ACCOUNT_PATH.exists()}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"\nEmail Alerts Enabled: {ALERT_CONFIG['enabled']}")
    if ALERT_CONFIG['enabled']:
        print(f"Email Recipients: {', '.join(ALERT_CONFIG['email_recipients'])}")
    print(f"\nAirflow URL: {AIRFLOW_URL}")
    print(f"MLflow URL: {MLFLOW_URL}")

# Auto-validate on import (but don't fail on warnings)
if os.getenv("SKIP_CONFIG_VALIDATION", "false").lower() != "true":
    try:
        validate_config()
    except ValueError as e:
        print(f"Config Error: {e}")
        raise
    except Warning:
        # Warnings are OK during import
        pass