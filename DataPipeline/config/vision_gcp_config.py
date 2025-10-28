"""
GCP Configuration for Vision Pipeline
Complete configuration matching enhanced alert system
"""
from pathlib import Path
import os

# ============================================================================
# GCP Project Configuration
# ============================================================================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "medscanai-476500")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "medscan-pipeline-medscanai-476500")
SERVICE_ACCOUNT_PATH = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", 
                                      str(Path.home() / "gcp-service-account.json")))

# ============================================================================
# GCS Base Paths (all under vision/)
# ============================================================================
GCS_BASE = {
    'raw': 'vision/raw/',
    'preprocessed': 'vision/preprocessed/',
    'metadata': 'vision/metadata/',
    'metadata_mitigated': 'vision/metadata_mitigated/',
    'ge_outputs': 'vision/ge_outputs/',
    'mlflow': 'vision/mlflow/',
}

# ============================================================================
# Local Temp Paths
# ============================================================================
PROJECT_ROOT = Path("/opt/airflow/DataPipeline")
LOCAL_PATHS = {
    'temp': PROJECT_ROOT / "temp",
    'data': PROJECT_ROOT / "data",
    'logs': PROJECT_ROOT / "logs",
    'ge_outputs': PROJECT_ROOT / "data" / "ge_outputs",
}

# ============================================================================
# Email Alert Configuration
# ============================================================================
ALERT_CONFIG = {
    'enabled': True,
    'email_recipients': [
        os.getenv("ALERT_EMAIL_PRIMARY", "harshitha8.shekar@gmail.com"),
        os.getenv("ALERT_EMAIL_SECONDARY", "chandrashekar.h@northeastern.edu"),
    ],
    'smtp_settings': {
        'host': os.getenv("SMTP_HOST", "smtp.gmail.com"),
        'port': int(os.getenv("SMTP_PORT", "587")),
        'use_tls': True,
        'username': os.getenv("SMTP_USER", "harshitha8.shekar@gmail.com"),
        'password': os.getenv("SMTP_PASSWORD", ""),  # Use App Password for Gmail
    }
}

# ============================================================================
# Alert Thresholds (matching original DAG logic)
# ============================================================================
ALERT_THRESHOLDS = {
    # Validation Thresholds
    'validation_max_anomaly_pct': 25.0,          # Alert if >25% of data has anomalies
    'validation_max_anomalies': 10,              # Alert if >10 anomalies found
    'validation_max_drift_features': 3,          # Alert if >3 features show drift
    'validation_min_completeness': 0.95,         # Alert if <95% data completeness
    
    # Data Volume Thresholds
    'expected_tb_images': 700,                   # Expected TB images
    'expected_lc_images': 1000,                  # Expected Lung Cancer images
    'variance_tolerance': 0.1,                   # ±10% tolerance
    'min_image_count': 500,                      # Alert if <500 images total
    
    # Processing Thresholds
    'preprocessing_min_success_rate': 0.95,      # Alert if <95% images processed
    'max_upload_time_minutes': 30,               # Alert if upload takes >30 min
    'max_preprocessing_time_minutes': 60,        # Alert if preprocessing >60 min
    
    # Bias Thresholds
    'bias_max_demographic_variance': 0.3,        # Alert if demographic variance >30%
    'bias_min_group_representation': 0.05,       # Alert if any group <5%
}

# ============================================================================
# Dashboard and Monitoring URLs
# ============================================================================
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
DASHBOARD_URL = os.getenv("DASHBOARD_URL", 
                         f"{AIRFLOW_URL}/dags/medscan_vision_pipeline_gcs/grid")

# ============================================================================
# Helper Functions
# ============================================================================

def get_gcs_path(data_type: str, dataset: str, partition: str = None) -> str:
    """
    Generate GCS path for vision pipeline data.
    
    Args:
        data_type: 'raw', 'preprocessed', 'metadata', 'ge_outputs', etc.
        dataset: 'tb' or 'lung_cancer'
        partition: Optional date partition 'YYYY/MM/DD'
    
    Returns:
        Full GCS path (e.g., 'vision/raw/tb/2025/10/28/')
    
    Examples:
        >>> get_gcs_path('raw', 'tb', '2025/10/28')
        'vision/raw/tb/2025/10/28/'
        
        >>> get_gcs_path('preprocessed', 'lung_cancer')
        'vision/preprocessed/lung_cancer/'
    """
    base = GCS_BASE.get(data_type, f'vision/{data_type}/')
    path = f"{base}{dataset}/"
    
    if partition:
        partition = partition.strip('/')  # Remove leading/trailing slashes
        path = f"{path}{partition}/"
    
    return path


def get_alert_config() -> dict:
    """
    Get alert configuration.
    
    Returns:
        Dictionary with email recipients and SMTP settings
    """
    return ALERT_CONFIG


def get_alert_thresholds() -> dict:
    """
    Get alert thresholds.
    
    Returns:
        Dictionary with all threshold values
    """
    return ALERT_THRESHOLDS


def should_alert_on_drift(num_drifted_features: int) -> bool:
    """
    Check if drift detection should trigger alert.
    
    Args:
        num_drifted_features: Number of features showing drift
    
    Returns:
        True if alert should be sent
    """
    return num_drifted_features > ALERT_THRESHOLDS['validation_max_drift_features']


def should_alert_on_anomalies(num_anomalies: int, total_records: int = None) -> bool:
    """
    Check if anomaly count should trigger alert.
    
    Args:
        num_anomalies: Number of anomalies detected
        total_records: Total number of records (optional)
    
    Returns:
        True if alert should be sent
    """
    # Absolute count check
    if num_anomalies > ALERT_THRESHOLDS['validation_max_anomalies']:
        return True
    
    # Percentage check if total_records provided
    if total_records and total_records > 0:
        anomaly_pct = (num_anomalies / total_records) * 100
        return anomaly_pct > ALERT_THRESHOLDS['validation_max_anomaly_pct']
    
    return False


def should_alert_on_volume(actual: int, expected: int) -> bool:
    """
    Check if data volume variance should trigger alert.
    
    Args:
        actual: Actual number of items
        expected: Expected number of items
    
    Returns:
        True if alert should be sent
    """
    if expected == 0:
        return actual == 0
    
    tolerance = ALERT_THRESHOLDS['variance_tolerance']
    lower_bound = expected * (1 - tolerance)
    upper_bound = expected * (1 + tolerance)
    
    return not (lower_bound <= actual <= upper_bound)


def should_alert_on_processing(success_count: int, total_count: int) -> bool:
    """
    Check if processing success rate should trigger alert.
    
    Args:
        success_count: Number of successfully processed items
        total_count: Total number of items
    
    Returns:
        True if alert should be sent
    """
    if total_count == 0:
        return True
    
    success_rate = success_count / total_count
    return success_rate < ALERT_THRESHOLDS['preprocessing_min_success_rate']


def get_dashboard_url(partition: str = "") -> str:
    """
    Get dashboard URL for monitoring.
    
    Args:
        partition: Optional partition date for filtering
    
    Returns:
        URL to Airflow DAG dashboard
    """
    return DASHBOARD_URL


def get_gcs_console_url(gcs_path: str) -> str:
    """
    Get GCS console URL for a specific path.
    
    Args:
        gcs_path: GCS path (e.g., 'vision/raw/tb/2025/10/28')
    
    Returns:
        URL to GCS console browser
    
    Example:
        >>> get_gcs_console_url('vision/raw/tb/2025/10/28')
        'https://console.cloud.google.com/storage/browser/medscan-pipeline-medscanai-476500/vision/raw/tb/2025/10/28'
    """
    gcs_path = gcs_path.strip('/')
    return f"https://console.cloud.google.com/storage/browser/{GCS_BUCKET_NAME}/{gcs_path}"


def get_mlflow_url() -> str:
    """Get MLflow tracking UI URL."""
    return MLFLOW_URL


def get_local_path(path_type: str) -> Path:
    """
    Get local filesystem path.
    
    Args:
        path_type: 'temp', 'data', 'logs', 'ge_outputs'
    
    Returns:
        Path object
    """
    return LOCAL_PATHS.get(path_type, PROJECT_ROOT / path_type)


# ============================================================================
# Email Alert Templates
# ============================================================================
ALERT_TEMPLATES = {
    'drift_detected': """
🚨 DATA DRIFT ALERT - MedScan Vision Pipeline

Partition: {partition}
Dataset: {dataset}
Drifted Features: {num_drifted}
Threshold: {threshold}

Action Required:
- Review drift analysis report in GCS: {gcs_url}
- Consider model retraining if drift is significant
- Investigate data collection process changes

Dashboard: {dashboard_url}
MLflow: {mlflow_url}
""",
    
    'bias_detected': """
⚠️ BIAS ALERT - MedScan Vision Pipeline

Partition: {partition}
Dataset: {dataset}
Bias Metrics:
{bias_details}

Action Required:
- Review bias analysis report in GCS: {gcs_url}
- Consider data augmentation or resampling
- Validate against clinical guidelines
- Check demographic distribution

Dashboard: {dashboard_url}
""",
    
    'volume_mismatch': """
📊 DATA VOLUME ALERT - MedScan Vision Pipeline

Partition: {partition}
Dataset: {dataset}
Expected: {expected} images
Actual: {actual} images
Variance: {variance}%

Action Required:
- Verify Kaggle download completed successfully
- Check for data corruption or missing files
- Review preprocessing logs for errors
- Investigate data acquisition issues

Dashboard: {dashboard_url}
""",
    
    'processing_failure': """
❌ PIPELINE FAILURE - MedScan Vision Pipeline

Task: {task_name}
Partition: {partition}
Error: {error_message}

Stack Trace:
{stack_trace}

Action Required:
- Check Airflow logs for detailed error information
- Verify GCS bucket permissions and quotas
- Ensure all dependencies are available
- Check system resources (CPU, memory, disk)

Airflow: {airflow_url}
GCS Bucket: gs://{bucket_name}
""",
    
    'validation_success': """
✅ PIPELINE SUCCESS - MedScan Vision Pipeline

Partition: {partition}

Summary:
- TB Images: {tb_count} (raw), {tb_processed} (preprocessed)
- Lung Cancer Images: {lc_count} (raw), {lc_processed} (preprocessed)
- Metadata Records: {metadata_count}
- Validation Reports: {validation_files}

Quality Checks:
✓ All validation checks passed
✓ No significant drift detected
✓ Bias within acceptable limits
✓ Data completeness: 100%

GCS Bucket: gs://{bucket_name}/vision/
Airflow: {airflow_url}
MLflow: {mlflow_url}
"""
}


def get_alert_template(alert_type: str) -> str:
    """
    Get email template for alert type.
    
    Args:
        alert_type: One of 'drift_detected', 'bias_detected', 'volume_mismatch',
                   'processing_failure', 'validation_success'
    
    Returns:
        Email template string
    """
    return ALERT_TEMPLATES.get(alert_type, "Alert: {message}")


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        True if configuration is valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check GCP settings
    if not GCP_PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID is not set")
    
    if not GCS_BUCKET_NAME:
        raise ValueError("GCS_BUCKET_NAME is not set")
    
    if not SERVICE_ACCOUNT_PATH.exists():
        raise ValueError(f"Service account file not found: {SERVICE_ACCOUNT_PATH}")
    
    # Check alert settings
    if ALERT_CONFIG['enabled']:
        if not ALERT_CONFIG['email_recipients']:
            raise ValueError("No email recipients configured")
        
        smtp = ALERT_CONFIG['smtp_settings']
        if not smtp.get('username') or not smtp.get('host'):
            raise ValueError("SMTP settings incomplete")
    
    return True


# ============================================================================
# Configuration Info
# ============================================================================

def print_config_summary():
    """Print configuration summary for debugging."""
    print("="*80)
    print("MedScan Vision Pipeline - Configuration Summary")
    print("="*80)
    print(f"GCP Project ID: {GCP_PROJECT_ID}")
    print(f"GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"Service Account: {SERVICE_ACCOUNT_PATH}")
    print(f"Service Account Exists: {SERVICE_ACCOUNT_PATH.exists()}")
    print(f"\nEmail Alerts Enabled: {ALERT_CONFIG['enabled']}")
    print(f"Email Recipients: {', '.join(ALERT_CONFIG['email_recipients'])}")
    print(f"SMTP Host: {ALERT_CONFIG['smtp_settings']['host']}")
    print(f"\nAirflow URL: {AIRFLOW_URL}")
    print(f"MLflow URL: {MLFLOW_URL}")
    print(f"Dashboard URL: {DASHBOARD_URL}")
    print("="*80)


# Auto-validate on import (optional, can be commented out)
try:
    validate_config()
except ValueError as e:
    print(f"⚠️ Configuration Warning: {e}")