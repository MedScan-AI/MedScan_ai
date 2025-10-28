"""
MedScan AI Vision Pipeline DAG with GCS Storage
Complete version with:
"""
from datetime import timedelta
from pathlib import Path
import sys
import subprocess
import json
import logging
import shutil
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.email import EmailOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

# Add paths
PROJECT_ROOT = Path("/opt/airflow")
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "config"))

try:
    import vision_gcp_config
    from data_preprocessing.vision_gcs_utils import VisionGCSManager
    from data_preprocessing import alert_utils
except ImportError as e:
    print(f"Import error: {e}")
    raise

logger = logging.getLogger(__name__)

# Get alert emails from environment
ALERT_EMAILS = os.environ.get('ALERT_EMAIL_RECIPIENTS', 
                               ','.join(vision_gcp_config.ALERT_CONFIG['email_recipients'])).split(',')

default_args = {
    'owner': 'vision-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ALERT_EMAILS,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# UTILITY FUNCTIONS

def setup_task_logging(task_name: str, partition: str) -> logging.Logger:
    """Setup file-based logging for task."""
    log_dir = Path('/opt/airflow/DataPipeline/data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{task_name}_{partition.replace('/', '_')}.log"
    
    task_logger = logging.getLogger(f"medscan.{task_name}")
    task_logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    task_logger.addHandler(fh)
    
    return task_logger


def get_partition(context, task_ids=None):
    """Get partition with fallback logic."""
    ti = context['ti']
    
    # Try to get from previous task
    if task_ids:
        if isinstance(task_ids, list):
            for task_id in task_ids:
                partition = ti.xcom_pull(task_ids=task_id, key='partition')
                if partition:
                    return partition
        else:
            partition = ti.xcom_pull(task_ids=task_ids, key='partition')
            if partition:
                return partition
    
    # Fallback to execution_date
    execution_date = context['execution_date']
    partition = execution_date.strftime('%Y/%m/%d')
    logger.info(f"Using execution_date for partition: {partition}")
    return partition

# MAIN PIPELINE TASKS

def download_kaggle_to_gcs(**context):
    """Download from Kaggle and upload to GCS."""
    execution_date = context['execution_date']
    partition = execution_date.strftime('%Y/%m/%d')
    
    task_logger = setup_task_logging('download_kaggle', partition)
    
    logger.info("="*80)
    logger.info("TASK: Download Kaggle Data to GCS - Started")
    logger.info("="*80)
    task_logger.info(f"Starting download task for partition: {partition}")
    
    try:
        gcs = VisionGCSManager()
        
        # Check if already exists
        tb_marker = f"vision/raw/tb/{partition}/.complete"
        lc_marker = f"vision/raw/lung_cancer/{partition}/.complete"
        
        if gcs.blob_exists(tb_marker) and gcs.blob_exists(lc_marker):
            logger.info(f"✓ Data already exists for {partition} - SKIPPING")
            task_logger.info(f"Skipped - markers exist")
            context['ti'].xcom_push(key='partition', value=partition)
            context['ti'].xcom_push(key='skipped', value=True)
            return "Success - Skipped"
        
        # Download from Kaggle
        logger.info("Downloading from Kaggle...")
        task_logger.info("Starting Kaggle download")
        
        cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_acquisition/fetch_data.py',
            '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        
        if result.returncode != 0:
            error_msg = f"Kaggle download failed: {result.stderr}"
            logger.error(error_msg)
            task_logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info("✓ Download complete")
        logger.info(result.stdout)
        task_logger.info("Kaggle download complete")
        
        # Upload to GCS
        local_raw = Path('/opt/airflow/DataPipeline/data/raw')
        tb_count = 0
        lc_count = 0
        
        if local_raw.exists():
            # Upload TB
            for tb_dir in local_raw.glob('tb/*'):
                if tb_dir.is_dir():
                    tb_gcs_path = f"vision/raw/tb/{partition}"
                    logger.info(f"Uploading TB to {tb_gcs_path}")
                    task_logger.info(f"Uploading TB: {tb_dir} -> {tb_gcs_path}")
                    tb_count += gcs.upload_directory(str(tb_dir), tb_gcs_path, max_workers=15)
            
            if tb_count > 0:
                gcs.create_marker(tb_marker, f"TB: {tb_count} files")
                task_logger.info(f"Created TB marker: {tb_count} files")
            
            # Upload Lung Cancer
            for lc_dir in local_raw.glob('lung_cancer*/*'):
                if lc_dir.is_dir():
                    lc_gcs_path = f"vision/raw/lung_cancer/{partition}"
                    logger.info(f"Uploading Lung Cancer to {lc_gcs_path}")
                    task_logger.info(f"Uploading LC: {lc_dir} -> {lc_gcs_path}")
                    lc_count += gcs.upload_directory(str(lc_dir), lc_gcs_path, max_workers=15)
            
            if lc_count > 0:
                gcs.create_marker(lc_marker, f"Lung Cancer: {lc_count} files")
                task_logger.info(f"Created LC marker: {lc_count} files")
        
        context['ti'].xcom_push(key='partition', value=partition)
        context['ti'].xcom_push(key='skipped', value=False)
        context['ti'].xcom_push(key='tb_files', value=tb_count)
        context['ti'].xcom_push(key='lc_files', value=lc_count)
        
        logger.info("="*80)
        logger.info(f"✓ Download complete (TB: {tb_count}, LC: {lc_count})")
        logger.info("="*80)
        task_logger.info(f"Task complete: TB={tb_count}, LC={lc_count}")
        return "Success"
        
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        raise


def preprocess_images_gcs(**context):
    """Preprocess images and upload to GCS."""
    partition = get_partition(context, task_ids='download_kaggle_to_gcs')
    task_logger = setup_task_logging('preprocess_images', partition)
    
    logger.info("="*80)
    logger.info("TASK: Preprocess Images - Started")
    logger.info("="*80)
    task_logger.info(f"Starting preprocessing for partition: {partition}")
    
    try:
        gcs = VisionGCSManager()
        
        # Check if already preprocessed
        tb_marker = f"vision/preprocessed/tb/{partition}/.complete"
        lc_marker = f"vision/preprocessed/lung_cancer/{partition}/.complete"
        
        if gcs.blob_exists(tb_marker) and gcs.blob_exists(lc_marker):
            logger.info("="*80)
            logger.info(f"✓ Preprocessed data already exists for {partition} - SKIPPING")
            logger.info("="*80)
            task_logger.info("Skipped - markers exist")
            context['ti'].xcom_push(key='preprocessing_complete', value=True)
            context['ti'].xcom_push(key='skipped', value=True)
            context['ti'].xcom_push(key='partition', value=partition)
            return "Success - Skipped"
        
        # Run TB preprocessing
        logger.info("Processing TB images...")
        task_logger.info("Starting TB preprocessing")
        
        tb_cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_preprocessing/process_tb.py',
            '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
        ]
        result = subprocess.run(tb_cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        if result.returncode != 0:
            raise Exception(f"TB preprocessing failed: {result.stderr}")
        logger.info("✓ TB complete")
        logger.info(result.stdout)
        task_logger.info("TB preprocessing complete")
        
        # Run Lung Cancer preprocessing
        logger.info("Processing Lung Cancer images...")
        task_logger.info("Starting Lung Cancer preprocessing")
        
        lc_cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_preprocessing/process_lungcancer.py',
            '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
        ]
        result = subprocess.run(lc_cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        if result.returncode != 0:
            raise Exception(f"Lung cancer preprocessing failed: {result.stderr}")
        logger.info("✓ Lung Cancer complete")
        logger.info(result.stdout)
        task_logger.info("Lung Cancer preprocessing complete")
        
        # Upload preprocessed to GCS
        local_preprocessed = Path('/opt/airflow/DataPipeline/data/preprocessed')
        tb_count = 0
        lc_count = 0
        
        if local_preprocessed.exists():
            logger.info("Uploading preprocessed images...")
            task_logger.info("Starting upload to GCS")
            
            for tb_dir in local_preprocessed.glob('tb/*'):
                if tb_dir.is_dir():
                    tb_gcs = f"vision/preprocessed/tb/{partition}"
                    tb_count += gcs.upload_directory(str(tb_dir), tb_gcs, max_workers=15)
            
            if tb_count > 0:
                gcs.create_marker(tb_marker, f"TB preprocessed: {tb_count} files")
                task_logger.info(f"Created TB marker: {tb_count} files")
            
            for lc_dir in local_preprocessed.glob('lung_cancer*/*'):
                if lc_dir.is_dir():
                    lc_gcs = f"vision/preprocessed/lung_cancer/{partition}"
                    lc_count += gcs.upload_directory(str(lc_dir), lc_gcs, max_workers=15)
            
            if lc_count > 0:
                gcs.create_marker(lc_marker, f"Lung Cancer preprocessed: {lc_count} files")
                task_logger.info(f"Created LC marker: {lc_count} files")
        
        context['ti'].xcom_push(key='preprocessing_complete', value=True)
        context['ti'].xcom_push(key='tb_processed', value=tb_count)
        context['ti'].xcom_push(key='lc_processed', value=lc_count)
        context['ti'].xcom_push(key='partition', value=partition)
        
        logger.info("="*80)
        logger.info(f"✓ Preprocessing complete (TB: {tb_count}, LC: {lc_count})")
        logger.info("="*80)
        task_logger.info(f"Task complete: TB={tb_count}, LC={lc_count}")
        return "Success"
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        raise


def generate_metadata_gcs(**context):
    """Generate metadata and upload to GCS."""
    partition = get_partition(context, task_ids=['preprocess_images_gcs', 'download_kaggle_to_gcs'])
    task_logger = setup_task_logging('generate_metadata', partition)
    
    logger.info("="*80)
    logger.info("TASK: Generate Metadata - Started")
    logger.info("="*80)
    task_logger.info(f"Starting metadata generation for partition: {partition}")
    
    try:
        gcs = VisionGCSManager()
        
        # Check if already generated
        tb_marker = f"vision/metadata/tb/{partition}/.complete"
        lc_marker = f"vision/metadata/lung_cancer/{partition}/.complete"
        
        if gcs.blob_exists(tb_marker) and gcs.blob_exists(lc_marker):
            logger.info(f"✓ Metadata already exists for {partition} - SKIPPING")
            task_logger.info("Skipped - markers exist")
            context['ti'].xcom_push(key='partition', value=partition)
            return "Success - Skipped"
        
        # Generate metadata
        logger.info("Generating patient metadata...")
        task_logger.info("Starting metadata generation")
        
        cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_preprocessing/baseline_synthetic_data_generator.py',
            '--config', '/opt/airflow/DataPipeline/config/synthetic_data.yml'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        if result.returncode != 0:
            raise Exception(f"Metadata generation failed: {result.stderr}")
        logger.info("✓ Generation complete")
        logger.info(result.stdout)
        task_logger.info("Metadata generation complete")
        
        # Upload metadata to GCS
        local_metadata = Path('/opt/airflow/DataPipeline/data/synthetic_metadata')
        
        if local_metadata.exists():
            logger.info("Uploading metadata...")
            task_logger.info("Starting metadata upload")
            
            tb_csv = local_metadata / f'tb/{partition}/tb_patients.csv'
            if tb_csv.exists():
                tb_gcs = f"vision/metadata/tb/{partition}/tb_patients.csv"
                gcs.upload_file(str(tb_csv), tb_gcs)
                gcs.create_marker(tb_marker, f"TB metadata uploaded")
                logger.info(f"✓ TB metadata uploaded")
                task_logger.info(f"TB metadata: {tb_csv.stat().st_size} bytes")
            
            lc_csv = local_metadata / f'lung_cancer/{partition}/lung_cancer_ct_scan_patients.csv'
            if lc_csv.exists():
                lc_gcs = f"vision/metadata/lung_cancer/{partition}/lung_cancer_ct_scan_patients.csv"
                gcs.upload_file(str(lc_csv), lc_gcs)
                gcs.create_marker(lc_marker, f"Lung Cancer metadata uploaded")
                logger.info(f"✓ Lung Cancer metadata uploaded")
                task_logger.info(f"LC metadata: {lc_csv.stat().st_size} bytes")
        
        context['ti'].xcom_push(key='metadata_uploaded', value=True)
        context['ti'].xcom_push(key='partition', value=partition)
        
        logger.info("="*80)
        logger.info("✓ Metadata generation complete")
        logger.info("="*80)
        task_logger.info("Task complete")
        return "Success"
        
    except Exception as e:
        logger.error(f"Metadata generation failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        raise


def validate_and_upload_gcs(**context):
    """Run validation and upload to GCS."""
    partition = get_partition(context, task_ids=['generate_metadata_gcs', 'preprocess_images_gcs', 
                                                  'download_kaggle_to_gcs'])
    task_logger = setup_task_logging('validate_and_upload', partition)
    
    logger.info("="*80)
    logger.info("TASK: Validate Data - Started")
    logger.info("="*80)
    logger.info(f"Using partition: {partition}")
    task_logger.info(f"Starting validation for partition: {partition}")
    
    try:
        gcs = VisionGCSManager()
        
        # Check if already validated
        marker = f"vision/ge_outputs/validations/{partition}/.complete"
        if gcs.blob_exists(marker):
            logger.info(f"✓ Validation already complete for {partition} - SKIPPING")
            task_logger.info("Skipped - marker exists")
            context['ti'].xcom_push(key='partition', value=partition)
            return "Success - Skipped"
        
        # Run validation
        logger.info("Running validation...")
        task_logger.info("Starting schema_statistics.py")
        
        cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_preprocessing/schema_statistics.py',
            '--config', '/opt/airflow/DataPipeline/config/metadata.yml'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        if result.returncode != 0:
            raise Exception(f"Validation failed: {result.stderr}")
        logger.info("✓ Validation complete")
        logger.info(result.stdout)
        task_logger.info("Validation complete")
        
        # Upload ge_outputs to GCS
        local_ge = Path('/opt/airflow/DataPipeline/data/ge_outputs')
        uploaded_count = 0
        
        if local_ge.exists():
            logger.info("Uploading validation reports...")
            task_logger.info("Starting ge_outputs upload")
            
            for subdir in ['baseline', 'new_data', 'schemas', 'validations', 
                          'drift', 'bias_analysis', 'eda', 'reports']:
                local_subdir = local_ge / subdir
                if local_subdir.exists():
                    gcs_path = f"vision/ge_outputs/{subdir}/{partition}"
                    count = gcs.upload_directory(str(local_subdir), gcs_path, max_workers=10)
                    uploaded_count += count
                    logger.info(f"  ✓ {subdir}: {count} files")
                    task_logger.info(f"{subdir}: {count} files uploaded")
        
        # Upload mitigated metadata
        local_mitigated = Path('/opt/airflow/DataPipeline/data/synthetic_metadata_mitigated')
        if local_mitigated.exists():
            logger.info("Uploading bias-mitigated metadata...")
            task_logger.info("Uploading mitigated metadata")
            mitigated_gcs = f"vision/metadata_mitigated/{partition}"
            count = gcs.upload_directory(str(local_mitigated), mitigated_gcs, max_workers=10)
            uploaded_count += count
            logger.info(f"  ✓ Uploaded {count} mitigated files")
            task_logger.info(f"Mitigated metadata: {count} files")
        
        # Upload MLflow
        local_mlflow = Path('/tmp/mlflow')
        if local_mlflow.exists():
            logger.info("Uploading MLflow artifacts...")
            task_logger.info("Uploading MLflow artifacts")
            mlflow_gcs = f"vision/mlflow/{partition}"
            count = gcs.upload_directory(str(local_mlflow), mlflow_gcs, max_workers=10)
            uploaded_count += count
            logger.info(f"  ✓ Uploaded {count} MLflow files")
            task_logger.info(f"MLflow: {count} files")
        
        # Create completion marker
        gcs.create_marker(marker, f"Validation completed: {uploaded_count} files uploaded")
        task_logger.info(f"Created validation marker: {uploaded_count} files")
        
        context['ti'].xcom_push(key='validation_complete', value=True)
        context['ti'].xcom_push(key='uploaded_files', value=uploaded_count)
        context['ti'].xcom_push(key='partition', value=partition)
        
        logger.info("="*80)
        logger.info(f"✓ Validation complete ({uploaded_count} files uploaded to {partition})")
        logger.info("="*80)
        task_logger.info(f"Task complete: {uploaded_count} files")
        return "Success"
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        raise


# ALERT CHECKING TASKS

def check_validation_results(**context):
    """Check validation JSON files for anomalies."""
    partition = get_partition(context, task_ids='validate_and_upload_gcs')
    task_logger = setup_task_logging('check_validation', partition)
    
    logger.info(f"Checking validation results for partition: {partition}")
    task_logger.info(f"Checking validation for partition: {partition}")
    
    result = alert_utils.check_validation_results()
    
    # Push to XCom
    context['ti'].xcom_push(key='anomalies', value=result.get('anomalies', []))
    context['ti'].xcom_push(key='total_anomalies', value=result.get('total_anomalies', 0))
    context['ti'].xcom_push(key='alert_needed', value=result.get('alert_needed', False))
    
    task_logger.info(f"Anomalies found: {result.get('total_anomalies', 0)}")
    return result


def check_drift_results(**context):
    """Check drift JSON files."""
    partition = get_partition(context, task_ids='validate_and_upload_gcs')
    task_logger = setup_task_logging('check_drift', partition)
    
    logger.info(f"Checking drift results for partition: {partition}")
    task_logger.info(f"Checking drift for partition: {partition}")
    
    result = alert_utils.check_drift_results()
    
    # Push to XCom
    context['ti'].xcom_push(key='drift_details', value=result.get('drift_details', []))
    context['ti'].xcom_push(key='total_drifted_features', value=result.get('total_drifted_features', 0))
    context['ti'].xcom_push(key='drift_detected', value=result.get('drift_detected', False))
    
    task_logger.info(f"Drift detected: {result.get('drift_detected', False)}")
    return result


def check_bias_results(**context):
    """Check bias analysis JSON files."""
    partition = get_partition(context, task_ids='validate_and_upload_gcs')
    task_logger = setup_task_logging('check_bias', partition)
    
    logger.info("="*80)
    logger.info("Checking bias analysis results...")
    logger.info("="*80)
    task_logger.info(f"Checking bias for partition: {partition}")
    
    bias_base = '/opt/airflow/DataPipeline/data/ge_outputs/bias_analysis'
    
    bias_files = []
    if os.path.exists(bias_base):
        for root, dirs, files in os.walk(bias_base):
            for file in files:
                if file.endswith('_bias_analysis.json'):
                    bias_files.append(os.path.join(root, file))
    
    if not bias_files:
        logger.info("No bias analysis files found.")
        task_logger.info("No bias files found")
        return {'bias_detected': False, 'total_biases': 0, 'bias_details': []}
    
    bias_found = []
    total_biases = 0
    
    for bias_file in bias_files:
        try:
            with open(bias_file, 'r') as f:
                results = json.load(f)
            
            dataset_name = results.get('dataset_name', 'Unknown')
            has_bias = results.get('bias_detected', False)
            num_biases = results.get('num_significant_biases', 0)
            significant_biases = results.get('significant_biases', [])
            
            logger.info(f"\nDataset: {dataset_name}")
            logger.info(f"  Bias Detected: {has_bias}")
            logger.info(f"  Significant Biases: {num_biases}")
            
            task_logger.info(f"{dataset_name}: bias={has_bias}, count={num_biases}")
            
            if has_bias and num_biases > 0:
                bias_found.append({
                    'dataset': dataset_name,
                    'num_biases': num_biases,
                    'details': significant_biases[:5]  # First 5
                })
                total_biases += num_biases
        
        except Exception as e:
            logger.error(f"Error reading bias file {bias_file}: {e}")
            task_logger.error(f"Error reading {bias_file}: {e}")
    
    if bias_found:
        logger.warning("="*80)
        logger.warning(f"⚠️  BIAS ALERT: {total_biases} biases detected!")
        logger.warning("="*80)
        task_logger.warning(f"BIAS DETECTED: {total_biases} total biases")
        
        context['ti'].xcom_push(key='bias_details', value=bias_found)
        context['ti'].xcom_push(key='total_biases', value=total_biases)
        
        return {'bias_detected': True, 'total_biases': total_biases, 'bias_details': bias_found}
    else:
        logger.info("="*80)
        logger.info("✓ No significant bias detected")
        logger.info("="*80)
        task_logger.info("No significant bias detected")
        return {'bias_detected': False, 'total_biases': 0, 'bias_details': []}


def should_send_alert(**context):
    """Gate function - sends email if anomalies, drift, OR bias detected."""
    ti = context['ti']
    
    anomalies = ti.xcom_pull(task_ids='check_validation_results', key='anomalies')
    drift_details = ti.xcom_pull(task_ids='check_drift_results', key='drift_details')
    bias_details = ti.xcom_pull(task_ids='check_bias_results', key='bias_details')
    
    has_anomalies = anomalies and len(anomalies) > 0
    has_drift = drift_details and len(drift_details) > 0
    has_bias = bias_details and len(bias_details) > 0
    
    if has_anomalies or has_drift or has_bias:
        logger.warning("="*80)
        logger.warning("🚨 ALERT NEEDED - Issues detected")
        if has_anomalies:
            logger.warning(f"  - Anomalies: {len(anomalies)} datasets")
        if has_drift:
            logger.warning(f"  - Drift: {len(drift_details)} datasets")
        if has_bias:
            logger.warning(f"  - Bias: {len(bias_details)} datasets")
        logger.warning("="*80)
        return True
    else:
        logger.info("="*80)
        logger.info("✓ No alerts needed - All checks passed")
        logger.info("="*80)
        return False


def generate_alert_email(**context):
    """Generate email content for alerts."""
    ti = context['ti']
    
    anomalies = ti.xcom_pull(task_ids='check_validation_results', key='anomalies') or []
    total_anomalies = ti.xcom_pull(task_ids='check_validation_results', key='total_anomalies') or 0
    drift_details = ti.xcom_pull(task_ids='check_drift_results', key='drift_details') or []
    total_drifted = ti.xcom_pull(task_ids='check_drift_results', key='total_drifted_features') or 0
    bias_details = ti.xcom_pull(task_ids='check_bias_results', key='bias_details') or []
    total_biases = ti.xcom_pull(task_ids='check_bias_results', key='total_biases') or 0
    
    dag_run_id = context['dag_run'].run_id
    execution_date = str(context['execution_date'])
    partition = get_partition(context, task_ids='validate_and_upload_gcs')
    
    plain_text, html_body = alert_utils.generate_alert_email_content(
        anomalies=anomalies,
        total_anomalies=total_anomalies,
        drift_details=drift_details,
        total_drifted=total_drifted,
        bias_details=bias_details,
        total_biases=total_biases,
        dag_run_id=dag_run_id,
        execution_date=execution_date,
        partition=partition
    )
    
    # Push to XCom for EmailOperator
    ti.xcom_push(key='email_body_plain', value=plain_text)
    ti.xcom_push(key='email_body_html', value=html_body)
    
    logger.info("\n" + "="*80)
    logger.info("EMAIL CONTENT GENERATED")
    logger.info("="*80)
    logger.info(plain_text)
    
    return plain_text


def cleanup_temp_data(**context):
    """Cleanup local temp files."""
    partition = get_partition(context, task_ids='validate_and_upload_gcs')
    task_logger = setup_task_logging('cleanup', partition)
    
    logger.info("="*80)
    logger.info("TASK: Cleanup - Started")
    logger.info("="*80)
    task_logger.info("Starting cleanup")
    
    try:
        temp_paths = [
            Path('/opt/airflow/DataPipeline/data'),
            Path('/opt/airflow/DataPipeline/temp'),
        ]
        
        for path in temp_paths:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"✓ Cleaned: {path}")
                task_logger.info(f"Cleaned: {path}")
        
        Path('/opt/airflow/DataPipeline/data').mkdir(exist_ok=True)
        Path('/opt/airflow/DataPipeline/temp').mkdir(exist_ok=True)
        
        logger.info("="*80)
        logger.info("✓ Cleanup complete")
        logger.info("="*80)
        task_logger.info("Cleanup complete")
        return "Success"
        
    except Exception as e:
        logger.warning(f"Cleanup failed (non-critical): {e}")
        task_logger.warning(f"Cleanup failed: {e}")
        return "Partial"

# DAG DEFINITION

with DAG(
    dag_id='medscan_vision_pipeline_gcs',
    default_args=default_args,
    description='MedScan AI Vision Pipeline with GCS Storage and Comprehensive Alerts',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['vision', 'medical-imaging', 'gcs', 'bias-detection', 'alerts'],
    max_active_runs=1,
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    # Main pipeline tasks
    download = PythonOperator(
        task_id='download_kaggle_to_gcs',
        python_callable=download_kaggle_to_gcs,
        provide_context=True,
    )
    
    preprocess = PythonOperator(
        task_id='preprocess_images_gcs',
        python_callable=preprocess_images_gcs,
        provide_context=True,
    )
    
    metadata = PythonOperator(
        task_id='generate_metadata_gcs',
        python_callable=generate_metadata_gcs,
        provide_context=True,
    )
    
    validate = PythonOperator(
        task_id='validate_and_upload_gcs',
        python_callable=validate_and_upload_gcs,
        provide_context=True,
    )
    
    # Alert checking tasks
    check_validation = PythonOperator(
        task_id='check_validation_results',
        python_callable=check_validation_results,
        provide_context=True,
    )
    
    check_drift = PythonOperator(
        task_id='check_drift_results',
        python_callable=check_drift_results,
        provide_context=True,
    )
    
    check_bias = PythonOperator(
        task_id='check_bias_results',
        python_callable=check_bias_results,
        provide_context=True,
    )
    
    # Gate for conditional email
    check_if_alert_needed = ShortCircuitOperator(
        task_id='check_if_alert_needed',
        python_callable=should_send_alert,
        provide_context=True,
    )
    
    generate_email = PythonOperator(
        task_id='generate_alert_email',
        python_callable=generate_alert_email,
        provide_context=True,
    )
    
    # Email operator (only runs if alerts needed)
    send_alert_email = EmailOperator(
        task_id='send_alert_email',
        to=ALERT_EMAILS,
        subject='🚨 MedScan AI Pipeline Alert - {{ execution_date.strftime("%Y-%m-%d %H:%M") }}',
        html_content="{{ task_instance.xcom_pull(task_ids='generate_alert_email', key='email_body_html') }}",
    )
    
    cleanup = PythonOperator(
        task_id='cleanup_temp',
        python_callable=cleanup_temp_data,
        provide_context=True,
    )
    
    complete = EmptyOperator(task_id='complete')
    
    # Pipeline flow with bias check added
    start >> download >> preprocess >> metadata >> validate
    validate >> check_validation >> check_drift >> check_bias >> check_if_alert_needed
    check_if_alert_needed >> generate_email >> send_alert_email >> cleanup >> complete