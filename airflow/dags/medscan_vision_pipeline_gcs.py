"""
MedScan AI Vision Pipeline DAG with DVC-Primary Storage
Uses DVC for data versioning, GCS only for reports/logs
"""
from datetime import timedelta
from pathlib import Path
import sys
import subprocess
import json
import logging
import shutil
import os
import time

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.email import EmailOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path("/opt/airflow")
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "config"))

from DataPipeline.config import gcp_config
from DataPipeline.scripts.common.gcs_manager import GCSManager
from DataPipeline.scripts.common.dvc_helper import DVCManager
from DataPipeline.scripts.data_preprocessing import alert_utils

logger = logging.getLogger(__name__)

ALERT_EMAILS = gcp_config.ALERT_CONFIG['email_recipients']

default_args = {
    'owner': 'vision-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ALERT_EMAILS,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def setup_task_logging(task_name: str, partition: str) -> logging.Logger:
    """Setup logging for task."""
    log_dir = gcp_config.LOCAL_PATHS['logs']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{task_name}_{partition.replace('/', '_')}.log"
    
    task_logger = logging.getLogger(f"medscan.{task_name}")
    task_logger.setLevel(logging.INFO)
    
    if not task_logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        task_logger.addHandler(fh)
    
    return task_logger


def get_partition(context, task_ids=None):
    """Get partition with fallback logic."""
    ti = context['ti']
    
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
    
    execution_date = context['execution_date']
    partition = execution_date.strftime('%Y/%m/%d')
    return partition


def download_kaggle_data(**context):
    """Download from Kaggle."""
    execution_date = context['execution_date']
    partition = execution_date.strftime('%Y/%m/%d')
    
    task_logger = setup_task_logging('download_kaggle', partition)
    task_logger.info(f"Starting download for partition: {partition}")
    
    try:
        # Check if data already exists locally
        local_raw = Path('/opt/airflow/DataPipeline/data/raw')
        tb_path = local_raw / f'tb/{partition}'
        lc_path = local_raw / f'lung_cancer_ct_scan/{partition}'
        
        # Count existing files
        tb_count = 0
        lc_count = 0
        
        if tb_path.exists():
            # Count all files recursively
            for item in tb_path.rglob('*'):
                if item.is_file() and not item.name.startswith('.'):
                    tb_count += 1
        
        if lc_path.exists():
            for item in lc_path.rglob('*'):
                if item.is_file() and not item.name.startswith('.'):
                    lc_count += 1
        
        if tb_count > 100 and lc_count > 100:
            logger.info(f"Data exists for {partition} - skipping download")
            task_logger.info(f"Skipped - data exists (TB: {tb_count}, LC: {lc_count})")
            context['ti'].xcom_push(key='partition', value=partition)
            context['ti'].xcom_push(key='skipped', value=True)
            context['ti'].xcom_push(key='tb_files', value=tb_count)
            context['ti'].xcom_push(key='lc_files', value=lc_count)
            return "Success - Skipped"
        
        # Download from Kaggle
        task_logger.info("Starting Kaggle download")
        
        cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_acquisition/fetch_data.py',
            '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        
        if result.returncode != 0:
            error_msg = f"Kaggle download failed: {result.stderr}"
            task_logger.error(error_msg)
            raise Exception(error_msg)
        
        # Log stdout for debugging
        task_logger.info("Kaggle output:")
        for line in result.stdout.split('\n'):
            if line.strip():
                task_logger.info(f"  {line}")
        
        # Count downloaded files
        tb_count = 0
        lc_count = 0
        
        if local_raw.exists():
            # Count TB files
            tb_base = local_raw / "tb"
            if tb_base.exists():
                for item in tb_base.rglob('*'):
                    if item.is_file() and not item.name.startswith('.'):
                        tb_count += 1
            
            # Count LC files  
            lc_base = local_raw / "lung_cancer_ct_scan"
            if lc_base.exists():
                for item in lc_base.rglob('*'):
                    if item.is_file() and not item.name.startswith('.'):
                        lc_count += 1
        
        task_logger.info(f"Downloaded: TB={tb_count}, LC={lc_count}")
        
        # Verify both datasets downloaded
        if tb_count == 0:
            logger.warning("TB dataset appears to be empty!")
            task_logger.warning("TB dataset not downloaded or empty")
            
            # List what we actually have
            if (local_raw / "tb").exists():
                task_logger.info("TB directory structure:")
                for item in (local_raw / "tb").rglob('*')[:20]:  # First 20 items
                    task_logger.info(f"  {item.relative_to(local_raw)}")
        
        if lc_count == 0:
            logger.warning("LC dataset appears to be empty!")
            task_logger.warning("LC dataset not downloaded or empty")
        
        context['ti'].xcom_push(key='partition', value=partition)
        context['ti'].xcom_push(key='skipped', value=False)
        context['ti'].xcom_push(key='tb_files', value=tb_count)
        context['ti'].xcom_push(key='lc_files', value=lc_count)
        
        logger.info(f"Download complete (TB: {tb_count}, LC: {lc_count})")
        task_logger.info(f"Task complete: TB={tb_count}, LC={lc_count}")
        return "Success"
        
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        raise


def preprocess_images(**context):
    """Preprocess images (no DVC yet)."""
    partition = get_partition(context, task_ids='download_kaggle_data')
    task_logger = setup_task_logging('preprocess_images', partition)
    task_logger.info(f"Starting preprocessing for partition: {partition}")
    
    try:
        # Check if already preprocessed
        local_preprocessed = Path('/opt/airflow/DataPipeline/data/preprocessed')
        tb_path = local_preprocessed / f'tb/{partition}'
        lc_path = local_preprocessed / f'lung_cancer_ct_scan/{partition}'
        
        if tb_path.exists() and lc_path.exists():
            tb_count = sum(1 for _ in tb_path.rglob('*.jpg'))
            lc_count = sum(1 for _ in lc_path.rglob('*.jpg'))
            if tb_count > 0 and lc_count > 0:
                logger.info(f"Preprocessed data exists for {partition} - skipping")
                task_logger.info("Skipped - data exists")
                context['ti'].xcom_push(key='preprocessing_complete', value=True)
                context['ti'].xcom_push(key='skipped', value=True)
                context['ti'].xcom_push(key='partition', value=partition)
                return "Success - Skipped"
        
        # Run TB preprocessing
        task_logger.info("Starting TB preprocessing")
        tb_cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_preprocessing/process_tb.py',
            '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
        ]
        result = subprocess.run(tb_cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        
        # Log subprocess output for debugging
        if result.stdout:
            task_logger.info(f"TB preprocessing stdout: {result.stdout[-1000:]}")  # Last 1000 chars
        if result.stderr:
            task_logger.warning(f"TB preprocessing stderr: {result.stderr[-1000:]}")
        
        if result.returncode != 0:
            task_logger.error(f"TB preprocessing failed with return code {result.returncode}")
            raise Exception(f"TB preprocessing failed: {result.stderr}")
        task_logger.info("TB preprocessing complete")
        
        # Run Lung Cancer preprocessing
        task_logger.info("Starting Lung Cancer preprocessing")
        lc_cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_preprocessing/process_lungcancer.py',
            '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
        ]
        result = subprocess.run(lc_cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        
        # Log subprocess output for debugging
        if result.stdout:
            task_logger.info(f"Lung Cancer preprocessing stdout: {result.stdout[-1000:]}")
        if result.stderr:
            task_logger.warning(f"Lung Cancer preprocessing stderr: {result.stderr[-1000:]}")
        
        if result.returncode != 0:
            task_logger.error(f"Lung Cancer preprocessing failed with return code {result.returncode}")
            raise Exception(f"Lung cancer preprocessing failed: {result.stderr}")
        task_logger.info("Lung Cancer preprocessing complete")
        
        # Count processed files
        tb_count = sum(1 for _ in local_preprocessed.glob('tb/**/*.jpg')) if local_preprocessed.exists() else 0
        lc_count = sum(1 for _ in local_preprocessed.glob('lung_cancer*/**/*.jpg')) if local_preprocessed.exists() else 0
        
        context['ti'].xcom_push(key='preprocessing_complete', value=True)
        context['ti'].xcom_push(key='tb_processed', value=tb_count)
        context['ti'].xcom_push(key='lc_processed', value=lc_count)
        context['ti'].xcom_push(key='partition', value=partition)
        
        logger.info(f"Preprocessing complete (TB: {tb_count}, LC: {lc_count})")
        task_logger.info(f"Task complete: TB={tb_count}, LC={lc_count}")
        return "Success"
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        raise


def generate_metadata(**context):
    """Generate metadata (no DVC yet)."""
    partition = get_partition(context, task_ids=['preprocess_images', 'download_kaggle_data'])
    task_logger = setup_task_logging('generate_metadata', partition)
    task_logger.info(f"Starting metadata generation for partition: {partition}")
    
    try:
        # Check if already generated
        local_metadata = Path('/opt/airflow/DataPipeline/data/synthetic_metadata')
        tb_csv = local_metadata / f'tb/{partition}/tb_patients.csv'
        lc_csv = local_metadata / f'lung_cancer/{partition}/lung_cancer_ct_scan_patients.csv'
        
        if tb_csv.exists() and lc_csv.exists():
            logger.info(f"Metadata exists for {partition} - skipping")
            task_logger.info("Skipped - data exists")
            context['ti'].xcom_push(key='partition', value=partition)
            return "Success - Skipped"
        
        # Generate metadata
        task_logger.info("Starting metadata generation")
        cmd = [
            'python',
            '/opt/airflow/DataPipeline/scripts/data_preprocessing/baseline_synthetic_data_generator.py',
            '--config', '/opt/airflow/DataPipeline/config/synthetic_data.yml'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
        if result.returncode != 0:
            raise Exception(f"Metadata generation failed: {result.stderr}")
        task_logger.info("Metadata generation complete")
        
        context['ti'].xcom_push(key='metadata_uploaded', value=True)
        context['ti'].xcom_push(key='partition', value=partition)
        
        task_logger.info("Task complete")
        return "Success"
        
    except Exception as e:
        logger.error(f"Metadata generation failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        raise


def validate_and_upload_reports(**context):
    """Run validation and upload ONLY reports to GCS."""
    partition = get_partition(context, task_ids=['generate_metadata', 'preprocess_images', 
                                                  'download_kaggle_data'])
    task_logger = setup_task_logging('validate', partition)
    task_logger.info(f"Starting validation for partition: {partition}")
    
    try:
        gcs = GCSManager.from_config()
        
        # Check if already validated
        # Marker includes partition in filename to avoid creating unnecessary directory structure
        # Use filename format: .complete_{YYYY}_{MM}_{DD} instead of creating {YYYY}/{MM}/{DD}/.complete
        partition_flat = partition.replace('/', '_')  # Convert 2025/11/05 to 2025_11_05
        marker = gcp_config.get_gcs_path('vision', 'ge_outputs/validations', partition=None).rstrip('/') + f"/.complete_{partition_flat}"
        validation_skipped = gcs.blob_exists(marker)
        
        if validation_skipped:
            logger.info(f"Validation complete for {partition} - skipping validation step")
            task_logger.info("Skipped - marker exists")
        else:
            # Run validation
            task_logger.info("Starting validation")
            cmd = [
                'python',
                '/opt/airflow/DataPipeline/scripts/data_preprocessing/schema_statistics.py',
                '--config', '/opt/airflow/DataPipeline/config/metadata.yml'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/opt/airflow/DataPipeline')
            if result.returncode != 0:
                raise Exception(f"Validation failed: {result.stderr}")
            task_logger.info("Validation complete")
        
        # Upload ONLY ge_outputs (reports/logs) to GCS (even if validation was skipped)
        local_ge = Path('/opt/airflow/DataPipeline/data/ge_outputs')
        uploaded_count = 0
        
        if local_ge.exists():
            task_logger.info("Uploading validation reports to GCS")
            
            for subdir in ['baseline', 'new_data', 'schemas', 'validations', 
                          'drift', 'bias_analysis', 'eda', 'reports']:
                local_subdir = local_ge / subdir
                if local_subdir.exists():
                    # Don't add partition here - partition is already in local directory structure
                    # (e.g., baseline/lung_cancer/2025/11/04/ already contains partition)
                    gcs_path = gcp_config.get_gcs_path('vision', f'ge_outputs/{subdir}', partition=None).rstrip('/')
                    count = gcs.upload_directory(str(local_subdir), gcs_path, max_workers=10)
                    uploaded_count += count
        
        # Upload MLflow artifacts (even if validation was skipped - artifacts may exist from previous run)
        # MLflow stores artifacts in structure: artifacts/{run_id}/artifacts/{category}/files
        local_mlflow_base = Path('/opt/airflow/DataPipeline/data/mlflow_store/metadata/artifacts')
        if local_mlflow_base.exists():
            mlflow_gcs_base = gcp_config.get_gcs_path('vision', 'mlflow', partition=partition).rstrip('/')
            
            # Find all run directories
            run_dirs = [d for d in local_mlflow_base.iterdir() if d.is_dir()]
            
            if run_dirs:
                task_logger.info(f"Found {len(run_dirs)} MLflow runs to upload")
                for run_dir in run_dirs:
                    # Upload from artifacts/{run_id}/artifacts/
                    artifacts_dir = run_dir / 'artifacts'
                    if artifacts_dir.exists() and any(artifacts_dir.iterdir()):
                        run_gcs_path = f"{mlflow_gcs_base}/{run_dir.name}/artifacts"
                        count = gcs.upload_directory(str(artifacts_dir), run_gcs_path, max_workers=10)
                        uploaded_count += count
                        task_logger.info(f"Uploaded {count} artifacts from run {run_dir.name}")
            else:
                task_logger.info("No MLflow run directories found")
        else:
            task_logger.info("MLflow artifacts directory does not exist")
        
        # Create completion marker
        gcs.create_marker(marker, f"Validation completed: {uploaded_count} files uploaded")
        
        context['ti'].xcom_push(key='validation_complete', value=True)
        context['ti'].xcom_push(key='uploaded_files', value=uploaded_count)
        context['ti'].xcom_push(key='partition', value=partition)
        
        logger.info(f"Validation complete ({uploaded_count} report files uploaded to GCS)")
        task_logger.info(f"Task complete: {uploaded_count} report files")
        return "Success"
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        raise


def track_all_with_dvc(**context):
    """Track all vision data with DVC using batch operations."""
    partition = get_partition(context, task_ids='validate_and_upload_reports')
    task_logger = setup_task_logging('dvc_tracking', partition)
    task_logger.info(f"Starting DVC tracking for partition: {partition}")
    
    try:
        dvc = DVCManager()
        
        # Collect all paths to track
        paths_to_track = []
        
        if Path("/opt/airflow/DataPipeline/data/raw").exists():
            paths_to_track.append("data/raw")
        
        if Path("/opt/airflow/DataPipeline/data/preprocessed").exists():
            paths_to_track.append("data/preprocessed")
        
        if Path("/opt/airflow/DataPipeline/data/synthetic_metadata").exists():
            paths_to_track.append("data/synthetic_metadata")
        
        if Path("/opt/airflow/DataPipeline/data/synthetic_metadata_mitigated").exists():
            paths_to_track.append("data/synthetic_metadata_mitigated")
        
        if not paths_to_track:
            logger.warning("No paths to track with DVC")
            return "No data to track"
        
        logger.info(f"Tracking {len(paths_to_track)} paths with DVC (batch mode)")
        
        # Add all paths in one batch operation
        results = dvc.add_batch(paths_to_track)
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Successfully tracked {success_count}/{len(paths_to_track)} paths")
        
        # Push to DVC remote
        if success_count > 0:
            logger.info("Pushing to DVC remote")
            if dvc.push(remote='vision', jobs=4):
                logger.info("Successfully pushed to DVC remote")
            else:
                logger.warning("DVC push had issues (non-critical)")
        
        # Commit DVC metadata
        commit_msg = f"Vision pipeline {partition.replace('/', '-')}"
        dvc.commit_dvc_files(commit_msg)
        
        logger.info("DVC tracking complete")
        task_logger.info("Task complete")
        return "Success"
        
    except Exception as e:
        logger.error(f"DVC tracking failed: {e}", exc_info=True)
        task_logger.error(f"Task failed: {e}", exc_info=True)
        return "Partial"


def check_validation_results(**context):
    """Check validation JSON files for anomalies."""
    partition = get_partition(context, task_ids='validate_and_upload_reports')
    task_logger = setup_task_logging('check_validation', partition)
    
    result = alert_utils.check_validation_results()
    
    context['ti'].xcom_push(key='anomalies', value=result.get('anomalies', []))
    context['ti'].xcom_push(key='total_anomalies', value=result.get('total_anomalies', 0))
    context['ti'].xcom_push(key='alert_needed', value=result.get('alert_needed', False))
    
    task_logger.info(f"Anomalies found: {result.get('total_anomalies', 0)}")
    return result


def check_drift_results(**context):
    """Check drift JSON files."""
    partition = get_partition(context, task_ids='validate_and_upload_reports')
    task_logger = setup_task_logging('check_drift', partition)
    
    result = alert_utils.check_drift_results()
    
    context['ti'].xcom_push(key='drift_details', value=result.get('drift_details', []))
    context['ti'].xcom_push(key='total_drifted_features', value=result.get('total_drifted_features', 0))
    context['ti'].xcom_push(key='drift_detected', value=result.get('drift_detected', False))
    
    task_logger.info(f"Drift detected: {result.get('drift_detected', False)}")
    return result


def check_bias_results(**context):
    """Check bias analysis JSON files."""
    partition = get_partition(context, task_ids='validate_and_upload_reports')
    task_logger = setup_task_logging('check_bias', partition)
    
    bias_base = '/opt/airflow/DataPipeline/data/ge_outputs/bias_analysis'
    
    bias_files = []
    if os.path.exists(bias_base):
        for root, dirs, files in os.walk(bias_base):
            for file in files:
                if file.endswith('_bias_analysis.json'):
                    bias_files.append(os.path.join(root, file))
    
    if not bias_files:
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
            num_biases = len(results.get('significant_biases', []))
            
            task_logger.info(f"{dataset_name}: bias={has_bias}, count={num_biases}")
            
            if has_bias and num_biases > 0:
                bias_found.append({
                    'dataset': dataset_name,
                    'num_biases': num_biases,
                    'details': results.get('significant_biases', [])[:5]
                })
                total_biases += num_biases
        
        except Exception as e:
            task_logger.error(f"Error reading {bias_file}: {e}")
    
    if bias_found:
        task_logger.warning(f"BIAS DETECTED: {total_biases} total biases")
        context['ti'].xcom_push(key='bias_details', value=bias_found)
        context['ti'].xcom_push(key='total_biases', value=total_biases)
        return {'bias_detected': True, 'total_biases': total_biases, 'bias_details': bias_found}
    else:
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
    
    return has_anomalies or has_drift or has_bias


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
    partition = get_partition(context, task_ids='validate_and_upload_reports')
    
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
    
    ti.xcom_push(key='email_body_plain', value=plain_text)
    ti.xcom_push(key='email_body_html', value=html_body)
    
    return plain_text


def cleanup_temp_data(**context):
    """Cleanup local temp files."""
    partition = get_partition(context, task_ids='validate_and_upload_reports')
    task_logger = setup_task_logging('cleanup', partition)
    task_logger.info("Starting cleanup")
    
    try:
        # Only clean temp directories, keep DVC-tracked data
        temp_paths = [
            Path('/opt/airflow/DataPipeline/temp'),
        ]
        
        for path in temp_paths:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                task_logger.info(f"Cleaned: {path}")
        
        Path('/opt/airflow/DataPipeline/temp').mkdir(exist_ok=True)
        
        task_logger.info("Cleanup complete")
        return "Success"
        
    except Exception as e:
        logger.warning(f"Cleanup failed (non-critical): {e}")
        return "Partial"


# DAG DEFINITION
with DAG(
    dag_id='medscan_vision_pipeline_dvc',
    default_args=default_args,
    description='MedScan AI Vision Pipeline with DVC-primary storage',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['vision', 'medical-imaging', 'dvc'],
    max_active_runs=1,
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    # Main pipeline tasks
    download = PythonOperator(
        task_id='download_kaggle_data',
        python_callable=download_kaggle_data,
        provide_context=True,
    )
    
    preprocess = PythonOperator(
        task_id='preprocess_images',
        python_callable=preprocess_images,
        provide_context=True,
    )
    
    metadata = PythonOperator(
        task_id='generate_metadata',
        python_callable=generate_metadata,
        provide_context=True,
    )
    
    validate = PythonOperator(
        task_id='validate_and_upload_reports',
        python_callable=validate_and_upload_reports,
        provide_context=True,
    )
    
    # Single DVC tracking task at the end
    dvc_track = PythonOperator(
        task_id='track_all_with_dvc',
        python_callable=track_all_with_dvc,
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
    
    send_alert_email = EmailOperator(
        task_id='send_alert_email',
        to=ALERT_EMAILS,
        subject='MedScan AI Pipeline Alert - {{ execution_date.strftime("%Y-%m-%d %H:%M") }}',
        html_content="{{ task_instance.xcom_pull(task_ids='generate_alert_email', key='email_body_html') }}",
    )
    
    cleanup = PythonOperator(
        task_id='cleanup_temp',
        python_callable=cleanup_temp_data,
        provide_context=True,
    )
    
    complete = EmptyOperator(task_id='complete')
    
    # Pipeline flow
    start >> download >> preprocess >> metadata >> validate >> dvc_track
    dvc_track >> check_validation >> check_drift >> check_bias >> check_if_alert_needed
    check_if_alert_needed >> generate_email >> send_alert_email >> cleanup >> complete