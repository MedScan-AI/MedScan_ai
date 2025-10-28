from airflow import DAG
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
import sys
import subprocess
import json
from glob import glob

sys.path.insert(0, '/opt/airflow/DataPipeline')

# Get alert email recipients from environment variable
ALERT_EMAILS = os.environ.get('ALERT_EMAIL_RECIPIENTS', 'default@example.com').split(',')

default_args = {
    'owner': 'medscan-ai',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,   
    'email_on_retry': False,
    'email': ALERT_EMAILS,       
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ALERT FUNCTIONS

def check_validation_results(**context):
    """Check validation results and trigger alert if anomalies found."""
    print("=" * 80)
    print("Checking validation results for anomalies...")
    print("=" * 80)
    
    validation_base = '/opt/airflow/DataPipeline/data/ge_outputs/validations'
    
    validation_files = []
    if os.path.exists(validation_base):
        for root, dirs, files in os.walk(validation_base):
            for file in files:
                if file.endswith('_validation.json'):
                    validation_files.append(os.path.join(root, file))
    
    if not validation_files:
        print("No validation files found. Skipping anomaly check.")
        return {'alert_needed': False}
    
    anomalies_found = []
    total_anomalies = 0
    
    for val_file in validation_files:
        try:
            with open(val_file, 'r') as f:
                results = json.load(f)
            
            dataset_name = results.get('dataset_name', 'Unknown')
            is_valid = results.get('is_valid', True)
            num_anomalies = results.get('num_anomalies', 0)
            anomaly_list = results.get('anomalies', [])
            
            print(f"\nDataset: {dataset_name}")
            print(f"  Valid: {is_valid}")
            print(f"  Anomalies: {num_anomalies}")
            
            if not is_valid and num_anomalies > 0:
                anomalies_found.append({
                    'dataset': dataset_name,
                    'num_anomalies': num_anomalies,
                    'details': anomaly_list[:3]
                })
                total_anomalies += num_anomalies
        
        except Exception as e:
            print(f"Error reading validation file {val_file}: {e}")
    
    if anomalies_found:
        print("=" * 80)
        print(f"ALERT: {total_anomalies} total anomalies found!")
        print("=" * 80)
        
        context['task_instance'].xcom_push(key='anomalies', value=anomalies_found)
        context['task_instance'].xcom_push(key='total_anomalies', value=total_anomalies)
        
        return {'alert_needed': True, 'total_anomalies': total_anomalies}
    else:
        print("=" * 80)
        print("✓ No anomalies detected")
        print("=" * 80)
        return {'alert_needed': False}


def check_drift_results(**context):
    """Check drift detection results and trigger alert if drift found."""
    print("=" * 80)
    print("Checking drift detection results...")
    print("=" * 80)
    
    drift_base = '/opt/airflow/DataPipeline/data/ge_outputs/drift'
    
    drift_files = []
    if os.path.exists(drift_base):
        for root, dirs, files in os.walk(drift_base):
            for file in files:
                if file.endswith('_drift.json'):
                    drift_files.append(os.path.join(root, file))
    
    if not drift_files:
        print("No drift files found. Skipping drift check.")
        return {'drift_detected': False}
    
    drift_found = []
    total_drifted_features = 0
    
    for drift_file in drift_files:
        try:
            with open(drift_file, 'r') as f:
                results = json.load(f)
            
            dataset_name = results.get('dataset_name', 'Unknown')
            has_drift = results.get('has_drift', False)
            num_drifted = results.get('num_drifted_features', 0)
            drifted_features = results.get('drifted_features', [])
            
            print(f"\nDataset: {dataset_name}")
            print(f"  Drift Detected: {has_drift}")
            print(f"  Drifted Features: {num_drifted}")
            
            if has_drift and num_drifted > 0:
                drift_found.append({
                    'dataset': dataset_name,
                    'num_drifted': num_drifted,
                    'features': [f['feature'] for f in drifted_features[:5]]
                })
                total_drifted_features += num_drifted
        
        except Exception as e:
            print(f"Error reading drift file {drift_file}: {e}")
    
    if drift_found:
        print("=" * 80)
        print(f"DRIFT ALERT: {total_drifted_features} features drifted!")
        print("=" * 80)
        
        context['task_instance'].xcom_push(key='drift_details', value=drift_found)
        context['task_instance'].xcom_push(key='total_drifted_features', value=total_drifted_features)
        
        return {'drift_detected': True, 'total_drifted': total_drifted_features}
    else:
        print("=" * 80)
        print("No drift detected")
        print("=" * 80)
        return {'drift_detected': False}


def should_send_alert(**context):
    """
    Determine if alert email should be sent.
    Returns True only if anomalies or drift detected.
    """
    ti = context['task_instance']
    
    # Check for anomalies
    anomalies = ti.xcom_pull(task_ids='check_validation_results', key='anomalies')
    
    # Check for drift
    drift_details = ti.xcom_pull(task_ids='check_drift_results', key='drift_details')
    
    # Send email only if there are issues
    has_anomalies = anomalies and len(anomalies) > 0
    has_drift = drift_details and len(drift_details) > 0
    
    if has_anomalies or has_drift:
        print("=" * 80)
        print("ALERT NEEDED - Anomalies or drift detected")
        print("=" * 80)
        return True  # Email will be sent
    else:
        print("=" * 80)
        print("No alerts needed - All validations passed")
        print("=" * 80)
        return False  # Email task will be SKIPPED


def generate_alert_email_content(**context):
    """Generate email content based on validation and drift results."""
    ti = context['task_instance']
    
    anomalies = ti.xcom_pull(task_ids='check_validation_results', key='anomalies')
    total_anomalies = ti.xcom_pull(task_ids='check_validation_results', key='total_anomalies')
    drift_details = ti.xcom_pull(task_ids='check_drift_results', key='drift_details')
    total_drifted = ti.xcom_pull(task_ids='check_drift_results', key='total_drifted_features')
    
    email_lines = [
        "MedScan AI Data Pipeline Alert",
        "=" * 60,
        f"Pipeline Run: {context['dag_run'].run_id}",
        f"Execution Date: {context['execution_date']}",
        ""
    ]
    
    if anomalies:
        email_lines.extend([
            "DATA VALIDATION ALERTS",
            "=" * 60,
            f"Total Anomalies Detected: {total_anomalies}",
            ""
        ])
        
        for anom in anomalies:
            email_lines.append(f"Dataset: {anom['dataset']}")
            email_lines.append(f"  Anomalies: {anom['num_anomalies']}")
            email_lines.append("  Details:")
            for detail in anom['details']:
                email_lines.append(f"    - {detail['column']}: {detail['description']}")
            email_lines.append("")
    
    if drift_details:
        email_lines.extend([
            "DATA DRIFT ALERTS",
            "=" * 60,
            f"Total Drifted Features: {total_drifted}",
            ""
        ])
        
        for drift in drift_details:
            email_lines.append(f"Dataset: {drift['dataset']}")
            email_lines.append(f"  Drifted Features: {drift['num_drifted']}")
            email_lines.append(f"  Features: {', '.join(drift['features'])}")
            email_lines.append("")
    
    email_lines.extend([
        "RECOMMENDED ACTIONS",
        "=" * 60,
        "1. Review validation reports in: data/ge_outputs/reports/",
        "2. Check MLflow tracking UI: http://localhost:5000",
        "3. Investigate anomalies before proceeding to model training",
        "4. Consider retraining models if significant drift detected",
        "",
        "View full logs in Airflow UI: http://localhost:8080",
    ])
    
    email_content = "\n".join(email_lines)
    ti.xcom_push(key='email_body', value=email_content)
    
    print("\n" + "=" * 80)
    print("EMAIL CONTENT GENERATED")
    print("=" * 80)
    print(email_content)
    
    return email_content

# TEST FUNCTIONS

def run_data_acquisition_tests(**context):
    """Run tests for data acquisition module"""
    print("=" * 80)
    print("Running data acquisition tests...")
    print("=" * 80)
    
    cmd = [
        'pytest',
        '/opt/airflow/DataPipeline/tests/data_acquisition/fetch_data_test.py',
        '-q',
        '--tb=short',
        '--import-mode=importlib',
        '-p', 'no:cacheprovider'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("=" * 80)
        print("TEST FAILURES")
        print("=" * 80)
        print(result.stderr)
        raise Exception(f"Data acquisition tests failed")
    
    print("=" * 80)
    print("Tests passed!")
    print("=" * 80)
    return "Success"


def run_preprocessing_tests(**context):
    """Run tests for preprocessing modules"""
    print("=" * 80)
    print("Running preprocessing tests...")
    print("=" * 80)
    
    cmd = [
        'pytest',
        '/opt/airflow/DataPipeline/tests/data_preprocessing/preprocess_tb_test.py',
        '/opt/airflow/DataPipeline/tests/data_preprocessing/preprocess_lung_cancer_test.py',
        '-q',
        '--tb=short',
        '--import-mode=importlib',
        '-p', 'no:cacheprovider'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("=" * 80)
        print("TEST FAILURES")
        print("=" * 80)
        print(result.stderr)
        raise Exception(f"Preprocessing tests failed")
    
    print("=" * 80)
    print("Tests passed!")
    print("=" * 80)
    return "Success"


def run_synthetic_data_tests(**context):
    """Run tests for synthetic data generator"""
    print("=" * 80)
    print("Running synthetic data tests...")
    print("=" * 80)
    
    cmd = [
        'pytest',
        '/opt/airflow/DataPipeline/tests/data_preprocessing/baseline_synthetic_data_generator_test.py',
        '-q',
        '--tb=short',
        '--import-mode=importlib',
        '-p', 'no:cacheprovider'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("=" * 80)
        print("TEST FAILURES")
        print("=" * 80)
        print(result.stderr)
        raise Exception(f"Synthetic data tests failed")
    
    print("=" * 80)
    print("Tests passed!")
    print("=" * 80)
    return "Success"


def run_validation_tests(**context):
    """Run tests for schema validation"""
    print("=" * 80)
    print("Running validation tests...")
    print("=" * 80)
    
    cmd = [
        'pytest',
        '/opt/airflow/DataPipeline/tests/data_preprocessing/schema_statistics_test.py',
        '-q',
        '--tb=short',
        '--import-mode=importlib',
        '-p', 'no:cacheprovider'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("=" * 80)
        print("TEST FAILURES")
        print("=" * 80)
        print(result.stderr)
        raise Exception(f"Validation tests failed")
    
    print("=" * 80)
    print("Tests passed!")
    print("=" * 80)
    return "Success"

# PIPELINE TASK FUNCTIONS

def acquire_data(**context):
    """Download data from Kaggle"""
    print("=" * 80)
    print("Starting data acquisition from Kaggle...")
    print("=" * 80)
    
    cmd = [
        'python',
        '/opt/airflow/DataPipeline/scripts/data_acquisition/fetch_data.py',
        '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception(f"Data acquisition failed: {result.stderr}")
    
    print(result.stdout)
    print("=" * 80)
    print("Data acquisition completed successfully!")
    print("=" * 80)
    return "Success"


def preprocess_tb(**context):
    """Process TB images"""
    print("=" * 80)
    print("Starting TB preprocessing...")
    print("=" * 80)
    
    cmd = [
        'python',
        '/opt/airflow/DataPipeline/scripts/data_preprocessing/process_tb.py',
        '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception(f"TB preprocessing failed: {result.stderr}")
    
    print(result.stdout)
    print("=" * 80)
    print("TB preprocessing completed successfully!")
    print("=" * 80)
    return "Success"


def preprocess_lung_cancer(**context):
    """Process lung cancer images"""
    print("=" * 80)
    print("Starting lung cancer preprocessing...")
    print("=" * 80)
    
    cmd = [
        'python',
        '/opt/airflow/DataPipeline/scripts/data_preprocessing/process_lungcancer.py',
        '--config', '/opt/airflow/DataPipeline/config/vision_pipeline.yml'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception(f"Lung cancer preprocessing failed: {result.stderr}")
    
    print(result.stdout)
    print("=" * 80)
    print("Lung cancer preprocessing completed successfully!")
    print("=" * 80)
    return "Success"


def generate_synthetic_metadata(**context):
    """Generate fake patient data"""
    print("=" * 80)
    print("Starting synthetic metadata generation...")
    print("=" * 80)
    
    cmd = [
        'python',
        '/opt/airflow/DataPipeline/scripts/data_preprocessing/baseline_synthetic_data_generator.py',
        '--config', '/opt/airflow/DataPipeline/config/synthetic_data.yml'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception(f"Synthetic data generation failed: {result.stderr}")
    
    print(result.stdout)
    print("=" * 80)
    print("Synthetic metadata generation completed successfully!")
    print("=" * 80)
    return "Success"


def run_data_validation(**context):
    """Validate the data quality"""
    print("=" * 80)
    print("Starting data validation with bias detection...")
    print("=" * 80)
    
    cmd = [
        'python',
        '/opt/airflow/DataPipeline/scripts/data_preprocessing/schema_statistics.py',
        '--config', '/opt/airflow/DataPipeline/config/metadata.yml'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception(f"Data validation failed: {result.stderr}")
    
    print(result.stdout)
    print("=" * 80)
    print("Data validation completed successfully!")
    print("=" * 80)
    return "Success"


# DAG DEFINITION

dag = DAG(
    'medscan_vision_pipeline',
    default_args=default_args,
    description='MedScan AI Vision Data Pipeline with Bias Detection and Email Alerts',
    schedule=None,
    catchup=False,
    tags=['vision', 'medical-imaging', 'mlops', 'bias-detection', 'alerts'],
)

# TASK DEFINITIONS

start_task = EmptyOperator(task_id="start", dag=dag)

# Test Tasks
test_data_acquisition = PythonOperator(
    task_id='test_data_acquisition',
    python_callable=run_data_acquisition_tests,
    dag=dag,
)

download_task = PythonOperator(
    task_id='download_kaggle_data',
    python_callable=acquire_data,
    dag=dag,
)

test_preprocessing = PythonOperator(
    task_id='test_preprocessing',
    python_callable=run_preprocessing_tests,
    dag=dag,
)

tb_task = PythonOperator(
    task_id='process_tb',
    python_callable=preprocess_tb,
    dag=dag,
)

lung_task = PythonOperator(
    task_id='process_lung_cancer',
    python_callable=preprocess_lung_cancer,
    dag=dag,
)

test_synthetic_data = PythonOperator(
    task_id='test_synthetic_data',
    python_callable=run_synthetic_data_tests,
    dag=dag,
)

metadata_task = PythonOperator(
    task_id='generate_patient_metadata',
    python_callable=generate_synthetic_metadata,
    dag=dag,
)

test_validation = PythonOperator(
    task_id='test_validation',
    python_callable=run_validation_tests,
    dag=dag,
)

validation_task = PythonOperator(
    task_id='validate_data',
    python_callable=run_data_validation,
    dag=dag,
)

# Alert Tasks
check_validation = PythonOperator(
    task_id='check_validation_results',
    python_callable=check_validation_results,
    dag=dag,
)

check_drift = PythonOperator(
    task_id='check_drift_results',
    python_callable=check_drift_results,
    dag=dag,
)

# Gate for conditional email sending
check_if_alert_needed = ShortCircuitOperator(
    task_id='check_if_alert_needed',
    python_callable=should_send_alert,
    dag=dag,
)

generate_email = PythonOperator(
    task_id='generate_alert_email',
    python_callable=generate_alert_email_content,
    dag=dag,
)

send_alert_email = EmailOperator(
    task_id='send_alert_email',
    to=ALERT_EMAILS,
    subject='MedScan AI Pipeline Alert - {{ execution_date.strftime("%Y-%m-%d %H:%M") }}',
    html_content="""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; }
            .header { background-color: #ff4444; color: white; padding: 20px; }
            .content { padding: 20px; background-color: #f9f9f9; }
            .footer { padding: 10px; text-align: center; font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MedScan AI Pipeline Alert</h1>
        </div>
        <div class="content">
            <pre>{{ task_instance.xcom_pull(task_ids='generate_alert_email', key='email_body') }}</pre>
        </div>
        <div class="footer">
            <p>MedScan AI Automated Data Pipeline</p>
        </div>
    </body>
    </html>
    """,
    dag=dag,
)

end_task = EmptyOperator(task_id="complete", dag=dag)

# Main pipeline flow
start_task >> test_data_acquisition >> download_task
download_task >> test_preprocessing >> [tb_task, lung_task]
[tb_task, lung_task] >> test_synthetic_data >> metadata_task
metadata_task >> test_validation >> validation_task

# Alert flow with conditional email
validation_task >> check_validation >> check_drift >> check_if_alert_needed
check_if_alert_needed >> generate_email >> send_alert_email >> end_task
