"""
RAG Data Pipeline DAG - FIXED BASELINE Approach
Medical literature data pipeline with TFDV validation.
Runs alongside vision pipeline in shared Airflow instance.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import asyncio
import subprocess
import shutil

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Docker environment paths
PROJECT_ROOT = Path("/opt/airflow")
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "config"))

# Import configurations
try:
    import gcp_config
    from RAG.common_utils import GCSManager, upload_with_versioning
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    raise

# Default DAG arguments
default_args = {
    'owner': 'rag-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def scrape_data(**context):
    """Task 1: Scrape medical literature from configured URLs."""
    from RAG.scraper import main as scraper_main
    
    execution_date = context['execution_date']
    timestamp = execution_date.strftime('%Y%m%d_%H%M%S')
    
    # Get URLs from configuration (single source of truth)
    urls = gcp_config.get_scraping_urls()
    
    # Output file
    output_file = gcp_config.LOCAL_PATHS['raw_data'] / f"batch_{timestamp}.jsonl"
    
    print("=" * 60)
    print("TASK 1: SCRAPING MEDICAL DATA")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"URLs to scrape: {len(urls)}")
    print(f"Output: {output_file.name}")
    print()
    
    # Run existing scraper
    asyncio.run(scraper_main(urls, str(output_file), method='W'))
    
    # Upload incremental batch to GCS
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    gcs_path = f"RAG/raw_data/incremental/batch_{timestamp}.jsonl"
    gcs.upload_file(str(output_file), gcs_path)
    
    # Pass to next task via XCom
    context['ti'].xcom_push(key='batch_file', value=str(output_file))
    context['ti'].xcom_push(key='timestamp', value=timestamp)
    
    print()
    print(f"Scraping complete: {len(urls)} URLs processed")
    print("=" * 60)


def merge_data(**context):
    """Task 2: Merge new batch with previous merged data."""
    from RAG.merge_batches import merge_jsonl_files
    
    ti = context['ti']
    batch_file = ti.xcom_pull(task_ids='scrape_data', key='batch_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    print("=" * 60)
    print("TASK 2: MERGING DATA")
    print("=" * 60)
    print(f"Batch: {Path(batch_file).name}")
    print()
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Download previous merged data (or FIXED baseline if first run)
    local_baseline = gcp_config.LOCAL_PATHS['merged'] / "combined_latest.jsonl"
    
    if gcs.blob_exists("RAG/merged/combined_latest.jsonl"):
        print("  Downloading previous merged data...")
        gcs.download_file("RAG/merged/combined_latest.jsonl", str(local_baseline))
    else:
        print("  First merge: downloading FIXED baseline...")
        if not gcs.download_file("RAG/raw_data/baseline/baseline.jsonl", str(local_baseline)):
            raise Exception("FIXED baseline not found in GCS! Upload baseline.jsonl first.")
    
    # Merge using existing function
    output_file = gcp_config.LOCAL_PATHS['merged'] / f"combined_{timestamp}.jsonl"
    success = merge_jsonl_files(local_baseline, Path(batch_file), output_file)
    
    if not success:
        raise Exception("Merge operation failed")
    
    # Upload versioned to GCS
    version = upload_with_versioning(
        gcs,
        str(output_file),
        "RAG/merged",
        "combined_{version}.jsonl"
    )
    
    # Pass to next task
    ti.xcom_push(key='merged_file', value=str(output_file))
    ti.xcom_push(key='version', value=version)
    
    print()
    print(f"Merge complete: version {version}")
    print("=" * 60)


def validate_data(**context):
    """Task 3: Validate against FIXED baseline using TFDV."""
    import json
    
    ti = context['ti']
    merged_file = ti.xcom_pull(task_ids='merge_data', key='merged_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    print("=" * 60)
    print("TASK 3: TFDV VALIDATION")
    print("=" * 60)
    print("Strategy: FIXED BASELINE (best practice)")
    print()
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Check if FIXED baseline exists in GCS
    baseline_exists = gcs.blob_exists("RAG/raw_data/baseline/baseline.jsonl")
    
    if not baseline_exists:
        # FIRST RUN: No FIXED baseline uploaded yet
        print("FIRST RUN DETECTED")
        print("=" * 60)
        print("  Status: No FIXED baseline exists in GCS")
        print("  Action: Skipping TFDV validation")
        print()
        print("  Why skip?")
        print("    • First run has no reference point for comparison")
        print("    • TFDV requires a baseline to detect drift")
        print("    • Current data will be used as future baseline")
        print()
        print("  What happens next?")
        print("    1. Current merged data gets processed and versioned")
        print("    2. Upload this data as FIXED baseline (if quality is good)")
        print("    3. All future runs will validate against FIXED baseline")
        print("=" * 60)
        print()
        print(" Validation skipped (expected for first run)")
        print("=" * 60)
        
        # Mark as first run
        ti.xcom_push(key='validation_passed', value=True)
        ti.xcom_push(key='is_first_run', value=True)
        ti.xcom_push(key='drift_detected', value=False)
        return
    
    # SUBSEQUENT RUNS: Validate against FIXED baseline
    print(" FIXED BASELINE EXISTS")
    print("  Running TFDV validation against reference baseline...")
    print()
    
    from RAG.analysis.main import DataQualityAnalyzer
    
    # Prepare file paths (validator expects specific names)
    baseline_path = gcp_config.PROJECT_ROOT / "data" / "scraped_baseline.jsonl"
    new_data_path = gcp_config.PROJECT_ROOT / "data" / "scraped_updated.jsonl"
    
    # Download FIXED baseline (never changes automatically)
    print(" Downloading FIXED baseline from GCS...")
    success = gcs.download_file("RAG/raw_data/baseline/baseline.jsonl", str(baseline_path))
    
    if not success:
        raise Exception("Failed to download FIXED baseline from GCS")
    
    # Copy current merged data as "new data"
    shutil.copy(merged_file, new_data_path)
    
    # Count records
    with open(baseline_path, 'r') as f:
        baseline_count = sum(1 for _ in f)
    with open(new_data_path, 'r') as f:
        new_count = sum(1 for _ in f)
    
    print(f"   FIXED Baseline: {baseline_count} records")
    print(f"   Current Data: {new_count} records")
    print(f"   Growth: +{new_count - baseline_count} records ({((new_count/baseline_count - 1) * 100):.1f}%)")
    print()
    
    # Run TFDV validation
    print("  Running TFDV analysis...")
    print("  " + "-" * 56)
    print()
    
    analyzer = DataQualityAnalyzer(is_baseline=False)
    results = analyzer.analyze_new_data()
    
    if not results:
        raise Exception("TFDV validation failed")
    
    print()
    print("  " + "-" * 56)
    
    # Save validation results
    validation_dir = gcp_config.LOCAL_PATHS['validation']
    run_dir = validation_dir / "runs" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = run_dir / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Upload reports to GCS
    gcs.upload_file(
        str(report_file),
        f"RAG/validation/runs/run_{timestamp}/validation_report.json"
    )
    gcs.upload_file(
        str(report_file),
        "RAG/validation/latest/validation_report.json"
    )
    
    # Analyze and report drift
    drift_results = results.get('drift_results', {})
    drift_features = []
    critical_drift = []
    
    if drift_results:
        for feature, drift_info in drift_results.items():
            if drift_info and drift_info.get('has_drift'):
                drift_features.append(feature)
                
                # Identify critical drift
                if 'mean_shift' in drift_info:
                    mean_shift = drift_info['mean_shift']
                    pct_change = abs(drift_info.get('percent_mean_change', 0))
                    
                    if mean_shift > 0.5 or pct_change > 30:
                        critical_drift.append(feature)
    
    # Print summary
    print()
    print("  " + "=" * 56)
    print("  VALIDATION SUMMARY")
    print("  " + "=" * 56)
    
    if drift_features:
        print(f"   Drift Detected: {len(drift_features)} features")
        
        if critical_drift:
            print(f"  CRITICAL DRIFT: {len(critical_drift)} features")
            for feature in critical_drift[:3]:
                drift_info = drift_results[feature]
                change = drift_info.get('percent_mean_change', 0)
                print(f"     • {feature}: {change:+.1f}% change vs baseline")
            
            print()
            print("    ACTION REQUIRED:")
            print("     • Review validation report in GCS")
            print("     • Investigate cause of critical drift")
            print("     • Determine if baseline update needed")
        else:
            print(f"   Moderate drift detected")
            print(f"     Top drifted features:")
            for feature in drift_features[:3]:
                drift_info = drift_results[feature]
                if 'percent_mean_change' in drift_info:
                    print(f"     • {feature}: {drift_info['percent_mean_change']:+.1f}%")
            
            print()
            print("   Monitor trend over next few runs")
    else:
        print("   No significant drift from FIXED baseline")
        print("     Data quality is consistent with reference")
    
    # Report anomalies
    anomalies = results.get('anomalies', {})
    total_anomalies = anomalies.get('total_anomalies', 0)
    
    if total_anomalies > 0:
        print(f"   Anomalies: {total_anomalies} records violate baseline statistics")
    else:
        print(f"   No anomalies detected")
    
    print("  " + "=" * 56)
    print()
    print(" Validation complete")
    print("=" * 60)
    
    # Store validation results
    ti.xcom_push(key='validation_passed', value=True)
    ti.xcom_push(key='is_first_run', value=False)
    ti.xcom_push(key='drift_detected', value=len(drift_features) > 0)
    ti.xcom_push(key='critical_drift', value=len(critical_drift) > 0)


def chunk_data(**context):
    """Task 4: Chunk merged data into smaller sections."""
    from RAG import chunking
    
    ti = context['ti']
    merged_file = ti.xcom_pull(task_ids='merge_data', key='merged_file')
    
    print("=" * 60)
    print("TASK 4: CHUNKING DATA")
    print("=" * 60)
    
    output_file = gcp_config.LOCAL_PATHS['chunked_data'] / "chunks_temp.json"
    
    # Temporarily override module-level paths
    original_input = chunking.INPUT_FILE
    original_output = chunking.OUTPUT_FILE
    
    chunking.INPUT_FILE = Path(merged_file)
    chunking.OUTPUT_FILE = output_file
    
    # Call existing chunking main()
    chunking.main()
    
    # Restore original paths
    chunking.INPUT_FILE = original_input
    chunking.OUTPUT_FILE = original_output
    
    # Upload to GCS with versioning
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    version = upload_with_versioning(
        gcs,
        str(output_file),
        "RAG/chunked_data",
        "chunks_{version}.json"
    )
    
    ti.xcom_push(key='chunks_file', value=str(output_file))
    
    print()
    print(f"Chunking complete: version {version}")
    print("=" * 60)


def generate_embeddings(**context):
    """Task 5: Generate vector embeddings for chunks."""
    from RAG import embedding
    
    ti = context['ti']
    chunks_file = ti.xcom_pull(task_ids='chunk_data', key='chunks_file')
    
    print("=" * 60)
    print("TASK 5: GENERATING EMBEDDINGS")
    print("=" * 60)
    print(f"Model: {gcp_config.get_embedding_model()}")
    print()
    
    output_file = gcp_config.LOCAL_PATHS['index'] / "embeddings_temp.json"
    
    # Temporarily override module-level paths
    original_input = embedding.INPUT_FILE
    original_output = embedding.OUTPUT_FILE
    
    embedding.INPUT_FILE = Path(chunks_file)
    embedding.OUTPUT_FILE = output_file
    
    # Call existing embedding main()
    embedding.main()
    
    # Restore original paths
    embedding.INPUT_FILE = original_input
    embedding.OUTPUT_FILE = original_output
    
    # Upload to GCS with versioning
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    version = upload_with_versioning(
        gcs,
        str(output_file),
        "RAG/index",
        "embeddings_{version}.json"
    )
    
    ti.xcom_push(key='embeddings_file', value=str(output_file))
    
    print()
    print(f" Embeddings complete: version {version}")
    print("=" * 60)


def create_index(**context):
    """Task 6: Create FAISS index for similarity search."""
    from RAG import indexing
    
    ti = context['ti']
    embeddings_file = ti.xcom_pull(task_ids='generate_embeddings', key='embeddings_file')
    
    print("=" * 60)
    print("TASK 6: CREATING FAISS INDEX")
    print("=" * 60)
    
    index_file = gcp_config.LOCAL_PATHS['index'] / "index_temp.bin"
    data_file = gcp_config.LOCAL_PATHS['index'] / "data_temp.pkl"
    
    # Temporarily override module-level paths
    original_input = indexing.INPUT_FILE
    original_index = indexing.OUTPUT_FILE_INDEX
    original_data = indexing.OUTPUT_FILE_DATA
    
    indexing.INPUT_FILE = Path(embeddings_file)
    indexing.OUTPUT_FILE_INDEX = index_file
    indexing.OUTPUT_FILE_DATA = data_file
    
    # Call existing indexing main()
    indexing.main()
    
    # Restore original paths
    indexing.INPUT_FILE = original_input
    indexing.OUTPUT_FILE_INDEX = original_index
    indexing.OUTPUT_FILE_DATA = original_data
    
    # Upload to GCS with versioning
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    idx_version = upload_with_versioning(
        gcs,
        str(index_file),
        "RAG/index",
        "index_{version}.bin"
    )
    
    upload_with_versioning(
        gcs,
        str(data_file),
        "RAG/index",
        "data_{version}.pkl"
    )
    
    print()
    print(f" Indexing complete: version {idx_version}")
    print("=" * 60)


def dvc_operations(**context):
    """Task 7: Version control with DVC (push to RAG remote)."""
    ti = context['ti']
    version = ti.xcom_pull(task_ids='merge_data', key='version')
    
    print("=" * 60)
    print("TASK 7: DVC VERSION CONTROL")
    print("=" * 60)
    print(f"Data version: {version}")
    print(f"DVC remote: rag (gs://medscan-rag-data)")
    print()
    
    # DVC add RAG data directory
    print("  Adding data/RAG/ to DVC...")
    result = subprocess.run(
        ['dvc', 'add', 'data/RAG/'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"   DVC add output: {result.stderr}")
    else:
        print("  DVC add successful")
    
    # DVC push to RAG remote (IMPORTANT: explicit -r rag flag)
    print("  📤 Pushing to GCS (rag remote)...")
    result = subprocess.run(
        ['dvc', 'push', '-r', 'rag'],  # Use RAG remote explicitly
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"    DVC push output: {result.stderr}")
    else:
        print("   DVC push to GCS successful")
    
    # Git operations (commit .dvc files)
    print("  Committing .dvc files to Git...")
    
    subprocess.run(
        ['git', 'add', 'data/RAG.dvc', '.gitignore'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True
    )
    
    result = subprocess.run(
        ['git', 'commit', '-m', f'Update RAG data version {version}'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
        print("   No changes to commit")
    else:
        print("  Git commit successful")
    
    print()
    print(f" DVC operations complete (version {version})")
    print("=" * 60)


# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    dag_id='rag_data_pipeline',
    default_args=default_args,
    description='RAG data pipeline with FIXED baseline validation',
    schedule_interval='0 10 * * *',  # Daily at 10 AM UTC
    start_date=days_ago(1),
    catchup=False,
    tags=['rag', 'mlops', 'medical', 'fixed-baseline'],
    max_active_runs=1,
) as dag:
    
    # Define tasks
    scrape = PythonOperator(
        task_id='scrape_data',
        python_callable=scrape_data,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
    )
    
    merge = PythonOperator(
        task_id='merge_data',
        python_callable=merge_data,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
    )
    
    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
    )
    
    chunk = PythonOperator(
        task_id='chunk_data',
        python_callable=chunk_data,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
    )
    
    embed = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
    )
    
    index = PythonOperator(
        task_id='create_index',
        python_callable=create_index,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
    )
    
    dvc = PythonOperator(
        task_id='dvc_operations',
        python_callable=dvc_operations,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
    )
    
    # Task dependencies (linear flow matching main.py)
    scrape >> merge >> validate >> chunk >> embed >> index >> dvc