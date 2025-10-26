"""RAG Data Pipeline DAG - Works alongside existing vision pipeline."""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import asyncio
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Add paths
PROJECT_ROOT = Path("/opt/airflow")  # Inside Docker
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "config"))

# Import configurations
import gcp_config
from RAG.common_utils import GCSManager, upload_with_versioning

default_args = {
    'owner': 'rag-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def scrape_data(**context):
    """Task 1: Scrape data."""
    from RAG.scraper import main as scraper_main
    
    execution_date = context['execution_date']
    timestamp = execution_date.strftime('%Y%m%d_%H%M%S')
    
    urls = gcp_config.get_scraping_urls()
    output_file = gcp_config.LOCAL_PATHS['raw_data'] / f"batch_{timestamp}.jsonl"
    
    print(f"Scraping {len(urls)} URLs...")
    asyncio.run(scraper_main(urls, str(output_file), method='W'))
    
    # Upload to GCS
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    gcs.upload_file(str(output_file), f"RAG/raw_data/incremental/batch_{timestamp}.jsonl")
    
    context['ti'].xcom_push(key='batch_file', value=str(output_file))
    context['ti'].xcom_push(key='timestamp', value=timestamp)
    print(f" Scraped {len(urls)} URLs")


def merge_data(**context):
    """Task 2: Merge with baseline."""
    from RAG.merge_batches import merge_jsonl_files
    
    ti = context['ti']
    batch_file = ti.xcom_pull(task_ids='scrape_data', key='batch_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Download baseline or latest merged
    local_baseline = gcp_config.LOCAL_PATHS['merged'] / "combined_latest.jsonl"
    
    if gcs.blob_exists("RAG/merged/combined_latest.jsonl"):
        gcs.download_file("RAG/merged/combined_latest.jsonl", str(local_baseline))
    else:
        gcs.download_file("RAG/raw_data/baseline/baseline.jsonl", str(local_baseline))
    
    # Merge
    output_file = gcp_config.LOCAL_PATHS['merged'] / f"combined_{timestamp}.jsonl"
    merge_jsonl_files(local_baseline, Path(batch_file), output_file)
    
    # Upload versioned
    version = upload_with_versioning(
        gcs, str(output_file), "RAG/merged", "combined_{version}.jsonl"
    )
    
    ti.xcom_push(key='merged_file', value=str(output_file))
    ti.xcom_push(key='version', value=version)
    print(f" Merged → version {version}")


def validate_data(**context):
    """Task 3: Run validation."""
    import json
    import shutil
    
    ti = context['ti']
    merged_file = ti.xcom_pull(task_ids='merge_data', key='merged_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Check if baseline exists
    baseline_exists = gcs.blob_exists("RAG/validation/baseline/baseline_report.json")
    
    # Prepare validation inputs
    if not baseline_exists:
        baseline_input = gcp_config.PROJECT_ROOT / "data" / "scraped_baseline.jsonl"
        shutil.copy(merged_file, baseline_input)
    else:
        baseline_path = gcp_config.PROJECT_ROOT / "data" / "scraped_baseline.jsonl"
        new_data_path = gcp_config.PROJECT_ROOT / "data" / "scraped_updated.jsonl"
        gcs.download_file("RAG/raw_data/baseline/baseline.jsonl", str(baseline_path))
        shutil.copy(merged_file, new_data_path)
    
    # Run validation
    from RAG.analysis.main import DataQualityAnalyzer
    
    analyzer = DataQualityAnalyzer(is_baseline=not baseline_exists)
    results = analyzer.analyze_baseline() if not baseline_exists else analyzer.analyze_new_data()
    
    # Save results
    validation_dir = gcp_config.LOCAL_PATHS['validation']
    
    if not baseline_exists:
        report_file = validation_dir / "baseline" / "baseline_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        gcs.upload_file(str(report_file), "RAG/validation/baseline/baseline_report.json")
    else:
        report_file = validation_dir / "runs" / f"run_{timestamp}" / "validation_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        gcs.upload_file(str(report_file), f"RAG/validation/runs/run_{timestamp}/validation_report.json")
        gcs.upload_file(str(report_file), "RAG/validation/latest/validation_report.json")
    
    print(" Validation complete")


def chunk_data(**context):
    """Task 4: Chunk data."""
    from RAG import chunking
    
    ti = context['ti']
    merged_file = ti.xcom_pull(task_ids='merge_data', key='merged_file')
    
    output_file = gcp_config.LOCAL_PATHS['chunked_data'] / "chunks_temp.json"
    
    # Temporarily override paths
    original_input = chunking.INPUT_FILE
    original_output = chunking.OUTPUT_FILE
    
    chunking.INPUT_FILE = Path(merged_file)
    chunking.OUTPUT_FILE = output_file
    chunking.main()
    
    chunking.INPUT_FILE = original_input
    chunking.OUTPUT_FILE = original_output
    
    # Upload
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    version = upload_with_versioning(gcs, str(output_file), "RAG/chunked_data", "chunks_{version}.json")
    
    ti.xcom_push(key='chunks_file', value=str(output_file))
    print(f" Chunked → version {version}")


def generate_embeddings(**context):
    """Task 5: Generate embeddings."""
    from RAG import embedding
    
    ti = context['ti']
    chunks_file = ti.xcom_pull(task_ids='chunk_data', key='chunks_file')
    
    output_file = gcp_config.LOCAL_PATHS['index'] / "embeddings_temp.json"
    
    # Temporarily override paths
    original_input = embedding.INPUT_FILE
    original_output = embedding.OUTPUT_FILE
    
    embedding.INPUT_FILE = Path(chunks_file)
    embedding.OUTPUT_FILE = output_file
    embedding.main()
    
    embedding.INPUT_FILE = original_input
    embedding.OUTPUT_FILE = original_output
    
    # Upload
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    version = upload_with_versioning(gcs, str(output_file), "RAG/index", "embeddings_{version}.json")
    
    ti.xcom_push(key='embeddings_file', value=str(output_file))
    print(f" Embedded → version {version}")


def create_index(**context):
    """Task 6: Create FAISS index."""
    from RAG import indexing
    
    ti = context['ti']
    embeddings_file = ti.xcom_pull(task_ids='generate_embeddings', key='embeddings_file')
    
    index_file = gcp_config.LOCAL_PATHS['index'] / "index_temp.bin"
    data_file = gcp_config.LOCAL_PATHS['index'] / "data_temp.pkl"
    
    # Temporarily override paths
    original_input = indexing.INPUT_FILE
    original_index = indexing.OUTPUT_FILE_INDEX
    original_data = indexing.OUTPUT_FILE_DATA
    
    indexing.INPUT_FILE = Path(embeddings_file)
    indexing.OUTPUT_FILE_INDEX = index_file
    indexing.OUTPUT_FILE_DATA = data_file
    indexing.main()
    
    indexing.INPUT_FILE = original_input
    indexing.OUTPUT_FILE_INDEX = original_index
    indexing.OUTPUT_FILE_DATA = original_data
    
    # Upload
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    upload_with_versioning(gcs, str(index_file), "RAG/index", "index_{version}.bin")
    upload_with_versioning(gcs, str(data_file), "RAG/index", "data_{version}.pkl")
    
    print(" Indexed")


def dvc_operations(**context):
    """Task 7: DVC commit and push to RAG remote."""
    import subprocess
    
    ti = context['ti']
    version = ti.xcom_pull(task_ids='merge_data', key='version')
    
    print("📦 Running DVC operations for RAG data...")
    
    # DVC add RAG data
    result = subprocess.run(
        ['dvc', 'add', 'data/RAG/'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  DVC add warning: {result.stderr}")
    
    # DVC push to RAG remote (IMPORTANT: use -r rag)
    result = subprocess.run(
        ['dvc', 'push', '-r', 'rag'],  # ← Explicitly use 'rag' remote
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  DVC push warning: {result.stderr}")
    else:
        print("✓ Pushed to GCS rag remote")
    
    # Git commit .dvc files
    subprocess.run(
        ['git', 'add', 'data/RAG.dvc', '.gitignore'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True
    )
    
    subprocess.run(
        ['git', 'commit', '-m', f'Update RAG data v{version}'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True
    )
    
    print(f"DVC operations complete (version {version})")


with DAG(
    dag_id='rag_data_pipeline',
    default_args=default_args,
    description='RAG pipeline',
    schedule_interval='0 10 * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['rag', 'mlops'],
) as dag:
    
    scrape = PythonOperator(task_id='scrape_data', python_callable=scrape_data, provide_context=True)
    merge = PythonOperator(task_id='merge_data', python_callable=merge_data, provide_context=True)
    validate = PythonOperator(task_id='validate_data', python_callable=validate_data, provide_context=True)
    chunk = PythonOperator(task_id='chunk_data', python_callable=chunk_data, provide_context=True)
    embed = PythonOperator(task_id='generate_embeddings', python_callable=generate_embeddings, provide_context=True)
    index = PythonOperator(task_id='create_index', python_callable=create_index, provide_context=True)
    dvc = PythonOperator(task_id='dvc_operations', python_callable=dvc_operations, provide_context=True)
    
    scrape >> merge >> validate >> chunk >> embed >> index >> dvc