"""
RAG Data Pipeline DAG

Flow:
1. Scrape ALL URLs (from GCS urls.txt)
2a. Check if baseline exists
2b. Branch: Create baseline OR Validate against baseline
3. Chunk → 4. Embed → 5. Index
6. DVC track
"""
from datetime import timedelta
from pathlib import Path
import sys
import asyncio
import subprocess
import shutil
import json

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path("/opt/airflow")
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "config"))

try:
    import gcp_config
    from RAG.common_utils import GCSManager
    from RAG.url_manager import read_urls_from_gcs
except ImportError as e:
    print(f"Import error: {e}")
    raise

default_args = {
    'owner': 'rag-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=45),
}


def scrape_data(**context):
    """Task 1: Scrape ALL URLs from GCS config."""
    from RAG.scraper import main as scraper_main
    
    execution_date = context['execution_date']
    timestamp = execution_date.strftime('%Y%m%d_%H%M%S')
    
    # Get GCS manager
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Read URLs from GCS
    urls_file_path = gcp_config.get_urls_file_path()
    urls = read_urls_from_gcs(gcs, urls_file_path)
    
    print(f"Total URLs: {len(urls)}")
    
    # Scrape all URLs with timeout handling
    output_file = gcp_config.LOCAL_PATHS['raw_data'] / f"scraped_{timestamp}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run scraper with overall timeout
        asyncio.run(
            asyncio.wait_for(
                scraper_main(urls, str(output_file), method='W'),
                timeout=1800  # 30 minute max for all URLs
            )
        )
    except asyncio.TimeoutError:
        print("Scraping timeout - processing partial results")
    except Exception as e:
        print(f"Scraping error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    if not output_file.exists():
        print("Output file was not created!")
        raise FileNotFoundError(f"Scraper did not create output file: {output_file}")
    
    # Analyze results
    import json
    total_records = 0
    error_records = 0
    success_records = 0
    
    with open(output_file, 'r') as f:
        for line in f:
            total_records += 1
            try:
                record = json.loads(line)
                if 'error' in record:
                    error_records += 1
                    print(f"Error in record {total_records}:")
                    print(f"URL: {record.get('link', 'unknown')}")
                    print(f"Error: {record.get('error', 'unknown error')}")
                else:
                    success_records += 1
                    print(f"Success record {total_records}:")
                    print(f"URL: {record.get('link', 'unknown')}")
            except json.JSONDecodeError:
                error_records += 1
                print(f"  Invalid JSON at line {total_records}")
    
    # Upload to GCS incremental
    gcs.upload_file(str(output_file), f"RAG/raw_data/incremental/scraped_{timestamp}.jsonl")
    
    context['ti'].xcom_push(key='scraped_file', value=str(output_file))
    context['ti'].xcom_push(key='timestamp', value=timestamp)
    context['ti'].xcom_push(key='scraped_count', value=success_records)  # Use success count
    print("Scraping task complete")


def check_baseline_exists(**context):
    """Task 2a: Check if baseline exists."""
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    baseline_exists = gcs.blob_exists("RAG/raw_data/baseline/baseline.jsonl")
    stats_exist = gcs.blob_exists("RAG/validation/baseline_stats.json")
    
    # Both must exist for validation to work
    is_first_run = not (baseline_exists and stats_exist)
    
    if is_first_run:
        if baseline_exists and not stats_exist:
            print("WARNING: Baseline exists but stats missing!")
        else:
            print("First run detected")
    else:
        print("Baseline ready for validation")
    
    print()
    
    context['ti'].xcom_push(key='baseline_exists', value=baseline_exists)
    context['ti'].xcom_push(key='stats_exist', value=stats_exist)
    context['ti'].xcom_push(key='is_first_run', value=is_first_run)

def decide_validation_path(**context):
    """Decide whether to create baseline or validate."""
    ti = context['ti']
    is_first_run = ti.xcom_pull(task_ids='check_baseline', key='is_first_run')
    baseline_exists = ti.xcom_pull(task_ids='check_baseline', key='baseline_exists')
    stats_exist = ti.xcom_pull(task_ids='check_baseline', key='stats_exist')
    
    print(f"Baseline exists: {baseline_exists}")
    print(f"Stats exist: {stats_exist}")
    print(f"Is first run: {is_first_run}")
    print()
    
    if is_first_run:
        return 'create_baseline'
    else:
        return 'validate_data'


def create_baseline(**context):
    """Task 2b: Create baseline (first run only)."""
    from RAG.analysis.main import DataQualityAnalyzer
    
    ti = context['ti']
    scraped_file = ti.xcom_pull(task_ids='scrape_data', key='scraped_file')
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Upload scraped data as baseline
    gcs.upload_file(
        scraped_file,
        "RAG/raw_data/baseline/baseline.jsonl"
    )
    # Generate baseline stats
    analyzer = DataQualityAnalyzer(gcs_manager=gcs, is_baseline=True)
    
    # Load data
    baseline_df = analyzer.load_jsonl(Path(scraped_file))
    
    # Generate stats (this also uploads to GCS)
    baseline_data = analyzer.generate_baseline_stats(baseline_df)
    
    # Verify upload
    stats_uploaded = gcs.blob_exists("RAG/validation/baseline_stats.json")
    
    # Pass through for next tasks
    ti.xcom_push(key='validation_passed', value=True)
    ti.xcom_push(key='validated_file', value=scraped_file)


def validate_data(**context):
    """Task 2c: Validate against baseline."""
    from RAG.analysis.main import DataQualityAnalyzer
    
    ti = context['ti']
    scraped_file = ti.xcom_pull(task_ids='scrape_data', key='scraped_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    scraped_count = ti.xcom_pull(task_ids='scrape_data', key='scraped_count')
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Double-check that baseline stats exist
    baseline_exists = gcs.blob_exists("RAG/raw_data/baseline/baseline.jsonl")
    stats_exist = gcs.blob_exists("RAG/validation/baseline_stats.json")
    
    if not baseline_exists or not stats_exist:
        error_msg = "Cannot validate: "
        if not baseline_exists:
            error_msg += "baseline data missing"
        if not stats_exist:
            error_msg += (" and " if not baseline_exists else "") + "baseline stats missing"
        
        raise FileNotFoundError(f"{error_msg}. Run create_baseline first.")
    
    # Initialize analyzer with GCS manager
    analyzer = DataQualityAnalyzer(gcs_manager=gcs, is_baseline=False)
    
    # Load new data
    new_df = analyzer.load_jsonl(Path(scraped_file))
    
    # Validate against baseline
    results = analyzer.validate_against_baseline(new_df)
    
    # Save validation report
    report_file = gcp_config.LOCAL_PATHS['validation'] / f"validation_{timestamp}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Upload report
    gcs.upload_file(
        str(report_file),
        f"RAG/validation/reports/validation_{timestamp}.json"
    )
    
    # Quality gate
    anomaly_pct = results['anomalies']['percentage']
    val_config = gcp_config.get_validation_config()
    MAX_ANOMALY_PCT = val_config.get('max_anomaly_pct', 25.0)
    
    print(f"Total records: {results['total_records']}")
    print(f"Anomalies: {results['anomalies']['total']} ({anomaly_pct:.1f}%)")
    print(f"Completeness issues: {sum(len(v) for v in results['completeness'].values())}")
    
    drift_count = len([f for f, d in results['drift'].items() if d and d.get('has_drift')])
    print(f"Drift features: {drift_count}")
    
    # Quality gate check
    if anomaly_pct > MAX_ANOMALY_PCT:
        raise Exception(f"Validation failed: {anomaly_pct:.1f}% anomalies exceeds threshold")
    
    ti.xcom_push(key='validation_passed', value=True)
    ti.xcom_push(key='validated_file', value=scraped_file)
    ti.xcom_push(key='validation_report', value=str(report_file))
    

def chunk_data(**context):
    """Task 3: Chunk data."""
    from RAG.chunking import RAGChunker
    
    ti = context['ti']
    
    # Get validated file from either branch
    validated_file = ti.xcom_pull(task_ids='validate_data', key='validated_file')
    if not validated_file:
        validated_file = ti.xcom_pull(task_ids='create_baseline', key='validated_file')
    
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    # Verify input file exists
    if not Path(validated_file).exists():
        raise FileNotFoundError(f"Input file not found: {validated_file}")
    
    # Initialize chunker
    chunker = RAGChunker()
    
    # Process JSONL file (returns chunks in memory)
    chunks = chunker.process_jsonl(Path(validated_file))
    
    # Save to temp file for next task
    temp_file = gcp_config.LOCAL_PATHS['temp'] / f"chunks_{timestamp}.json"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    chunker.save_chunks(chunks, temp_file)
    
    ti.xcom_push(key='chunks_file', value=str(temp_file))
    ti.xcom_push(key='chunk_count', value=len(chunks))


def generate_embeddings(**context):
    """Task 4: Generate embeddings."""
    from RAG.embedding import ChunkEmbedder
    import json
    
    ti = context['ti']
    chunks_file = ti.xcom_pull(task_ids='chunk_data', key='chunks_file')
    chunk_count = ti.xcom_pull(task_ids='chunk_data', key='chunk_count')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    if not Path(chunks_file).exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Generate embeddings
    embedder = ChunkEmbedder(model_name=gcp_config.get_embedding_model())
    embedded_chunks = embedder.embed_chunks(chunks)
    
    if len(embedded_chunks) == 0:
        raise ValueError(
            "Failed to generate any embeddings! "
            "Check logs for details about chunk content and structure."
        )
    
    # Save to temp file
    temp_file = gcp_config.LOCAL_PATHS['temp'] / f"embeddings_{timestamp}.json"
    embedder.save_embeddings(embedded_chunks, temp_file)
    
    ti.xcom_push(key='embeddings_file', value=str(temp_file))
    ti.xcom_push(key='embedding_count', value=len(embedded_chunks))
    
    # Cleanup chunks file
    Path(chunks_file).unlink(missing_ok=True)
    

def create_index(**context):
    """Task 5: Create and save FAISS index."""
    from RAG.indexing import FAISSIndex
    from RAG.embedding import ChunkEmbedder
    
    ti = context['ti']
    embeddings_file = ti.xcom_pull(task_ids='generate_embeddings', key='embeddings_file')
    embedding_count = ti.xcom_pull(task_ids='generate_embeddings', key='embedding_count')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    if not Path(embeddings_file).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    # Load embeddings
    embedded_chunks = ChunkEmbedder.load_embeddings(Path(embeddings_file))
    
    # SAFETY CHECK
    if not embedded_chunks or len(embedded_chunks) == 0:
        raise ValueError(
            "No embedded chunks loaded! "
            "The embeddings file exists but contains no data. "
            "Check the embedding task output."
        )
    
    # Create index
    embedding_dim = embedded_chunks[0].embedding.shape[0]
    print(f"  🔢 Embedding dimension: {embedding_dim}")
    
    faiss_index = FAISSIndex(dimension=embedding_dim, index_type='flat')
    faiss_index.build_index(embedded_chunks)
    
    print(f"Built FAISS index: {faiss_index.index.ntotal} vectors")
    print()
    
    # Define output paths
    index_dir = gcp_config.LOCAL_PATHS['index']
    index_dir.mkdir(parents=True, exist_ok=True)
    
    index_timestamped = index_dir / f"index_{timestamp}.bin"
    data_timestamped = index_dir / f"data_{timestamp}.pkl"
    index_latest = index_dir / "index_latest.bin"
    data_latest = index_dir / "data_latest.pkl"
    
    # Save timestamped version
    faiss_index.save_index(index_timestamped, data_timestamped)
    
    # Save latest version
    shutil.copy(index_timestamped, index_latest)
    shutil.copy(data_timestamped, data_latest)
    
    # Upload to GCS
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    gcs.upload_file(str(index_timestamped), f"RAG/index/index_{timestamp}.bin")
    gcs.upload_file(str(data_timestamped), f"RAG/index/data_{timestamp}.pkl")
    gcs.upload_file(str(index_latest), "RAG/index/index_latest.bin")
    gcs.upload_file(str(data_latest), "RAG/index/data_latest.pkl")
    
    # Cleanup temp files
    Path(embeddings_file).unlink(missing_ok=True)
    
    ti.xcom_push(key='index_timestamp', value=timestamp)
    

def dvc_operations(**context):
    """Task 6: DVC track incremental + index."""
    ti = context['ti']
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    # Track specific paths only
    paths_to_track = [
        "data/RAG/raw_data/incremental/",
        "data/RAG/index/",
    ]
    
    for path in paths_to_track:
        subprocess.run(
            ['dvc', 'add', path],
            cwd=str(gcp_config.PROJECT_ROOT),
            capture_output=True
        )
    
    # Push to RAG remote
    result = subprocess.run(
        ['dvc', 'push', '-r', 'rag'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    # Git commit
    subprocess.run(
        ['git', 'add', 'data/RAG/raw_data/incremental.dvc', 'data/RAG/index.dvc'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True
    )
    
    subprocess.run(
        ['git', 'commit', '-m', f'RAG run {timestamp}'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True
    )

with DAG(
    dag_id='rag_data_pipeline',
    default_args=default_args,
    description='RAG pipeline',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['rag', 'split-validation', 'gcs-aware'],
    max_active_runs=1,
) as dag:
    
    scrape = PythonOperator(
        task_id='scrape_data',
        python_callable=scrape_data,
        provide_context=True,
    )
    
    check_baseline = PythonOperator(
        task_id='check_baseline',
        python_callable=check_baseline_exists,
        provide_context=True,
    )
    
    decide = BranchPythonOperator(
        task_id='decide_validation_path',
        python_callable=decide_validation_path,
        provide_context=True,
    )
    
    create_baseline_task = PythonOperator(
        task_id='create_baseline',
        python_callable=create_baseline,
        provide_context=True,
    )
    
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )
    
    chunk = PythonOperator(
        task_id='chunk_data',
        python_callable=chunk_data,
        provide_context=True,
        trigger_rule='none_failed_min_one_success',
    )
    
    embed = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
        provide_context=True,
    )
    
    index = PythonOperator(
        task_id='create_index',
        python_callable=create_index,
        provide_context=True,
    )
    
    dvc = PythonOperator(
        task_id='dvc_operations',
        python_callable=dvc_operations,
        provide_context=True,
    )
    
    # Flow
    scrape >> check_baseline >> decide
    decide >> [create_baseline_task, validate_task]
    [create_baseline_task, validate_task] >> chunk >> embed >> index >> dvc