"""
RAG Data Pipeline DAG - FINAL VERSION with Split Validation

Flow:
1. Scrape ALL URLs (from GCS urls.txt)
2a. Check if baseline exists
2b. Branch: Create baseline OR Validate against baseline
3. Chunk → 4. Embed → 5. Index
6. DVC track
"""
from datetime import datetime, timedelta
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
    print(f"❌ Import error: {e}")
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
    
    print("=" * 60)
    print("TASK 1: SCRAPING DATA")
    print("=" * 60)
    
    # Get GCS manager
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Read URLs from GCS
    print("  📥 Reading URLs from GCS...")
    urls_file_path = gcp_config.get_urls_file_path()
    urls = read_urls_from_gcs(gcs, urls_file_path)
    
    print(f"  📊 Total URLs: {len(urls)}")
    print()
    
    # Scrape all URLs with timeout handling
    output_file = gcp_config.LOCAL_PATHS['raw_data'] / f"scraped_{timestamp}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  🔍 Scraping {len(urls)} URLs (this may take 5-10 minutes)...")
    print()
    
    try:
        # Run scraper with overall timeout
        asyncio.run(
            asyncio.wait_for(
                scraper_main(urls, str(output_file), method='W'),
                timeout=1800  # 30 minute max for all URLs
            )
        )
    except asyncio.TimeoutError:
        print("⚠️  Scraping timeout - processing partial results")
    except Exception as e:
        print(f"Scraping error: {e}")
        raise

    
    # Count scraped
    with open(output_file, 'r') as f:
        scraped_count = sum(1 for _ in f)
    
    print(f"  ✅ Scraped {scraped_count} articles")
    print()
    
    # Upload to GCS incremental
    gcs.upload_file(str(output_file), f"RAG/raw_data/incremental/scraped_{timestamp}.jsonl")
    
    context['ti'].xcom_push(key='scraped_file', value=str(output_file))
    context['ti'].xcom_push(key='timestamp', value=timestamp)
    context['ti'].xcom_push(key='scraped_count', value=scraped_count)
    
    print("✅ Scraping complete")
    print("=" * 60)


def check_baseline_exists(**context):
    """Task 2a: Check if baseline exists."""
    print("=" * 60)
    print("TASK 2a: CHECK BASELINE")
    print("=" * 60)
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    baseline_exists = gcs.blob_exists("RAG/raw_data/baseline/baseline.jsonl")
    stats_exist = gcs.blob_exists("RAG/validation/baseline_stats.json")
    
    print(f"  Baseline data exists: {baseline_exists}")
    print(f"  Baseline stats exist: {stats_exist}")
    print()
    
    # Both must exist for validation to work
    is_first_run = not (baseline_exists and stats_exist)
    
    if is_first_run:
        if baseline_exists and not stats_exist:
            print("  ⚠️  WARNING: Baseline exists but stats missing!")
            print("  🔄 Will regenerate baseline with stats")
        else:
            print("  🆕 First run detected")
    else:
        print("  ✅ Baseline ready for validation")
    
    print()
    
    context['ti'].xcom_push(key='baseline_exists', value=baseline_exists)
    context['ti'].xcom_push(key='stats_exist', value=stats_exist)
    context['ti'].xcom_push(key='is_first_run', value=is_first_run)
    
    print("=" * 60)


def decide_validation_path(**context):
    """Decide whether to create baseline or validate."""
    ti = context['ti']
    is_first_run = ti.xcom_pull(task_ids='check_baseline', key='is_first_run')
    baseline_exists = ti.xcom_pull(task_ids='check_baseline', key='baseline_exists')
    stats_exist = ti.xcom_pull(task_ids='check_baseline', key='stats_exist')
    
    print("=" * 60)
    print("ROUTING DECISION")
    print("=" * 60)
    print(f"  Baseline exists: {baseline_exists}")
    print(f"  Stats exist: {stats_exist}")
    print(f"  Is first run: {is_first_run}")
    print()
    
    if is_first_run:
        print("  → Route: CREATE BASELINE")
        print("=" * 60)
        return 'create_baseline'
    else:
        print("  → Route: VALIDATE DATA")
        print("=" * 60)
        return 'validate_data'


def create_baseline(**context):
    """Task 2b: Create baseline (first run only)."""
    from RAG.analysis.main import DataQualityAnalyzer
    
    ti = context['ti']
    scraped_file = ti.xcom_pull(task_ids='scrape_data', key='scraped_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    print("=" * 60)
    print("TASK 2b: CREATE BASELINE")
    print("=" * 60)
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    # Upload scraped data as baseline
    print("📤 Uploading baseline to GCS...")
    gcs.upload_file(
        scraped_file,
        "RAG/raw_data/baseline/baseline.jsonl"
    )
    print("✅ Baseline data uploaded to GCS")
    print()
    
    # Generate baseline stats
    print("📊 Generating baseline statistics...")
    analyzer = DataQualityAnalyzer(gcs_manager=gcs, is_baseline=True)
    
    # Load data
    baseline_df = analyzer.load_jsonl(Path(scraped_file))
    print(f"Loaded {len(baseline_df)} records")
    print()
    
    # Generate stats (this also uploads to GCS)
    baseline_data = analyzer.generate_baseline_stats(baseline_df)
    
    # Verify upload
    stats_uploaded = gcs.blob_exists("RAG/validation/baseline_stats.json")
    if stats_uploaded:
        print("✅ Baseline stats verified in GCS")
    else:
        print("⚠️  WARNING: Stats upload may have failed")
    
    print()
    print("✅ Baseline creation complete")
    print("=" * 60)
    
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
    
    print("=" * 60)
    print("TASK 2c: VALIDATE DATA")
    print("=" * 60)
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    print(f"📊 Validating {scraped_count} records")
    print()
    
    # Double-check that baseline stats exist
    print("🔍 Verifying baseline stats in GCS...")
    baseline_exists = gcs.blob_exists("RAG/raw_data/baseline/baseline.jsonl")
    stats_exist = gcs.blob_exists("RAG/validation/baseline_stats.json")
    
    print(f"  Baseline data: {baseline_exists}")
    print(f"  Baseline stats: {stats_exist}")
    print()
    
    if not baseline_exists or not stats_exist:
        error_msg = "Cannot validate: "
        if not baseline_exists:
            error_msg += "baseline data missing"
        if not stats_exist:
            error_msg += (" and " if not baseline_exists else "") + "baseline stats missing"
        
        print(f"❌ {error_msg}")
        print()
        print("💡 Solution: This task should not have run.")
        print("   The baseline creation task should have run first.")
        print("   Please check the decide_validation_path task.")
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
    MAX_ANOMALY_PCT = val_config.get('max_anomaly_pct', 20.0)
    
    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total records: {results['total_records']}")
    print(f"Anomalies: {results['anomalies']['total']} ({anomaly_pct:.1f}%)")
    print(f"Completeness issues: {sum(len(v) for v in results['completeness'].values())}")
    
    drift_count = len([f for f, d in results['drift'].items() if d and d.get('has_drift')])
    print(f"Drift features: {drift_count}")
    print("=" * 60)
    
    # Quality gate check
    if anomaly_pct > MAX_ANOMALY_PCT:
        print(f"🔴 FAILED: {anomaly_pct:.1f}% anomalous > {MAX_ANOMALY_PCT}% threshold")
        raise Exception(f"Validation failed: {anomaly_pct:.1f}% anomalies exceeds threshold")
    
    print(f"✅ PASSED: {anomaly_pct:.1f}% anomalies < {MAX_ANOMALY_PCT}% threshold")
    print()
    
    ti.xcom_push(key='validation_passed', value=True)
    ti.xcom_push(key='validated_file', value=scraped_file)
    ti.xcom_push(key='validation_report', value=str(report_file))
    
    print("✅ Validation complete")
    print("=" * 60)


def chunk_data(**context):
    """Task 3: Chunk data."""
    from RAG.chunking import RAGChunker
    
    ti = context['ti']
    
    # Get validated file from either branch
    validated_file = ti.xcom_pull(task_ids='validate_data', key='validated_file')
    if not validated_file:
        validated_file = ti.xcom_pull(task_ids='create_baseline', key='validated_file')
    
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    print("=" * 60)
    print("TASK 3: CHUNKING")
    print("=" * 60)
    
    print(f"  📄 Input file: {validated_file}")
    
    # Initialize chunker
    chunker = RAGChunker()
    
    # Process JSONL file (returns chunks in memory)
    chunks = chunker.process_jsonl(Path(validated_file))
    
    print(f"  ✅ Generated {len(chunks)} chunks")
    
    # Save to temp file for next task
    temp_file = gcp_config.LOCAL_PATHS['temp'] / f"chunks_{timestamp}.json"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    chunker.save_chunks(chunks, temp_file)
    
    ti.xcom_push(key='chunks_file', value=str(temp_file))
    ti.xcom_push(key='chunk_count', value=len(chunks))
    
    print("✅ Chunking complete")
    print("=" * 60)


def generate_embeddings(**context):
    """Task 4: Generate embeddings."""
    from RAG.embedding import ChunkEmbedder
    import json
    
    ti = context['ti']
    chunks_file = ti.xcom_pull(task_ids='chunk_data', key='chunks_file')
    chunk_count = ti.xcom_pull(task_ids='chunk_data', key='chunk_count')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    print("=" * 60)
    print("TASK 4: EMBEDDINGS")
    print("=" * 60)
    
    # Load chunks from file
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"  🧠 Generating embeddings for {chunk_count} chunks...")
    
    # Generate embeddings
    embedder = ChunkEmbedder(model_name=gcp_config.get_embedding_model())
    embedded_chunks = embedder.embed_chunks(chunks)
    
    print(f"  ✅ Generated {len(embedded_chunks)} embeddings")
    
    # Save to temp file
    temp_file = gcp_config.LOCAL_PATHS['temp'] / f"embeddings_{timestamp}.json"
    embedder.save_embeddings(embedded_chunks, temp_file)
    
    ti.xcom_push(key='embeddings_file', value=str(temp_file))
    ti.xcom_push(key='embedding_count', value=len(embedded_chunks))
    
    # Cleanup chunks file
    Path(chunks_file).unlink(missing_ok=True)
    
    print("✅ Embeddings complete")
    print("=" * 60)


def create_index(**context):
    """Task 5: Create and save FAISS index."""
    from RAG.indexing import FAISSIndex
    from RAG.embedding import ChunkEmbedder
    
    ti = context['ti']
    embeddings_file = ti.xcom_pull(task_ids='generate_embeddings', key='embeddings_file')
    embedding_count = ti.xcom_pull(task_ids='generate_embeddings', key='embedding_count')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    print("=" * 60)
    print("TASK 5: CREATE INDEX")
    print("=" * 60)
    
    print(f"  📚 Loading {embedding_count} embeddings...")
    
    # Load embeddings
    embedded_chunks = ChunkEmbedder.load_embeddings(Path(embeddings_file))
    
    # Create index
    embedding_dim = embedded_chunks[0].embedding.shape[0]
    faiss_index = FAISSIndex(dimension=embedding_dim, index_type='flat')
    faiss_index.build_index(embedded_chunks)
    
    print(f"  ✅ Built FAISS index: {faiss_index.index.ntotal} vectors")
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
    
    print(f"  💾 Saved locally:")
    print(f"     {index_timestamped.name}")
    print(f"     {data_timestamped.name}")
    print(f"     index_latest.bin")
    print(f"     data_latest.pkl")
    print()
    
    # Upload to GCS
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    gcs.upload_file(str(index_timestamped), f"RAG/index/index_{timestamp}.bin")
    gcs.upload_file(str(data_timestamped), f"RAG/index/data_{timestamp}.pkl")
    gcs.upload_file(str(index_latest), "RAG/index/index_latest.bin")
    gcs.upload_file(str(data_latest), "RAG/index/data_latest.pkl")
    
    print("  ☁️  Uploaded to GCS")
    
    # Cleanup temp files
    Path(embeddings_file).unlink(missing_ok=True)
    
    print("  🧹 Cleaned temp files")
    print()
    
    ti.xcom_push(key='index_timestamp', value=timestamp)
    
    print("✅ Index created")
    print("=" * 60)


def dvc_operations(**context):
    """Task 6: DVC track incremental + index."""
    ti = context['ti']
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    print("=" * 60)
    print("TASK 6: DVC VERSIONING")
    print("=" * 60)
    
    # Track specific paths only
    paths_to_track = [
        "data/RAG/raw_data/incremental/",
        "data/RAG/index/",
    ]
    
    for path in paths_to_track:
        print(f"  📦 Adding {path}...")
        subprocess.run(
            ['dvc', 'add', path],
            cwd=str(gcp_config.PROJECT_ROOT),
            capture_output=True
        )
    
    # Push to RAG remote
    print("  📤 Pushing to GCS (rag remote)...")
    result = subprocess.run(
        ['dvc', 'push', '-r', 'rag'],
        cwd=str(gcp_config.PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("  ✅ DVC push successful")
    else:
        print(f"  ⚠️  DVC: {result.stderr}")
    
    # Git commit
    print("  📝 Committing .dvc files...")
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
    
    print()
    print(f"✅ DVC complete ({timestamp})")
    print("=" * 60)


with DAG(
    dag_id='rag_data_pipeline',
    default_args=default_args,
    description='RAG pipeline - Split validation with branching',
    schedule_interval=None,  # Manual only
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
        trigger_rule='none_failed_min_one_success',  # Run after either branch
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
    
    # Flow with branching
    scrape >> check_baseline >> decide
    decide >> [create_baseline_task, validate_task]
    [create_baseline_task, validate_task] >> chunk >> embed >> index >> dvc