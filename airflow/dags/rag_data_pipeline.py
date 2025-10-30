"""RAG Data Pipeline DAG with DVC-Primary Storage"""
from datetime import timedelta
from pathlib import Path
import sys
import asyncio
import json
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path("/opt/airflow")
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "config"))

from DataPipeline.config import gcp_config
from DataPipeline.scripts.common.gcs_manager import GCSManager
from DataPipeline.scripts.common.dvc_helper import DVCManager
from DataPipeline.scripts.RAG.url_manager import read_urls_from_gcs
from DataPipeline.scripts.RAG import alert_utils

logger = logging.getLogger(__name__)

rag_pipeline_config = gcp_config.load_rag_pipeline_config()
ALERT_RECIPIENTS = gcp_config.ALERT_CONFIG['email_recipients']
ALERT_THRESHOLDS = gcp_config.RAG_THRESHOLDS

default_args = {
    'owner': 'rag-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ALERT_RECIPIENTS,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=45),
}


def scrape_data(**context):
    """Task 1: Scrape URLs."""
    from DataPipeline.scripts.RAG.scraper import main as scraper_main
    
    logger.info("TASK: Scrape Data - Started")
    execution_date = context['execution_date']
    timestamp = execution_date.strftime('%Y%m%d_%H%M%S')
    
    try:
        gcs = GCSManager.from_config()
        
        # Read URLs from GCS
        urls_file_path = rag_pipeline_config['scraping']['urls_file']
        urls = read_urls_from_gcs(gcs, urls_file_path)
        
        logger.info(f"Scraping {len(urls)} URLs")
        
        # Scrape all URLs
        output_file = gcp_config.LOCAL_PATHS['data'] / "RAG" / "raw_data" / "incremental" / f"scraped_{timestamp}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            asyncio.run(
                asyncio.wait_for(
                    scraper_main(urls, str(output_file), method='W'),
                    timeout=1800
                )
            )
        except asyncio.TimeoutError:
            logger.warning("Scraping timeout - processing partial results")
        
        if not output_file.exists():
            error_msg = "Scraper did not create output file"
            logger.error(error_msg)
            alert_utils.send_failure_alert('scrape_data', error_msg, context, ALERT_RECIPIENTS)
            raise FileNotFoundError(error_msg)
        
        # Analyze results
        success_records = 0
        error_records = 0
        
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'error' in record:
                        error_records += 1
                    else:
                        success_records += 1
                except json.JSONDecodeError:
                    error_records += 1
        
        success_rate = success_records / len(urls) if len(urls) > 0 else 0
        logger.info(f"Scraping: {success_records}/{len(urls)} successful ({success_rate*100:.1f}%)")
        
        # Check threshold
        min_success_rate = ALERT_THRESHOLDS.get('scraping_min_success_rate', 0.7)
        if success_rate < min_success_rate:
            alert_utils.send_threshold_alert(
                task_name='scrape_data',
                threshold_name='Scraping Success Rate',
                actual_value=success_rate * 100,
                threshold_value=min_success_rate * 100,
                context=context,
                recipients=ALERT_RECIPIENTS,
                additional_info={
                    'Successful URLs': success_records,
                    'Failed URLs': error_records,
                    'Total URLs': len(urls)
                }
            )
        
        context['ti'].xcom_push(key='scraped_file', value=str(output_file))
        context['ti'].xcom_push(key='timestamp', value=timestamp)
        context['ti'].xcom_push(key='success_count', value=success_records)
        
        logger.info("TASK: Scrape Data - Completed")
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        alert_utils.send_failure_alert('scrape_data', str(e), context, ALERT_RECIPIENTS)
        raise


def check_baseline_exists(**context):
    """Task 2a: Check if baseline exists."""
    logger.info("Checking for baseline data")
    
    baseline_path = Path("/opt/airflow/DataPipeline/data/RAG/raw_data/baseline/baseline.jsonl")
    stats_path = gcp_config.LOCAL_PATHS['validation'] / "baseline_stats.json"
    
    baseline_exists = baseline_path.exists()
    stats_exist = stats_path.exists()
    
    # Try to pull from DVC if not local
    if not baseline_exists:
        dvc = DVCManager()
        dvc.pull(remote="rag")
        baseline_exists = baseline_path.exists()
    
    # Check GCS for stats
    if not stats_exist:
        gcs = GCSManager.from_config()
        gcs_stats_path = gcp_config.get_gcs_path('rag', 'validation') + "baseline_stats.json"
        if gcs.blob_exists(gcs_stats_path):
            gcs.download_file(gcs_stats_path, str(stats_path))
            stats_exist = True
    
    is_first_run = not (baseline_exists and stats_exist)
    
    if is_first_run:
        logger.info("First run - will create baseline")
    else:
        logger.info("Baseline found - will validate")
    
    context['ti'].xcom_push(key='is_first_run', value=is_first_run)


def decide_validation_path(**context):
    """Decide whether to create baseline or validate."""
    ti = context['ti']
    is_first_run = ti.xcom_pull(task_ids='check_baseline', key='is_first_run')
    
    if is_first_run:
        logger.info("Decision: CREATE BASELINE")
        return 'create_baseline'
    else:
        logger.info("Decision: VALIDATE DATA")
        return 'validate_data'


def create_baseline(**context):
    """Task 2b: Create baseline."""
    from DataPipeline.scripts.RAG.analysis.main import DataQualityAnalyzer
    import shutil
    
    logger.info("TASK: Create Baseline - Started")
    
    ti = context['ti']
    scraped_file = ti.xcom_pull(task_ids='scrape_data', key='scraped_file')
    
    try:
        gcs = GCSManager.from_config()
        
        # Copy to baseline location
        baseline_path = Path("/opt/airflow/DataPipeline/data/RAG/raw_data/baseline/baseline.jsonl")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(scraped_file, baseline_path)
        
        # Generate stats
        analyzer = DataQualityAnalyzer(gcs_manager=gcs, is_baseline=True)
        baseline_df = analyzer.load_jsonl(baseline_path)
        analyzer.generate_baseline_stats(baseline_df)
        
        # Upload stats to GCS
        local_stats = gcp_config.LOCAL_PATHS['validation'] / "baseline_stats.json"
        if local_stats.exists():
            stats_path = gcp_config.get_gcs_path('rag', 'validation') + "baseline_stats.json"
            gcs.upload_file(str(local_stats), stats_path)
        
        context['ti'].xcom_push(key='baseline_created', value=True)
        logger.info("TASK: Create Baseline - Completed")
        
    except Exception as e:
        logger.error(f"Baseline creation failed: {e}", exc_info=True)
        alert_utils.send_failure_alert('create_baseline', str(e), context, ALERT_RECIPIENTS)
        raise


def validate_data(**context):
    """Task 2c: Validate against baseline."""
    from DataPipeline.scripts.RAG.analysis.main import DataQualityAnalyzer
    
    logger.info("TASK: Validate Data - Started")
    
    ti = context['ti']
    scraped_file = ti.xcom_pull(task_ids='scrape_data', key='scraped_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    try:
        gcs = GCSManager.from_config()
        
        # Verify baseline
        baseline_path = Path("/opt/airflow/DataPipeline/data/RAG/raw_data/baseline/baseline.jsonl")
        if not baseline_path.exists():
            raise FileNotFoundError("Baseline missing")
        
        # Validate
        analyzer = DataQualityAnalyzer(gcs_manager=gcs, is_baseline=False)
        new_df = analyzer.load_jsonl(Path(scraped_file))
        results = analyzer.validate_against_baseline(new_df)
        
        # Save report
        report_file = gcp_config.LOCAL_PATHS['validation'] / f"validation_{timestamp}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Upload to GCS
        report_gcs = gcp_config.get_gcs_path('rag', 'validation/reports') + f"validation_{timestamp}.json"
        gcs.upload_file(str(report_file), report_gcs)
        
        logger.info("TASK: Validate Data - Completed")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        alert_utils.send_failure_alert('validate_data', str(e), context, ALERT_RECIPIENTS)
        raise


def chunk_data(**context):
    """Task 3: Chunk data."""
    from DataPipeline.scripts.RAG.chunking import RAGChunker
    
    logger.info("TASK: Chunk Data - Started")
    
    ti = context['ti']
    scraped_file = ti.xcom_pull(task_ids='scrape_data', key='scraped_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    try:
        chunker = RAGChunker()
        chunks = chunker.process_jsonl(Path(scraped_file))
        
        if not chunks:
            raise ValueError("No chunks generated")
        
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Save locally
        chunks_file = gcp_config.LOCAL_PATHS['data'] / "RAG" / "chunks" / f"chunks_{timestamp}.json"
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        chunker.save_chunks(chunks, chunks_file)
        
        context['ti'].xcom_push(key='chunks_file', value=str(chunks_file))
        logger.info("TASK: Chunk Data - Completed")
        
    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        alert_utils.send_failure_alert('chunk_data', str(e), context, ALERT_RECIPIENTS)
        raise


def generate_embeddings(**context):
    """Task 4: Generate embeddings."""
    from DataPipeline.scripts.RAG.embedding import ChunkEmbedder
    
    logger.info("TASK: Generate Embeddings - Started")
    
    ti = context['ti']
    chunks_file = ti.xcom_pull(task_ids='chunk_data', key='chunks_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    try:
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        
        embedder = ChunkEmbedder(model_name=rag_pipeline_config['embedding']['model_name'])
        embedded_chunks = embedder.embed_chunks(chunks)
        
        if not embedded_chunks:
            raise ValueError("No embeddings generated")
        
        logger.info(f"Generated {len(embedded_chunks)} embeddings")
        
        # Save locally
        embeddings_file = gcp_config.LOCAL_PATHS['data'] / "RAG" / "embeddings" / f"embeddings_{timestamp}.json"
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        embedder.save_embeddings(embedded_chunks, embeddings_file)
        
        context['ti'].xcom_push(key='embeddings_file', value=str(embeddings_file))
        logger.info("TASK: Generate Embeddings - Completed")
        
    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        alert_utils.send_failure_alert('generate_embeddings', str(e), context, ALERT_RECIPIENTS)
        raise


def create_index(**context):
    """Task 5: Create FAISS index."""
    from DataPipeline.scripts.RAG.indexing import FAISSIndex
    from DataPipeline.scripts.RAG.embedding import ChunkEmbedder
    import shutil
    
    logger.info("TASK: Create Index - Started")
    
    ti = context['ti']
    embeddings_file = ti.xcom_pull(task_ids='generate_embeddings', key='embeddings_file')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    try:
        # Load embeddings
        embedded_chunks = ChunkEmbedder.load_embeddings(Path(embeddings_file))
        
        if not embedded_chunks:
            raise ValueError("No embeddings loaded")
        
        # Build index
        embedding_dim = embedded_chunks[0].embedding.shape[0]
        faiss_index = FAISSIndex(dimension=embedding_dim, index_type='flat')
        faiss_index.build_index(embedded_chunks)
        
        logger.info(f"Built index: {faiss_index.index.ntotal} vectors")
        
        # Save locally
        index_dir = gcp_config.LOCAL_PATHS['index']
        index_dir.mkdir(parents=True, exist_ok=True)
        
        index_file = index_dir / f"index_{timestamp}.bin"
        data_file = index_dir / f"data_{timestamp}.pkl"
        faiss_index.save_index(index_file, data_file)
        
        # Also save as latest
        shutil.copy(index_file, index_dir / "index_latest.bin")
        shutil.copy(data_file, index_dir / "data_latest.pkl")
        
        logger.info("TASK: Create Index - Completed")
        
    except Exception as e:
        logger.error(f"Index creation failed: {e}", exc_info=True)
        alert_utils.send_failure_alert('create_index', str(e), context, ALERT_RECIPIENTS)
        raise


def track_all_with_dvc(**context):
    """Final task: Track all RAG data with DVC (batch operation)."""
    logger.info("TASK: DVC Tracking - Started")
    
    ti = context['ti']
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    baseline_created = ti.xcom_pull(task_ids='create_baseline', key='baseline_created')
    
    try:
        dvc = DVCManager()
        
        # Collect paths to track
        paths = []
        
        rag_data_dir = Path("/opt/airflow/DataPipeline/data/RAG")
        
        if (rag_data_dir / "raw_data" / "incremental").exists():
            paths.append("data/RAG/raw_data/incremental")
        
        if baseline_created and (rag_data_dir / "raw_data" / "baseline").exists():
            paths.append("data/RAG/raw_data/baseline")
        
        if (rag_data_dir / "chunks").exists():
            paths.append("data/RAG/chunks")
        
        if (rag_data_dir / "embeddings").exists():
            paths.append("data/RAG/embeddings")
        
        if (rag_data_dir / "index").exists():
            paths.append("data/RAG/index")
        
        if not paths:
            logger.warning("No data to track")
            return "No data"
        
        logger.info(f"Tracking {len(paths)} paths with DVC")
        
        # Batch add
        results = dvc.add_batch(paths)
        success = sum(1 for v in results.values() if v)
        
        logger.info(f"Tracked {success}/{len(paths)} paths")
        
        # Push
        if success > 0:
            dvc.push(remote='rag', jobs=4)
        
        # Commit
        dvc.commit_dvc_files(f"RAG run {timestamp}")
        
        logger.info("TASK: DVC Tracking - Completed")
        return "Success"
        
    except Exception as e:
        logger.error(f"DVC tracking failed: {e}", exc_info=True)
        return "Partial"


# DAG DEFINITION
with DAG(
    dag_id='rag_data_pipeline_dvc',
    default_args=default_args,
    description='RAG pipeline with DVC',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['rag', 'dvc'],
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
    
    dvc_track = PythonOperator(
        task_id='track_all_with_dvc',
        python_callable=track_all_with_dvc,
        provide_context=True,
    )
    
    # Flow
    scrape >> check_baseline >> decide
    decide >> [create_baseline_task, validate_task]
    [create_baseline_task, validate_task] >> chunk >> embed >> index >> dvc_track