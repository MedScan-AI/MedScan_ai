"""
RAG Data Pipeline DAG - With simple email alerts and metadata-only XCom

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
import logging

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
    from RAG.alert_utils import send_failure_alert, send_threshold_alert
except ImportError as e:
    print(f"Import error: {e}")
    raise

# Get alert configuration
ALERT_CONFIG = gcp_config.get_alert_config()
ALERT_RECIPIENTS = ALERT_CONFIG.get('email_recipients', [])
ALERT_THRESHOLDS = gcp_config.get_alert_thresholds()

logger = logging.getLogger(__name__)

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
    """Task 1: Scrape ALL URLs from GCS config."""
    from RAG.scraper import main as scraper_main
    
    logger.info("="*80)
    logger.info("TASK: Scrape Data - Started")
    logger.info("="*80)
    
    execution_date = context['execution_date']
    timestamp = execution_date.strftime('%Y%m%d_%H%M%S')
    
    try:
        # Get GCS manager
        gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
        
        # Read URLs from GCS
        urls_file_path = gcp_config.get_urls_file_path()
        urls = read_urls_from_gcs(gcs, urls_file_path)
        
        logger.info(f"Total URLs to scrape: {len(urls)}")
        
        # Scrape all URLs with timeout handling
        output_file = gcp_config.LOCAL_PATHS['temp'] / f"scraped_{timestamp}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            asyncio.run(
                asyncio.wait_for(
                    scraper_main(urls, str(output_file), method='W'),
                    timeout=1800  # 30 minute max for all URLs
                )
            )
        except asyncio.TimeoutError:
            logger.warning("Scraping timeout - processing partial results")
        
        if not output_file.exists():
            error_msg = f"Scraper did not create output file: {output_file}"
            logger.error(error_msg)
            send_failure_alert('scrape_data', error_msg, context, ALERT_RECIPIENTS)
            raise FileNotFoundError(error_msg)
        
        # Analyze results
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
                        logger.warning(f"Error in record: {record.get('link', 'unknown')}")
                    else:
                        success_records += 1
                except json.JSONDecodeError:
                    error_records += 1
                    logger.error(f"Invalid JSON at line {total_records}")
        
        # Calculate success rate
        success_rate = success_records / len(urls) if len(urls) > 0 else 0
        
        logger.info(f"Scraping Results:")
        logger.info(f"  Total URLs: {len(urls)}")
        logger.info(f"  Successful: {success_records} ({success_rate*100:.1f}%)")
        logger.info(f"  Failed: {error_records}")
        
        # Upload to GCS
        gcs_path = f"RAG/raw_data/incremental/scraped_{timestamp}.jsonl"
        logger.info(f"Uploading to GCS: {gcs_path}")
        upload_success = gcs.upload_file(str(output_file), gcs_path)
        
        if not upload_success:
            error_msg = "Failed to upload scraped data to GCS"
            logger.error(error_msg)
            send_failure_alert('scrape_data', error_msg, context, ALERT_RECIPIENTS)
            raise Exception(error_msg)
        
        # Check threshold
        min_success_rate = ALERT_THRESHOLDS.get('scraping_min_success_rate', 0.7)
        if success_rate < min_success_rate:
            send_threshold_alert(
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
        
        # Push ONLY metadata to XCom
        context['ti'].xcom_push(key='gcs_path', value=gcs_path)
        context['ti'].xcom_push(key='timestamp', value=timestamp)
        context['ti'].xcom_push(key='success_count', value=success_records)
        context['ti'].xcom_push(key='error_count', value=error_records)
        context['ti'].xcom_push(key='success_rate', value=success_rate)
        
        logger.info("="*80)
        logger.info("TASK: Scrape Data - Completed")
        logger.info("="*80)
        
        # Cleanup local file
        output_file.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Scraping task failed: {e}", exc_info=True)
        send_failure_alert('scrape_data', str(e), context, ALERT_RECIPIENTS)
        raise


def check_baseline_exists(**context):
    """Task 2a: Check if baseline exists."""
    logger.info("Checking for baseline data...")
    
    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    
    baseline_exists = gcs.blob_exists("RAG/raw_data/baseline/baseline.jsonl")
    stats_exist = gcs.blob_exists("RAG/validation/baseline_stats.json")
    
    is_first_run = not (baseline_exists and stats_exist)
    
    if is_first_run:
        if baseline_exists and not stats_exist:
            logger.warning("Baseline exists but stats missing!")
        else:
            logger.info("First run detected - will create baseline")
    else:
        logger.info("Baseline found - will validate new data")
    
    # Push to XCom
    context['ti'].xcom_push(key='baseline_exists', value=baseline_exists)
    context['ti'].xcom_push(key='stats_exist', value=stats_exist)
    context['ti'].xcom_push(key='is_first_run', value=is_first_run)


def decide_validation_path(**context):
    """Decide whether to create baseline or validate."""
    ti = context['ti']
    is_first_run = ti.xcom_pull(task_ids='check_baseline', key='is_first_run')
    
    logger.info(f"Is first run: {is_first_run}")
    
    if is_first_run:
        logger.info("Decision: CREATE BASELINE")
        return 'create_baseline'
    else:
        logger.info("Decision: VALIDATE DATA")
        return 'validate_data'


def create_baseline(**context):
    """Task 2b: Create baseline (first run only)."""
    from RAG.analysis.main import DataQualityAnalyzer
    
    logger.info("="*80)
    logger.info("TASK: Create Baseline - Started")
    logger.info("="*80)
    
    ti = context['ti']
    gcs_path = ti.xcom_pull(task_ids='scrape_data', key='gcs_path')
    success_count = ti.xcom_pull(task_ids='scrape_data', key='success_count')
    
    logger.info(f"GCS Path: {gcs_path}")
    logger.info(f"Success Count: {success_count}")
    
    try:
        gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
        
        # Download from GCS
        local_file = gcp_config.LOCAL_PATHS['temp'] / f"baseline_temp.jsonl"
        logger.info(f"Downloading from GCS: {gcs_path}")
        download_success = gcs.download_file(gcs_path, str(local_file))
        
        if not download_success:
            error_msg = f"Failed to download from GCS: {gcs_path}"
            logger.error(error_msg)
            send_failure_alert('create_baseline', error_msg, context, ALERT_RECIPIENTS)
            raise FileNotFoundError(error_msg)
        
        # Upload as baseline
        logger.info("Uploading as baseline...")
        gcs.upload_file(str(local_file), "RAG/raw_data/baseline/baseline.jsonl")
        
        # Generate baseline stats
        logger.info("Generating baseline statistics...")
        analyzer = DataQualityAnalyzer(gcs_manager=gcs, is_baseline=True)
        baseline_df = analyzer.load_jsonl(Path(local_file))
        baseline_data = analyzer.generate_baseline_stats(baseline_df)
        
        # Verify upload
        stats_uploaded = gcs.blob_exists("RAG/validation/baseline_stats.json")
        
        if not stats_uploaded:
            error_msg = "Baseline stats were not uploaded to GCS"
            logger.error(error_msg)
            send_failure_alert('create_baseline', error_msg, context, ALERT_RECIPIENTS)
            raise Exception(error_msg)
        
        logger.info("Baseline created successfully")
        
        # Push metadata to XCom
        context['ti'].xcom_push(key='validation_passed', value=True)
        context['ti'].xcom_push(key='gcs_path', value=gcs_path)
        context['ti'].xcom_push(key='baseline_created', value=True)
        
        logger.info("="*80)
        logger.info("TASK: Create Baseline - Completed")
        logger.info("="*80)
        
        # Cleanup
        local_file.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Baseline creation failed: {e}", exc_info=True)
        send_failure_alert('create_baseline', str(e), context, ALERT_RECIPIENTS)
        raise


def validate_data(**context):
    """Task 2c: Validate against baseline."""
    from RAG.analysis.main import DataQualityAnalyzer
    
    logger.info("="*80)
    logger.info("TASK: Validate Data - Started")
    logger.info("="*80)
    
    ti = context['ti']
    gcs_path = ti.xcom_pull(task_ids='scrape_data', key='gcs_path')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    scraped_count = ti.xcom_pull(task_ids='scrape_data', key='success_count')
    
    logger.info(f"GCS Path: {gcs_path}")
    logger.info(f"Records to validate: {scraped_count}")
    
    try:
        gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
        
        # Double-check that baseline exists
        baseline_exists = gcs.blob_exists("RAG/raw_data/baseline/baseline.jsonl")
        stats_exist = gcs.blob_exists("RAG/validation/baseline_stats.json")
        
        if not baseline_exists or not stats_exist:
            error_msg = "Cannot validate: baseline data or stats missing"
            logger.error(error_msg)
            send_failure_alert('validate_data', error_msg, context, ALERT_RECIPIENTS)
            raise FileNotFoundError(f"{error_msg}. Run create_baseline first.")
        
        # Download from GCS
        local_file = gcp_config.LOCAL_PATHS['temp'] / f"validate_temp_{timestamp}.jsonl"
        logger.info(f"Downloading from GCS: {gcs_path}")
        download_success = gcs.download_file(gcs_path, str(local_file))
        
        if not download_success:
            error_msg = f"Failed to download from GCS: {gcs_path}"
            logger.error(error_msg)
            send_failure_alert('validate_data', error_msg, context, ALERT_RECIPIENTS)
            raise FileNotFoundError(error_msg)
        
        # Initialize analyzer
        logger.info("Initializing data quality analyzer...")
        analyzer = DataQualityAnalyzer(gcs_manager=gcs, is_baseline=False)
        
        # Load new data
        new_df = analyzer.load_jsonl(Path(local_file))
        
        # Validate against baseline
        logger.info("Validating against baseline...")
        results = analyzer.validate_against_baseline(new_df)
        
        # Save validation report
        report_file = gcp_config.LOCAL_PATHS['validation'] / f"validation_{timestamp}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Upload report to GCS
        report_gcs_path = f"RAG/validation/reports/validation_{timestamp}.json"
        gcs.upload_file(str(report_file), report_gcs_path)
        
        # Extract key metrics
        anomaly_pct = results['anomalies']['percentage']
        completeness_issues = sum(len(v) for v in results['completeness'].values())
        drift_count = len([f for f, d in results['drift'].items() if d and d.get('has_drift')])
        
        logger.info("Validation Results:")
        logger.info(f"  Total records: {results['total_records']}")
        logger.info(f"  Anomalies: {results['anomalies']['total']} ({anomaly_pct:.1f}%)")
        logger.info(f"  Completeness issues: {completeness_issues}")
        logger.info(f"  Features with drift: {drift_count}")
        
        # Check thresholds
        max_anomaly_pct = ALERT_THRESHOLDS.get('validation_max_anomaly_pct', 25.0)
        max_drift_features = ALERT_THRESHOLDS.get('validation_max_drift_features', 3)
        
        # Send alerts if thresholds exceeded
        if anomaly_pct > max_anomaly_pct:
            send_threshold_alert(
                task_name='validate_data',
                threshold_name='Anomaly Percentage',
                actual_value=anomaly_pct,
                threshold_value=max_anomaly_pct,
                context=context,
                recipients=ALERT_RECIPIENTS,
                additional_info={
                    'Anomaly Count': results['anomalies']['total'],
                    'Total Records': results['total_records'],
                    'Report': report_gcs_path
                }
            )
        
        if drift_count > max_drift_features:
            drifted_features = [f for f, d in results['drift'].items() if d and d.get('has_drift')]
            send_threshold_alert(
                task_name='validate_data',
                threshold_name='Drift Features',
                actual_value=drift_count,
                threshold_value=max_drift_features,
                context=context,
                recipients=ALERT_RECIPIENTS,
                additional_info={
                    'Drifted Features': ', '.join(drifted_features),
                    'Report': report_gcs_path
                }
            )
        
        # Push metadata to XCom
        context['ti'].xcom_push(key='validation_passed', value=True)
        context['ti'].xcom_push(key='gcs_path', value=gcs_path)
        context['ti'].xcom_push(key='validation_report_gcs_path', value=report_gcs_path)
        context['ti'].xcom_push(key='anomaly_pct', value=anomaly_pct)
        context['ti'].xcom_push(key='drift_count', value=drift_count)
        
        logger.info("="*80)
        logger.info("TASK: Validate Data - Completed")
        logger.info("="*80)
        
        # Cleanup
        local_file.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        send_failure_alert('validate_data', str(e), context, ALERT_RECIPIENTS)
        raise


def chunk_data(**context):
    """Task 3: Chunk data."""
    from RAG.chunking import RAGChunker
    
    logger.info("="*80)
    logger.info("TASK: Chunk Data - Started")
    logger.info("="*80)
    
    ti = context['ti']
    
    # Get GCS path from either branch
    gcs_path = ti.xcom_pull(task_ids='validate_data', key='gcs_path')
    if not gcs_path:
        gcs_path = ti.xcom_pull(task_ids='create_baseline', key='gcs_path')
    
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    logger.info(f"GCS Path: {gcs_path}")
    
    try:
        gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
        
        # Download from GCS
        local_file = gcp_config.LOCAL_PATHS['temp'] / f"chunk_input_{timestamp}.jsonl"
        logger.info(f"Downloading from GCS: {gcs_path}")
        download_success = gcs.download_file(gcs_path, str(local_file))
        
        if not download_success or not local_file.exists():
            error_msg = f"Failed to download input file from GCS: {gcs_path}"
            logger.error(error_msg)
            send_failure_alert('chunk_data', error_msg, context, ALERT_RECIPIENTS)
            raise FileNotFoundError(error_msg)
        
        # Initialize chunker
        logger.info("Processing documents...")
        chunker = RAGChunker()
        chunks = chunker.process_jsonl(Path(local_file))
        
        if not chunks:
            error_msg = "No chunks generated from input data!"
            logger.error(error_msg)
            send_failure_alert('chunk_data', error_msg, context, ALERT_RECIPIENTS)
            raise ValueError(error_msg)
        
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Save and upload to GCS
        chunks_file = gcp_config.LOCAL_PATHS['temp'] / f"chunks_{timestamp}.json"
        chunker.save_chunks(chunks, chunks_file)
        
        chunks_gcs_path = f"RAG/chunks/chunks_{timestamp}.json"
        logger.info(f"Uploading chunks to GCS: {chunks_gcs_path}")
        gcs.upload_file(str(chunks_file), chunks_gcs_path)
        
        # Push metadata to XCom
        context['ti'].xcom_push(key='chunks_gcs_path', value=chunks_gcs_path)
        context['ti'].xcom_push(key='chunk_count', value=len(chunks))
        
        logger.info("="*80)
        logger.info("TASK: Chunk Data - Completed")
        logger.info("="*80)
        
        # Cleanup
        local_file.unlink(missing_ok=True)
        chunks_file.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        send_failure_alert('chunk_data', str(e), context, ALERT_RECIPIENTS)
        raise


def generate_embeddings(**context):
    """Task 4: Generate embeddings."""
    from RAG.embedding import ChunkEmbedder
    import json
    
    logger.info("="*80)
    logger.info("TASK: Generate Embeddings - Started")
    logger.info("="*80)
    
    ti = context['ti']
    chunks_gcs_path = ti.xcom_pull(task_ids='chunk_data', key='chunks_gcs_path')
    chunk_count = ti.xcom_pull(task_ids='chunk_data', key='chunk_count')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    logger.info(f"Chunks GCS Path: {chunks_gcs_path}")
    logger.info(f"Chunk Count: {chunk_count}")
    
    try:
        gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
        
        # Download chunks from GCS
        chunks_file = gcp_config.LOCAL_PATHS['temp'] / f"chunks_{timestamp}.json"
        logger.info(f"Downloading chunks from GCS: {chunks_gcs_path}")
        download_success = gcs.download_file(chunks_gcs_path, str(chunks_file))
        
        if not download_success or not chunks_file.exists():
            error_msg = f"Failed to download chunks from GCS: {chunks_gcs_path}"
            logger.error(error_msg)
            send_failure_alert('generate_embeddings', error_msg, context, ALERT_RECIPIENTS)
            raise FileNotFoundError(error_msg)
        
        # Load chunks
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedder = ChunkEmbedder(model_name=gcp_config.get_embedding_model())
        embedded_chunks = embedder.embed_chunks(chunks)
        
        if len(embedded_chunks) == 0:
            error_msg = "Failed to generate any embeddings!"
            logger.error(error_msg)
            send_failure_alert('generate_embeddings', error_msg, context, ALERT_RECIPIENTS)
            raise ValueError(error_msg)
        
        # Calculate success rate
        success_rate = len(embedded_chunks) / len(chunks) if len(chunks) > 0 else 0
        
        logger.info(f"Embedding Results:")
        logger.info(f"  Total chunks: {len(chunks)}")
        logger.info(f"  Successful embeddings: {len(embedded_chunks)} ({success_rate*100:.1f}%)")
        logger.info(f"  Failed: {len(chunks) - len(embedded_chunks)}")
        
        # Check threshold
        min_success_rate = ALERT_THRESHOLDS.get('embedding_min_success_rate', 0.95)
        if success_rate < min_success_rate:
            send_threshold_alert(
                task_name='generate_embeddings',
                threshold_name='Embedding Success Rate',
                actual_value=success_rate * 100,
                threshold_value=min_success_rate * 100,
                context=context,
                recipients=ALERT_RECIPIENTS,
                additional_info={
                    'Successful Embeddings': len(embedded_chunks),
                    'Failed Embeddings': len(chunks) - len(embedded_chunks),
                    'Total Chunks': len(chunks)
                }
            )
        
        # Save and upload embeddings
        embeddings_file = gcp_config.LOCAL_PATHS['temp'] / f"embeddings_{timestamp}.json"
        embedder.save_embeddings(embedded_chunks, embeddings_file)
        
        embeddings_gcs_path = f"RAG/embeddings/embeddings_{timestamp}.json"
        logger.info(f"Uploading embeddings to GCS: {embeddings_gcs_path}")
        gcs.upload_file(str(embeddings_file), embeddings_gcs_path)
        
        # Push metadata to XCom
        context['ti'].xcom_push(key='embeddings_gcs_path', value=embeddings_gcs_path)
        context['ti'].xcom_push(key='embedding_count', value=len(embedded_chunks))
        
        logger.info("="*80)
        logger.info("TASK: Generate Embeddings - Completed")
        logger.info("="*80)
        
        # Cleanup
        chunks_file.unlink(missing_ok=True)
        embeddings_file.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        send_failure_alert('generate_embeddings', str(e), context, ALERT_RECIPIENTS)
        raise


def create_index(**context):
    """Task 5: Create and save FAISS index."""
    from RAG.indexing import FAISSIndex
    from RAG.embedding import ChunkEmbedder
    
    logger.info("="*80)
    logger.info("TASK: Create Index - Started")
    logger.info("="*80)
    
    ti = context['ti']
    embeddings_gcs_path = ti.xcom_pull(task_ids='generate_embeddings', key='embeddings_gcs_path')
    embedding_count = ti.xcom_pull(task_ids='generate_embeddings', key='embedding_count')
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    logger.info(f"Embeddings GCS Path: {embeddings_gcs_path}")
    logger.info(f"Embedding Count: {embedding_count}")
    
    try:
        gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
        
        # Download embeddings from GCS
        embeddings_file = gcp_config.LOCAL_PATHS['temp'] / f"embeddings_{timestamp}.json"
        logger.info(f"Downloading embeddings from GCS: {embeddings_gcs_path}")
        download_success = gcs.download_file(embeddings_gcs_path, str(embeddings_file))
        
        if not download_success or not embeddings_file.exists():
            error_msg = f"Failed to download embeddings from GCS: {embeddings_gcs_path}"
            logger.error(error_msg)
            send_failure_alert('create_index', error_msg, context, ALERT_RECIPIENTS)
            raise FileNotFoundError(error_msg)
        
        # Load embeddings
        logger.info("Loading embeddings...")
        embedded_chunks = ChunkEmbedder.load_embeddings(Path(embeddings_file))
        
        if not embedded_chunks or len(embedded_chunks) == 0:
            error_msg = "No embedded chunks loaded!"
            logger.error(error_msg)
            send_failure_alert('create_index', error_msg, context, ALERT_RECIPIENTS)
            raise ValueError(error_msg)
        
        logger.info(f"Loaded {len(embedded_chunks)} embeddings")
        
        # Check minimum threshold
        min_vectors = ALERT_THRESHOLDS.get('indexing_min_vectors', 100)
        if len(embedded_chunks) < min_vectors:
            send_threshold_alert(
                task_name='create_index',
                threshold_name='Minimum Vector Count',
                actual_value=len(embedded_chunks),
                threshold_value=min_vectors,
                context=context,
                recipients=ALERT_RECIPIENTS,
                additional_info={'Note': 'Index will still be created'}
            )
        
        # Create index
        embedding_dim = embedded_chunks[0].embedding.shape[0]
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        logger.info("Building FAISS index...")
        faiss_index = FAISSIndex(dimension=embedding_dim, index_type='flat')
        faiss_index.build_index(embedded_chunks)
        
        logger.info(f"Built FAISS index: {faiss_index.index.ntotal} vectors")
        
        # Define output paths
        index_dir = gcp_config.LOCAL_PATHS['index']
        index_dir.mkdir(parents=True, exist_ok=True)
        
        index_timestamped = index_dir / f"index_{timestamp}.bin"
        data_timestamped = index_dir / f"data_{timestamp}.pkl"
        index_latest = index_dir / "index_latest.bin"
        data_latest = index_dir / "data_latest.pkl"
        
        # Save locally
        logger.info("Saving index locally...")
        faiss_index.save_index(index_timestamped, data_timestamped)
        shutil.copy(index_timestamped, index_latest)
        shutil.copy(data_timestamped, data_latest)
        
        # Upload to GCS
        logger.info("Uploading index to GCS...")
        gcs.upload_file(str(index_timestamped), f"RAG/index/index_{timestamp}.bin")
        gcs.upload_file(str(data_timestamped), f"RAG/index/data_{timestamp}.pkl")
        gcs.upload_file(str(index_latest), "RAG/index/index_latest.bin")
        gcs.upload_file(str(data_latest), "RAG/index/data_latest.pkl")
        
        logger.info("Index uploaded successfully")
        
        # Push metadata to XCom
        context['ti'].xcom_push(key='index_timestamp', value=timestamp)
        context['ti'].xcom_push(key='vector_count', value=faiss_index.index.ntotal)
        
        logger.info("="*80)
        logger.info("TASK: Create Index - Completed")
        logger.info("="*80)
        
        # Cleanup
        embeddings_file.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Index creation failed: {e}", exc_info=True)
        send_failure_alert('create_index', str(e), context, ALERT_RECIPIENTS)
        raise


def dvc_operations(**context):
    """Task 6: DVC track incremental + index."""
    logger.info("="*80)
    logger.info("TASK: DVC Operations - Started")
    logger.info("="*80)
    
    ti = context['ti']
    timestamp = ti.xcom_pull(task_ids='scrape_data', key='timestamp')
    
    try:
        # Track specific paths only
        paths_to_track = [
            "data/RAG/raw_data/incremental/",
            "data/RAG/index/",
        ]
        
        logger.info("Adding paths to DVC...")
        for path in paths_to_track:
            result = subprocess.run(
                ['dvc', 'add', path],
                cwd=str(gcp_config.PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning(f"DVC add failed for {path}: {result.stderr}")
        
        # Push to RAG remote
        logger.info("Pushing to DVC remote...")
        result = subprocess.run(
            ['dvc', 'push', '-r', 'rag'],
            cwd=str(gcp_config.PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.warning(f"DVC push failed: {result.stderr}")
        
        # Git commit
        logger.info("Committing DVC metadata to git...")
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
        
        logger.info("="*80)
        logger.info("TASK: DVC Operations - Completed")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"DVC operations failed: {e}", exc_info=True)
        # DVC failures are not critical - just log warning
        logger.warning("DVC operations failed but pipeline will continue")


# Define the DAG
with DAG(
    dag_id='rag_data_pipeline',
    default_args=default_args,
    description='RAG pipeline with simple email alerts',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['rag', 'split-validation', 'gcs-aware', 'email-alerts'],
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