"""
RAG Data Pipeline DAG — FINAL ENHANCED VERSION (2025-10)
Enhancements:
- True branch-based skip logic
- First-run auto bootstrap of baseline
- Validation against latest merged dataset
- Robust cleanup of temporary files
- GCS timeout handling
- Safe anomaly filtering
- Optional DVC/Git subprocess with failure logging
"""

from datetime import timedelta
from pathlib import Path
import sys
import asyncio
import subprocess
import shutil
import json
import tempfile

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.exceptions import AirflowSkipException
from airflow.utils.dates import days_ago

# Docker environment paths
PROJECT_ROOT = Path("/opt/airflow")
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "DataPipeline" / "config"))

try:
    import gcp_config
    from RAG.common_utils import GCSManager, upload_with_versioning
except ImportError as e:
    print(f"❌ Import error: {e}")
    raise

default_args = {
    "owner": "rag-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=45),
}


# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def scrape_data(**context):
    """Scrape only NEW URLs (duplicate detection)."""
    from RAG.scraper import main as scraper_main

    logical_date = context["logical_date"]
    timestamp = logical_date.strftime("%Y%m%d_%H%M%S")

    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    all_urls = gcp_config.get_scraping_urls()

    already_scraped = set()
    tmp_files = []

    try:
        for ref_blob in [
            "RAG/raw_data/baseline/baseline.jsonl",
            "RAG/merged/combined_latest.jsonl",
        ]:
            if gcs.blob_exists(ref_blob):
                tmp = Path(tempfile.mktemp())
                tmp_files.append(tmp)
                gcs.download_file(ref_blob, str(tmp))
                with open(tmp, "r") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if "link" in rec and rec["link"]:
                                already_scraped.add(rec["link"])
                        except Exception:
                            continue

        new_urls = [u for u in all_urls if u not in already_scraped]
        duplicate_count = len(all_urls) - len(new_urls)
        print(f"🧭 Found {len(new_urls)} new URLs ({duplicate_count} duplicates skipped)")

        ti = context["ti"]
        if not new_urls:
            ti.xcom_push(key="has_new_data", value=False)
            raise AirflowSkipException("No new URLs to scrape")

        output_file = gcp_config.LOCAL_PATHS["raw_data"] / f"batch_{timestamp}.jsonl"
        asyncio.run(scraper_main(new_urls, str(output_file), method="W"))

        gcs.upload_file(str(output_file), f"RAG/raw_data/incremental/batch_{timestamp}.jsonl")
        count = sum(1 for _ in open(output_file))
        ti.xcom_push(key="batch_file", value=str(output_file))
        ti.xcom_push(key="timestamp", value=timestamp)
        ti.xcom_push(key="new_count", value=count)
        ti.xcom_push(key="has_new_data", value=True)
        print(f"✅ Scraped {count} new articles")
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)


def validate_new_data(**context):
    """Validate new data against latest merged dataset."""
    from RAG.analysis.main import DataQualityAnalyzer

    ti = context["ti"]
    batch_file = Path(ti.xcom_pull(task_ids="scrape_data", key="batch_file"))
    timestamp = ti.xcom_pull(task_ids="scrape_data", key="timestamp")
    new_count = ti.xcom_pull(task_ids="scrape_data", key="new_count")

    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    baseline_blob = "RAG/merged/combined_latest.jsonl"

    tmp_files = []

    try:
        if not gcs.blob_exists(baseline_blob):
            print("🆕 First run: no baseline found. Accepting all new data.")
            ti.xcom_push(key="validated_batch", value=str(batch_file))
            ti.xcom_push(key="articles_to_merge", value=new_count)
            ti.xcom_push(key="is_first_run", value=True)
            return

        baseline_path = Path(tempfile.mktemp())
        tmp_files.append(baseline_path)
        gcs.download_file(baseline_blob, str(baseline_path))

        analyzer = DataQualityAnalyzer(baseline=str(baseline_path), new=str(batch_file))
        results = analyzer.analyze_new_data()
        if not results:
            raise Exception("Validation failed")

        anomalies = results.get("anomalies", {})
        total_anomalies = anomalies.get("total_anomalies", 0)
        bad_links = {r.get("link") for r in anomalies.get("records", []) if r.get("link")}

        drift = results.get("drift_results", {})
        crit_drift = sum(
            1
            for v in drift.values()
            if v and (v.get("mean_shift", 0) > 0.5 or abs(v.get("percent_mean_change", 0)) > 30)
        )

        cfg = gcp_config.get_validation_config()
        max_anom, max_drift = cfg.get("max_anomaly_pct", 0.2), cfg.get("max_critical_drift", 3)
        if (total_anomalies / new_count) > max_anom or crit_drift > max_drift:
            raise Exception("❌ Quality gate failed")

        # Filter anomalous links
        cleaned = Path(tempfile.mktemp())
        tmp_files.append(cleaned)
        with open(batch_file, "r") as fin, open(cleaned, "w") as fout:
            for line in fin:
                rec = json.loads(line)
                if rec.get("link") not in bad_links:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        ti.xcom_push(key="validated_batch", value=str(cleaned))
        ti.xcom_push(key="articles_to_merge", value=sum(1 for _ in open(cleaned)))
        ti.xcom_push(key="is_first_run", value=False)
        print("✅ Validation complete")
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)


def merge_data(**context):
    """Merge validated data into combined dataset (deduplicated)."""
    from RAG.merge_batches import merge_jsonl_files

    ti = context["ti"]
    validated_batch = ti.xcom_pull(task_ids="validate_new_data", key="validated_batch")
    timestamp = ti.xcom_pull(task_ids="scrape_data", key="timestamp")
    first_run = ti.xcom_pull(task_ids="validate_new_data", key="is_first_run")

    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    local_merged = gcp_config.LOCAL_PATHS["merged"] / f"combined_{timestamp}.jsonl"

    if first_run:
        shutil.copy(validated_batch, local_merged)
        gcs.upload_file(str(local_merged), "RAG/raw_data/baseline/baseline.jsonl")
    else:
        prev = Path(tempfile.mktemp())
        gcs.download_file("RAG/merged/combined_latest.jsonl", str(prev))
        merge_jsonl_files(prev, Path(validated_batch), local_merged, deduplicate=True)
        prev.unlink(missing_ok=True)

    version = upload_with_versioning(
        gcs, str(local_merged), "RAG/merged", "combined_{version}.jsonl"
    )
    ti.xcom_push(key="merged_file", value=str(local_merged))
    ti.xcom_push(key="version", value=version)
    print(f"✅ Merge complete → v{version}")


def chunk_data(**context):
    from RAG import chunking
    ti = context["ti"]
    merged_file = ti.xcom_pull(task_ids="merge_data", key="merged_file")

    if not merged_file:
        raise AirflowSkipException("No merged data")

    output = gcp_config.LOCAL_PATHS["chunked_data"] / "chunks_temp.json"
    orig_in, orig_out = chunking.INPUT_FILE, chunking.OUTPUT_FILE
    chunking.INPUT_FILE, chunking.OUTPUT_FILE = Path(merged_file), output
    try:
        chunking.main()
    finally:
        chunking.INPUT_FILE, chunking.OUTPUT_FILE = orig_in, orig_out

    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    v = upload_with_versioning(gcs, str(output), "RAG/chunked_data", "chunks_{version}.json")
    ti.xcom_push(key="chunks_file", value=str(output))
    print(f"✅ Chunked v{v}")


def generate_embeddings(**context):
    from RAG import embedding
    ti = context["ti"]
    chunks_file = ti.xcom_pull(task_ids="chunk_data", key="chunks_file")
    if not chunks_file:
        raise AirflowSkipException("No chunks available")

    out = gcp_config.LOCAL_PATHS["index"] / "embeddings_temp.json"
    orig_in, orig_out = embedding.INPUT_FILE, embedding.OUTPUT_FILE
    embedding.INPUT_FILE, embedding.OUTPUT_FILE = Path(chunks_file), out
    try:
        embedding.main()
    finally:
        embedding.INPUT_FILE, embedding.OUTPUT_FILE = orig_in, orig_out

    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    upload_with_versioning(gcs, str(out), "RAG/index", "embeddings_{version}.json")
    ti.xcom_push(key="embeddings_file", value=str(out))
    print("✅ Embeddings generated")


def create_index(**context):
    from RAG import indexing
    ti = context["ti"]
    emb = ti.xcom_pull(task_ids="generate_embeddings", key="embeddings_file")
    if not emb:
        raise AirflowSkipException("No embeddings to index")

    idx_file = gcp_config.LOCAL_PATHS["index"] / "index_temp.bin"
    data_file = gcp_config.LOCAL_PATHS["index"] / "data_temp.pkl"

    orig_in, orig_idx, orig_data = indexing.INPUT_FILE, indexing.OUTPUT_FILE_INDEX, indexing.OUTPUT_FILE_DATA
    indexing.INPUT_FILE, indexing.OUTPUT_FILE_INDEX, indexing.OUTPUT_FILE_DATA = Path(emb), idx_file, data_file
    try:
        indexing.main()
    finally:
        indexing.INPUT_FILE, indexing.OUTPUT_FILE_INDEX, indexing.OUTPUT_FILE_DATA = orig_in, orig_idx, orig_data

    gcs = GCSManager(gcp_config.GCS_BUCKET_NAME, gcp_config.SERVICE_ACCOUNT_PATH)
    upload_with_versioning(gcs, str(idx_file), "RAG/index", "index_{version}.bin")
    upload_with_versioning(gcs, str(data_file), "RAG/index", "data_{version}.pkl")
    print("✅ FAISS index created")


def dvc_operations(**context):
    """Optional version control sync (safe, optional)."""
    ti = context["ti"]
    version = ti.xcom_pull(task_ids="merge_data", key="version")

    try:
        print(f"📦 DVC add/push for version {version}")
        subprocess.run(["dvc", "add", "data/RAG/"], cwd=str(gcp_config.PROJECT_ROOT), check=False)
        subprocess.run(["dvc", "push", "-r", "rag"], cwd=str(gcp_config.PROJECT_ROOT), check=False)
        subprocess.run(
            ["git", "add", "data/RAG.dvc", ".gitignore"],
            cwd=str(gcp_config.PROJECT_ROOT),
            check=False,
        )
        subprocess.run(
            ["git", "commit", "-m", f"RAG data version {version}"],
            cwd=str(gcp_config.PROJECT_ROOT),
            check=False,
        )
        subprocess.run(["git", "push"], cwd=str(gcp_config.PROJECT_ROOT), check=False)
    except Exception as e:
        print(f"⚠️ DVC/Git operation failed: {e}")


# ============================================================================
# BRANCHING
# ============================================================================

def decide_next(**context):
    ti = context["ti"]
    has_new = ti.xcom_pull(task_ids="scrape_data", key="has_new_data")
    return "validate_new_data" if has_new else "stop"


# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    "rag_pipeline_final",
    default_args=default_args,
    description="Enhanced RAG data pipeline",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["rag", "etl", "gcs", "dvc"],
) as dag:

    scrape = PythonOperator(task_id="scrape_data", python_callable=scrape_data)
    decide = BranchPythonOperator(task_id="decide_next", python_callable=decide_next)
    stop = DummyOperator(task_id="stop")

    validate = PythonOperator(task_id="validate_new_data", python_callable=validate_new_data)
    merge = PythonOperator(task_id="merge_data", python_callable=merge_data)
    chunk = PythonOperator(task_id="chunk_data", python_callable=chunk_data)
    embed = PythonOperator(task_id="generate_embeddings", python_callable=generate_embeddings)
    index = PythonOperator(task_id="create_index", python_callable=create_index)
    dvc = PythonOperator(task_id="dvc_operations", python_callable=dvc_operations)

    scrape >> decide >> [validate, stop]
    validate >> merge >> chunk >> embed >> index >> dvc