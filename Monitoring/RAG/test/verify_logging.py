#!/usr/bin/env python3
"""
Verify that all metrics are logged correctly in Cloud Logging and reports are saved to GCS.
"""
import argparse
import subprocess
import json
from google.cloud import storage, logging as cloud_logging
from datetime import datetime, timedelta


def check_cloud_logging(project_id: str, hours: int = 1):
    """Check Cloud Logging for prediction logs with all required metrics."""
    print("\n" + "="*70)
    print("CHECKING CLOUD LOGGING")
    print("="*70)
    
    logging_client = cloud_logging.Client(project=project_id)
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    
    filter_str = f"""
    resource.type="cloud_run_revision"
    resource.labels.service_name="rag-service"
    timestamp >= "{start_time.isoformat()}Z"
    jsonPayload.prediction_result:*
    """
    
    logs = []
    required_fields = [
        'query', 'latency', 'success',
        'composite_score', 'hallucination_score', 'avg_retrieval_score',
        'retrieved_doc_indices', 'index_size'
    ]
    
    print(f"Fetching logs from last {hours} hour(s)...")
    for entry in logging_client.list_entries(filter_=filter_str, max_results=100):
        try:
            payload = entry.payload
            if 'prediction_result' in payload:
                pred_result = payload.get('prediction_result', {})
                logs.append(pred_result)
        except Exception as e:
            print(f"Error parsing log: {e}")
    
    print(f"\nFound {len(logs)} prediction logs")
    
    if not logs:
        print("⚠️  No logs found. Make sure service is deployed and receiving requests.")
        return False
    
    # Check for required fields
    print("\nChecking for required fields in logs:")
    field_counts = {field: 0 for field in required_fields}
    
    for log in logs:
        for field in required_fields:
            if field in log and log[field] is not None:
                field_counts[field] += 1
    
    all_present = True
    for field, count in field_counts.items():
        percentage = (count / len(logs)) * 100 if logs else 0
        status = "✓" if count == len(logs) else "⚠️"
        print(f"  {status} {field}: {count}/{len(logs)} ({percentage:.1f}%)")
        if count == 0:
            all_present = False
    
    # Sample log
    if logs:
        print("\nSample log entry:")
        sample = logs[0]
        print(json.dumps({k: v for k, v in sample.items() if k in required_fields}, indent=2))
    
    return all_present


def check_gcs_reports(bucket_name: str):
    """Check GCS for monitoring reports."""
    print("\n" + "="*70)
    print("CHECKING GCS MONITORING REPORTS")
    print("="*70)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    reports_path = "RAG/monitoring/reports/"
    blobs = list(bucket.list_blobs(prefix=reports_path))
    
    print(f"Found {len(blobs)} monitoring reports")
    
    if not blobs:
        print("⚠️  No reports found. Run monitoring first.")
        return False
    
    # Get latest report
    latest_blob = sorted(blobs, key=lambda b: b.time_created, reverse=True)[0]
    print(f"\nLatest report: {latest_blob.name}")
    print(f"Created: {latest_blob.time_created}")
    
    # Download and display summary
    try:
        content = latest_blob.download_as_text()
        report = json.loads(content)
        
        print("\nReport Summary:")
        print(f"  Timestamp: {report.get('timestamp')}")
        print(f"  Needs Retraining: {report.get('decision', {}).get('needs_retraining', False)}")
        print(f"  Strategy: {report.get('decision', {}).get('strategy', 'none')}")
        
        metrics = report.get('metrics', {})
        print(f"\nMetrics:")
        print(f"  Total Predictions: {metrics.get('total_predictions', 0)}")
        print(f"  Error Rate: {metrics.get('error_rate', 0):.2%}")
        print(f"  Avg Latency: {metrics.get('avg_latency', 0):.2f}s")
        print(f"  Avg Composite Score: {metrics.get('avg_composite_score', 'N/A')}")
        print(f"  Embedding Coverage: {metrics.get('embedding_space_coverage', 0):.2f}%")
        
        drift = report.get('drift', {})
        print(f"\nDrift Detection:")
        print(f"  Has Drift: {drift.get('has_drift', False)}")
        
        return True
    except Exception as e:
        print(f"Error reading report: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify monitoring logging")
    parser.add_argument("--project-id", default="medscanai-476500", help="GCP project ID")
    parser.add_argument("--bucket", default="medscan-pipeline-medscanai-476500", help="GCS bucket")
    parser.add_argument("--hours", type=int, default=24, help="Hours of logs to check")
    
    args = parser.parse_args()
    
    print("="*70)
    print("VERIFYING MONITORING LOGGING")
    print("="*70)
    
    logs_ok = check_cloud_logging(args.project_id, args.hours)
    reports_ok = check_gcs_reports(args.bucket)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Cloud Logging: {'✓ PASS' if logs_ok else '✗ FAIL'}")
    print(f"GCS Reports: {'✓ PASS' if reports_ok else '✗ FAIL'}")
    
    if logs_ok and reports_ok:
        print("\n✓ All logging verification passed!")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review output above.")
        return 1


if __name__ == "__main__":
    exit(main())

