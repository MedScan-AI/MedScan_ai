#!/usr/bin/env python3
"""
Test script to trigger and verify each monitoring component.

Usage:
    python test_monitoring_triggers.py --service-url <URL> --component <component_name>
    
Components:
    - drift: Test data drift detection
    - error_rate: Test error rate trigger
    - latency: Test latency trigger
    - relevance: Test relevance trigger
    - composite_score: Test composite score trigger
    - embedding_coverage: Test embedding coverage trigger
"""
import argparse
import json
import time
import requests
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def send_request(service_url: str, query: str, wait_time: float = 0.1) -> Dict:
    """Send a request to the RAG service."""
    url = f"{service_url}/predict"
    payload = {"instances": [{"query": query}]}
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        time.sleep(wait_time)  # Small delay between requests
        return response.json()
    except Exception as e:
        print(f"Error sending request: {e}")
        return {"error": str(e)}


def generate_drift_queries() -> List[str]:
    """Generate queries with different patterns to trigger drift."""
    # Very short queries (different from baseline)
    short_queries = [
        "TB?",
        "Cancer?",
        "Diabetes?",
        "Fever?",
        "Cough?"
    ]
    
    # Very long queries (different from baseline)
    long_queries = [
        "What are the comprehensive diagnostic procedures and treatment protocols for advanced stage pulmonary tuberculosis in immunocompromised patients with drug-resistant strains?",
        "Can you provide detailed information about the molecular mechanisms, genetic predispositions, and environmental risk factors associated with the development and progression of non-small cell lung cancer?",
        "I need extensive information about the pathophysiology, clinical manifestations, differential diagnoses, and evidence-based management strategies for type 2 diabetes mellitus with complications."
    ]
    
    return short_queries + long_queries


def generate_error_queries() -> List[str]:
    """Generate queries that might cause errors."""
    # Empty or malformed queries
    return [
        "",  # Empty query
        " " * 100,  # Whitespace only
        "a" * 10000,  # Very long string
    ]


def generate_latency_queries() -> List[str]:
    """Generate complex queries that might take longer."""
    # Complex multi-part questions
    return [
        "What is tuberculosis? Also explain treatment, symptoms, diagnosis, prevention, and complications in detail.",
        "Compare and contrast lung cancer types, stages, treatments, survival rates, and risk factors comprehensively.",
        "Explain diabetes types, management, complications, monitoring, and lifestyle modifications with examples."
    ] * 20  # Send many to get high P95


def generate_low_relevance_queries() -> List[str]:
    """Generate queries that might have low relevance scores."""
    # Off-topic queries
    return [
        "What is the weather today?",
        "How do I cook pasta?",
        "What is the capital of France?",
        "Tell me a joke",
        "What time is it?"
    ] * 10


def generate_low_composite_queries() -> List[str]:
    """Generate queries that might result in low composite scores."""
    # Ambiguous or unclear queries
    return [
        "thing",
        "help",
        "info",
        "question",
        "problem"
    ] * 10


def generate_same_doc_queries() -> List[str]:
    """Generate queries that retrieve the same documents repeatedly."""
    # Very similar queries to get same docs
    return [
        "What is tuberculosis?",
        "Tell me about tuberculosis",
        "TB information",
        "Tuberculosis details",
        "Explain tuberculosis"
    ] * 20  # Many similar queries to reduce embedding space coverage


def wait_for_logs(hours: int = 1, min_logs: int = 10):
    """Wait for logs to appear in Cloud Logging."""
    print(f"Waiting {hours} hour(s) for logs to appear in Cloud Logging...")
    print("(In production, logs appear within seconds, but allowing buffer time)")
    time.sleep(5)  # Short wait for logs to propagate


def run_monitoring(project_id: str, bucket: str, hours: int = 1, trigger_retrain: bool = False) -> Dict:
    """Run the monitoring script and return results."""
    script_path = project_root / "Monitoring" / "RAG" / "run_monitoring.py"
    
    cmd = [
        "python", str(script_path),
        "--project-id", project_id,
        "--bucket", bucket,
        "--hours", str(hours)
    ]
    
    if trigger_retrain:
        cmd.append("--trigger-retrain")
    
    print(f"Running monitoring: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }


def check_workflow_triggered() -> List[Dict]:
    """Check if any workflows were triggered."""
    try:
        # Check for running workflows
        result = subprocess.run(
            ["gh", "run", "list", "--workflow=rag-data-pipeline.yaml", "--json", "databaseId,status,conclusion", "--limit", "5"],
            capture_output=True,
            text=True
        )
        data_pipeline_runs = json.loads(result.stdout) if result.returncode == 0 else []
        
        result = subprocess.run(
            ["gh", "run", "list", "--workflow=rag-training.yaml", "--json", "databaseId,status,conclusion", "--limit", "5"],
            capture_output=True,
            text=True
        )
        training_runs = json.loads(result.stdout) if result.returncode == 0 else []
        
        return data_pipeline_runs + training_runs
    except Exception as e:
        print(f"Error checking workflows: {e}")
        return []


def cancel_workflows(runs: List[Dict]):
    """Cancel running workflows."""
    for run in runs:
        if run.get("status") in ["queued", "in_progress"]:
            run_id = run.get("databaseId")
            if run_id:
                print(f"Cancelling workflow run {run_id}...")
                subprocess.run(["gh", "run", "cancel", str(run_id)], capture_output=True)


def run_component_test(service_url: str, component: str, project_id: str, bucket: str):
    """Test a specific monitoring component."""
    print(f"\n{'='*70}")
    print(f"Testing Component: {component.upper()}")
    print(f"{'='*70}\n")
    
    queries = []
    
    if component == "drift":
        queries = generate_drift_queries()
        print(f"Sending {len(queries)} queries with different patterns (short/long)...")
    elif component == "error_rate":
        queries = generate_error_queries()
        print(f"Sending {len(queries)} queries that may cause errors...")
    # elif component == "latency":
    #     queries = generate_latency_queries()
    #     print(f"Sending {len(queries)} complex queries to increase latency...")
    elif component == "relevance":
        queries = generate_low_relevance_queries()
        print(f"Sending {len(queries)} off-topic queries for low relevance...")
    elif component == "composite_score":
        queries = generate_low_composite_queries()
        print(f"Sending {len(queries)} ambiguous queries for low composite score...")
    elif component == "embedding_coverage":
        queries = generate_same_doc_queries()
        print(f"Sending {len(queries)} similar queries to reduce embedding coverage...")
    else:
        print(f"Unknown component: {component}")
        return
    
    # Send requests
    print(f"\nSending requests to {service_url}...")
    success_count = 0
    for i, query in enumerate(queries, 1):
        result = send_request(service_url, query)
        if result.get("predictions", [{}])[0].get("success"):
            success_count += 1
        if i % 10 == 0:
            print(f"  Sent {i}/{len(queries)} requests...")
    
    print(f"\nCompleted: {success_count}/{len(queries)} successful requests")
    
    # Wait for logs
    print("\nWaiting for logs to propagate to Cloud Logging...")
    wait_for_logs(hours=1)
    
    # Run monitoring
    print("\nRunning monitoring script...")
    result = run_monitoring(project_id, bucket, hours=1, trigger_retrain=False)
    
    print("\n" + "="*70)
    print("MONITORING OUTPUT")
    print("="*70)
    print(result["stdout"])
    if result["stderr"]:
        print("\nSTDERR:")
        print(result["stderr"])
    
    # Check for triggered workflows
    print("\n" + "="*70)
    print("CHECKING FOR TRIGGERED WORKFLOWS")
    print("="*70)
    workflows = check_workflow_triggered()
    if workflows:
        print(f"Found {len(workflows)} recent workflow runs")
        for wf in workflows[:5]:
            print(f"  - Run {wf.get('databaseId')}: {wf.get('status')} ({wf.get('conclusion', 'N/A')})")
        
        # Cancel running workflows
        running = [w for w in workflows if w.get("status") in ["queued", "in_progress"]]
        if running:
            print(f"\n⚠️  Found {len(running)} running workflows - cancelling to prevent costs...")
            cancel_workflows(running)
    else:
        print("No workflows triggered (this is expected if --trigger-retrain was not used)")
    
    print(f"\n{'='*70}")
    print(f"Component {component.upper()} test complete")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Test RAG monitoring components")
    parser.add_argument("--service-url", required=True, help="Cloud Run service URL")
    parser.add_argument("--component", required=True, 
                       choices=["drift", "error_rate", "latency", "relevance", 
                               "composite_score", "embedding_coverage"],
                       help="Component to test")
    parser.add_argument("--project-id", default="medscanai-476500", help="GCP project ID")
    parser.add_argument("--bucket", default="medscan-pipeline-medscanai-476500", help="GCS bucket name")
    parser.add_argument("--trigger-retrain", action="store_true", 
                       help="Actually trigger retraining (default: False, just test detection)")
    
    args = parser.parse_args()
    
    run_component_test(
        args.service_url,
        args.component,
        args.project_id,
        args.bucket
    )


if __name__ == "__main__":
    main()

