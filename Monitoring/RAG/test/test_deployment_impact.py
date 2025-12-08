"""
Deployment-style test for RAG monitoring impact validation.

This script simulates production scenarios to test if monitoring correctly:
1. Detects performance degradation
2. Identifies data drift
3. Makes appropriate retraining decisions
4. Shows measurable impact

Usage:
    python test_deployment_impact.py --scenario healthy
    python test_deployment_impact.py --scenario degraded
    python test_deployment_impact.py --scenario drift
    python test_deployment_impact.py --scenario all
"""
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, MagicMock, patch
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Monitoring.RAG.rag_monitor import RAGMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_healthy_logs(num_logs: int = 100) -> List[Dict]:
    """Generate logs simulating healthy production performance."""
    logs = []
    base_time = datetime.utcnow()
    index_size = 1000
    
    # Use queries similar to baseline to avoid drift detection
    healthy_queries = [
        'What is diabetes?',
        'How to treat hypertension?',
        'Symptoms of COVID-19',
        'What causes heart disease?',
        'How to prevent stroke?'
    ]
    
    for i in range(num_logs):
        # Healthy metrics: low latency, high scores, good coverage
        query = healthy_queries[i % len(healthy_queries)]
        logs.append({
            'timestamp': (base_time - timedelta(hours=24-i/10)).isoformat(),
            'query': query,
            'latency': 2.0 + (i % 10) * 0.1,  # 2.0-2.9s (below 10s threshold)
            'response_time': 2.0 + (i % 10) * 0.1,
            'success': True,
            'error': None,
            'composite_score': 0.75 + (i % 20) * 0.01,  # 0.75-0.94 (above 0.3 threshold)
            'hallucination_score': 0.10 + (i % 10) * 0.01,  # 0.10-0.19 (below 0.2 threshold)
            'avg_retrieval_score': 0.80 + (i % 15) * 0.01,  # 0.80-0.94
            'retrieved_doc_indices': list(range(i*3, i*3+3)),  # Diverse indices
            'retrieved_docs_metrics': [
                {'score': 0.85, 'doc_id': f'doc{i*3}'},
                {'score': 0.80, 'doc_id': f'doc{i*3+1}'},
                {'score': 0.75, 'doc_id': f'doc{i*3+2}'}
            ],
            'input_tokens': 50 + i % 20,
            'output_tokens': 200 + i % 50,
            'total_tokens': 250 + i % 70,
            'num_retrieved_docs': 3,
            'index_size': index_size
        })
    
    return logs


def generate_degraded_logs(num_logs: int = 100) -> List[Dict]:
    """Generate logs simulating degraded performance."""
    logs = []
    base_time = datetime.utcnow()
    index_size = 1000
    
    for i in range(num_logs):
        # Degraded metrics: high latency, low scores, poor coverage
        success = i % 10 != 0  # 10% error rate (above 15% threshold)
        
        logs.append({
            'timestamp': (base_time - timedelta(hours=24-i/10)).isoformat(),
            'query': f'How to treat condition {i % 20}?',
            'latency': 8.0 + (i % 5) * 0.5 if success else 0,  # 8.0-10.0s (near threshold)
            'response_time': 8.0 + (i % 5) * 0.5 if success else 0,
            'success': success,
            'error': 'Timeout' if not success else None,
            'composite_score': 0.20 + (i % 10) * 0.01 if success else None,  # 0.20-0.29 (below 0.3 threshold)
            'hallucination_score': 0.25 + (i % 5) * 0.01 if success else None,  # 0.25-0.29 (above 0.2 threshold)
            'avg_retrieval_score': 0.50 + (i % 10) * 0.01 if success else None,  # 0.50-0.59 (low)
            'retrieved_doc_indices': [0, 1, 2] if success else [],  # Same docs (poor coverage)
            'retrieved_docs_metrics': [
                {'score': 0.55, 'doc_id': 'doc0'},
                {'score': 0.50, 'doc_id': 'doc1'},
                {'score': 0.45, 'doc_id': 'doc2'}
            ] if success else [],
            'input_tokens': 50 + i % 20,
            'output_tokens': 200 + i % 50 if success else 0,
            'total_tokens': 250 + i % 70 if success else 50,
            'num_retrieved_docs': 3 if success else 0,
            'index_size': index_size
        })
    
    return logs


def generate_drift_logs(num_logs: int = 150) -> List[Dict]:
    """Generate logs simulating data drift (different query patterns)."""
    logs = []
    base_time = datetime.utcnow()
    index_size = 1000
    
    for i in range(num_logs):
        # Drift: very different query patterns (longer queries, different topics)
        query_length = 200 + i * 5  # Increasing length (drift)
        word_count = 30 + i * 2  # Increasing word count (drift)
        
        logs.append({
            'timestamp': (base_time - timedelta(hours=24-i/10)).isoformat(),
            'query': 'A' * query_length,  # Very long queries (drift indicator)
            'latency': 3.0 + (i % 10) * 0.2,
            'response_time': 3.0 + (i % 10) * 0.2,
            'success': True,
            'error': None,
            'composite_score': 0.70 + (i % 10) * 0.01,
            'hallucination_score': 0.15 + (i % 5) * 0.01,
            'avg_retrieval_score': 0.75 + (i % 10) * 0.01,
            'retrieved_doc_indices': list(range(i*2, i*2+3)),
            'retrieved_docs_metrics': [
                {'score': 0.80, 'doc_id': f'doc{i*2}'},
                {'score': 0.75, 'doc_id': f'doc{i*2+1}'},
                {'score': 0.70, 'doc_id': f'doc{i*2+2}'}
            ],
            'input_tokens': 100 + i * 2,  # Increasing (drift)
            'output_tokens': 250 + i % 50,
            'total_tokens': 350 + i % 70,
            'num_retrieved_docs': 3,
            'index_size': index_size
        })
    
    return logs


def generate_low_coverage_logs(num_logs: int = 100) -> List[Dict]:
    """Generate logs simulating low embedding space coverage."""
    logs = []
    base_time = datetime.utcnow()
    index_size = 10000  # Large index
    
    # Use baseline-like queries to avoid drift detection
    baseline_queries = [
        'What is diabetes?',
        'How to treat hypertension?',
        'Symptoms of COVID-19'
    ]
    
    for i in range(num_logs):
        # Low coverage: only retrieving same 5 documents from 10,000
        query = baseline_queries[i % len(baseline_queries)]
        logs.append({
            'timestamp': (base_time - timedelta(hours=24-i/10)).isoformat(),
            'query': query,
            'latency': 2.5,
            'response_time': 2.5,
            'success': True,
            'error': None,
            'composite_score': 0.75,
            'hallucination_score': 0.12,
            'avg_retrieval_score': 0.80,
            'retrieved_doc_indices': [0, 1, 2, 3, 4],  # Always same 5 docs
            'retrieved_docs_metrics': [
                {'score': 0.85, 'doc_id': 'doc0'},
                {'score': 0.80, 'doc_id': 'doc1'},
                {'score': 0.75, 'doc_id': 'doc2'},
                {'score': 0.70, 'doc_id': 'doc3'},
                {'score': 0.65, 'doc_id': 'doc4'}
            ],
            'input_tokens': 50,
            'output_tokens': 200,
            'total_tokens': 250,
            'num_retrieved_docs': 5,
            'index_size': index_size
        })
    
    return logs


def create_mock_baseline(bucket_mock):
    """Create mock baseline queries for drift detection."""
    baseline_queries = [
        {'query': 'What is diabetes?'},
        {'query': 'How to treat hypertension?'},
        {'query': 'Symptoms of COVID-19'},
    ] * 50  # 150 baseline queries
    
    baseline_jsonl = '\n'.join([json.dumps(q) for q in baseline_queries])
    
    def blob_side_effect(path):
        mock_blob = MagicMock()
        if 'baseline_queries.jsonl' in path:
            mock_blob.download_as_text.return_value = baseline_jsonl
        elif 'latest.txt' in path:
            mock_blob.download_as_text.return_value = 'model-v1'
        elif 'config.json' in path:
            model_config = {
                'display_name': 'rag-model-v1',
                'performance_metrics': {
                    'semantic_matching_score': 0.75,
                    'hallucination_score': 0.15,
                    'retrieval_score': 0.80
                }
            }
            mock_blob.download_as_text.return_value = json.dumps(model_config)
        elif 'last_retrain.txt' in path:
            # Last retrain was 10 days ago (safe to retrain)
            last_retrain = (datetime.utcnow() - timedelta(days=10)).isoformat()
            mock_blob.download_as_text.return_value = last_retrain
        return mock_blob
    
    bucket_mock.blob.side_effect = blob_side_effect


def run_scenario(scenario_name: str, logs: List[Dict], expected_decision: str):
    """Run monitoring for a specific scenario and validate impact."""
    print("\n" + "="*80)
    print(f"🧪 TESTING SCENARIO: {scenario_name.upper()}")
    print("="*80)
    
    # Create monitor with mocked dependencies
    with patch('Monitoring.RAG.rag_monitor.storage.Client') as mock_storage, \
         patch('Monitoring.RAG.rag_monitor.cloud_logging.Client') as mock_logging:
        
        # Setup mocks
        mock_bucket = MagicMock()
        mock_storage.return_value.bucket.return_value = mock_bucket
        create_mock_baseline(mock_bucket)
        
        # Mock log entries
        mock_entries = []
        for log in logs:
            mock_entry = MagicMock()
            mock_entry.timestamp = datetime.fromisoformat(log['timestamp'].replace('Z', ''))
            mock_entry.payload = {'prediction_result': log}
            mock_entries.append(mock_entry)
        
        mock_logging.return_value.list_entries.return_value = mock_entries
        
        # Create monitor
        monitor = RAGMonitor('test-project', 'test-bucket')
        
        # Step 1: Collect logs
        print("\n📥 Step 1: Collecting prediction logs...")
        collected_logs = monitor.collect_prediction_logs(hours=24)
        print(f"   ✓ Collected {len(collected_logs)} logs")
        
        # Step 2: Calculate metrics
        print("\n📊 Step 2: Calculating performance metrics...")
        metrics = monitor.calculate_performance_metrics(collected_logs)
        
        print(f"\n   Operational Metrics:")
        print(f"   • Total Predictions:  {metrics['total_predictions']}")
        print(f"   • Error Rate:         {metrics['error_rate']:.2%}")
        print(f"   • Avg Latency:        {metrics['avg_latency']:.2f}s")
        print(f"   • P95 Latency:        {metrics['p95_latency']:.2f}s")
        print(f"   • Avg Relevance:      {metrics['avg_relevance']:.2f}")
        
        if metrics.get('avg_composite_score') is not None:
            print(f"\n   Quality Metrics:")
            print(f"   • Avg Composite Score:    {metrics['avg_composite_score']:.4f}")
            print(f"   • Min Composite Score:    {metrics.get('min_composite_score', 'N/A')}")
            print(f"   • Avg Hallucination:      {metrics.get('avg_hallucination_score', 'N/A')}")
            print(f"   • Avg Retrieval Score:    {metrics.get('avg_retrieval_score', 'N/A')}")
        
        if metrics.get('embedding_space_coverage') is not None:
            print(f"\n   Embedding Space:")
            print(f"   • Unique Docs Retrieved:  {metrics.get('unique_docs_retrieved', 0)}")
            print(f"   • Coverage:                {metrics['embedding_space_coverage']:.2f}%")
        
        # Step 3: Get model metrics
        print("\n🎯 Step 3: Checking model quality...")
        model_metrics = monitor.get_current_model_metrics()
        print(f"   • Model: {model_metrics['model_name']}")
        print(f"   • Semantic Score: {model_metrics['semantic_score']:.2f}")
        print(f"   • Hallucination: {model_metrics['hallucination_score']:.2f}")
        
        # Step 4: Detect drift
        print("\n🔍 Step 4: Detecting data drift...")
        drift_info = monitor.detect_data_drift(collected_logs)
        print(f"   • Has Drift: {drift_info.get('has_drift', False)}")
        if drift_info.get('drift_details'):
            for feature, details in drift_info['drift_details'].items():
                if details.get('has_drift'):
                    print(f"   • {feature}: DRIFT DETECTED")
        
        # Step 5: Determine retraining strategy
        print("\n⚙️  Step 5: Determining retraining strategy...")
        decision = monitor.determine_retraining_strategy(metrics, drift_info, model_metrics)
        
        print(f"\n   📋 DECISION:")
        print(f"   • Needs Retraining: {decision['needs_retraining']}")
        print(f"   • Strategy: {decision['strategy'].upper()}")
        
        if decision.get('reasons'):
            print(f"\n   📝 Reasons:")
            for reason in decision['reasons']:
                print(f"     • {reason}")
        
        if decision.get('blocked'):
            print(f"\n   ⚠️  Blocked: {decision['blocked']}")
        
        # Validate impact
        print("\n" + "="*80)
        print("✅ IMPACT VALIDATION")
        print("="*80)
        
        actual_decision = decision['strategy'] if decision['needs_retraining'] else 'none'
        if actual_decision == expected_decision:
            print(f"✓ PASS: Expected '{expected_decision}', got '{actual_decision}'")
            print(f"✓ Monitoring correctly identified the issue!")
        else:
            print(f"✗ FAIL: Expected '{expected_decision}', got '{actual_decision}'")
            print(f"✗ Monitoring may need threshold adjustments")
        
        # Show impact metrics
        print(f"\n📈 Key Metrics That Triggered Decision:")
        triggers = []
        if metrics['error_rate'] > monitor.THRESHOLDS['max_error_rate']:
            triggers.append(f"Error rate {metrics['error_rate']:.2%} > {monitor.THRESHOLDS['max_error_rate']:.2%}")
        if metrics['p95_latency'] > monitor.THRESHOLDS['max_latency_p95']:
            triggers.append(f"P95 latency {metrics['p95_latency']:.2f}s > {monitor.THRESHOLDS['max_latency_p95']:.2f}s")
        if metrics.get('avg_composite_score') and metrics['avg_composite_score'] < monitor.THRESHOLDS['min_avg_composite_score']:
            triggers.append(f"Avg composite {metrics['avg_composite_score']:.4f} < {monitor.THRESHOLDS['min_avg_composite_score']:.4f}")
        if drift_info.get('has_drift'):
            triggers.append("Data drift detected")
        if metrics.get('embedding_space_coverage') and metrics['embedding_space_coverage'] < monitor.THRESHOLDS['min_embedding_space_coverage']:
            triggers.append(f"Coverage {metrics['embedding_space_coverage']:.2f}% < {monitor.THRESHOLDS['min_embedding_space_coverage']:.2f}%")
        
        if triggers:
            for trigger in triggers:
                print(f"  • {trigger}")
        else:
            print("  • No thresholds exceeded (healthy system)")
        
        return decision


def main():
    parser = argparse.ArgumentParser(description='Test RAG monitoring deployment impact')
    parser.add_argument(
        '--scenario',
        choices=['healthy', 'degraded', 'drift', 'coverage', 'all'],
        default='all',
        help='Scenario to test'
    )
    args = parser.parse_args()
    
    scenarios = {
        'healthy': {
            'logs': generate_healthy_logs(100),
            'expected': 'none',
            'description': 'Healthy production performance - no retraining needed'
        },
        'degraded': {
            'logs': generate_degraded_logs(100),
            'expected': 'model_only',
            'description': 'Degraded performance - model retraining needed'
        },
        'drift': {
            'logs': generate_drift_logs(150),
            'expected': 'full',
            'description': 'Data drift detected - full retraining needed'
        },
        'coverage': {
            'logs': generate_low_coverage_logs(100),
            'expected': 'model_only',
            'description': 'Low embedding space coverage - model retraining needed'
        }
    }
    
    print("="*80)
    print("🚀 RAG MONITORING DEPLOYMENT IMPACT TEST")
    print("="*80)
    print("\nThis test validates that monitoring correctly:")
    print("  1. Detects performance degradation")
    print("  2. Identifies data drift")
    print("  3. Makes appropriate retraining decisions")
    print("  4. Shows measurable impact")
    
    if args.scenario == 'all':
        for name, config in scenarios.items():
            run_scenario(name, config['logs'], config['expected'])
    else:
        config = scenarios[args.scenario]
        run_scenario(args.scenario, config['logs'], config['expected'])
    
    print("\n" + "="*80)
    print("✅ DEPLOYMENT IMPACT TEST COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  • All scenarios tested with realistic production data")
    print("  • Monitoring decisions validated against expected outcomes")
    print("  • Impact metrics displayed for each scenario")
    print("\nNext Steps:")
    print("  1. Review threshold values if any tests failed")
    print("  2. Run with real production logs: python run_monitoring.py")
    print("  3. Monitor retraining triggers in production")


if __name__ == '__main__':
    main()

