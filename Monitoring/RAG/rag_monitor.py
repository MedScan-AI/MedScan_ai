"""
rag_monitor.py - RAG monitoring with intelligent retraining decisions
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pandas as pd
from google.cloud import storage, logging as cloud_logging

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from DataPipeline.scripts.RAG.analysis.drift import DriftDetector

logger = logging.getLogger(__name__)

class RAGMonitor:
    """Monitor RAG model with intelligent retraining decisions"""
    
    THRESHOLDS = {
        'max_error_rate': 0.15,
        'max_latency_p95': 10.0,
        'min_relevance_score': 0.6,
        
        'min_semantic_score': 0.2,
        'max_hallucination_score': 0.2,
        
        'drift_p_value': 0.05,
        
        'min_days_between_retrains': 7
    }
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.drift_detector = DriftDetector()
    
    def collect_prediction_logs(self, hours: int = 24) -> List[Dict]:
        """Collect prediction logs from Cloud Logging"""
        logging_client = cloud_logging.Client()
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        filter_str = f"""
        resource.type="aiplatform.googleapis.com/Endpoint"
        timestamp >= "{start_time.isoformat()}Z"
        jsonPayload.prediction_result:*
        """
        
        logs = []
        for entry in logging_client.list_entries(filter_=filter_str, max_results=10000):
            try:
                payload = entry.payload
                if 'prediction_result' in payload:
                    logs.append({
                        'timestamp': entry.timestamp.isoformat(),
                        'query': payload.get('query', ''),
                        'latency': payload.get('latency', 0),
                        'retrieved_docs': payload.get('retrieved_docs', []),
                        'success': payload.get('success', False),
                        'error': payload.get('error', None)
                    })
            except:
                pass
        
        logger.info(f"Collected {len(logs)} logs")
        return logs
    
    def calculate_performance_metrics(self, logs: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        if not logs:
            return {
                'error_rate': 0, 'avg_latency': 0, 'p95_latency': 0,
                'avg_relevance': 0, 'total_predictions': 0
            }
        
        df = pd.DataFrame(logs)
        error_rate = (~df['success']).mean()
        
        latencies = df[df['success']]['latency']
        avg_latency = latencies.mean() if len(latencies) > 0 else 0
        p95_latency = latencies.quantile(0.95) if len(latencies) > 0 else 0
        
        relevance_scores = []
        for docs in df[df['success']]['retrieved_docs']:
            if docs:
                scores = [d.get('score', 0) for d in docs]
                if scores:
                    relevance_scores.append(sum(scores) / len(scores))
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        return {
            'error_rate': float(error_rate),
            'avg_latency': float(avg_latency),
            'p95_latency': float(p95_latency),
            'avg_relevance': float(avg_relevance),
            'total_predictions': len(df)
        }
    
    def get_current_model_metrics(self) -> Dict:
        """Get deployed model's quality metrics from config"""
        try:
            blob = self.bucket.blob("RAG/deployments/latest.txt")
            build_id = blob.download_as_text().strip()
            
            config_blob = self.bucket.blob(f"RAG/models/{build_id}/config.json")
            config = json.loads(config_blob.download_as_text())
            
            metrics = config.get('performance_metrics', {})
            
            return {
                'semantic_score': metrics.get('semantic_matching_score', 0.0),
                'hallucination_score': metrics.get('hallucination_score', 0.0),
                'retrieval_score': metrics.get('retrieval_score', 0.0),
                'model_name': config.get('display_name', 'unknown')
            }
        except Exception as e:
            logger.warning(f"Could not load model metrics: {e}")
            return {
                'semantic_score': 0.0,
                'hallucination_score': 0.0,
                'retrieval_score': 0.0,
                'model_name': 'unknown'
            }
    
    def detect_data_drift(self, logs: List[Dict]) -> Dict:
        """Detect drift using YOUR DriftDetector"""
        if len(logs) < 100:
            return {'has_drift': False, 'reason': 'insufficient_data'}
        
        try:
            blob = self.bucket.blob("RAG/monitoring/baseline_queries.jsonl")
            content = blob.download_as_text()
            baseline_df = pd.DataFrame([
                json.loads(line) for line in content.split('\n') if line.strip()
            ])
        except:
            return {'has_drift': False, 'reason': 'no_baseline'}
        
        current_df = pd.DataFrame(logs)
        
        baseline_features = pd.DataFrame({
            'query_length': baseline_df['query'].str.len(),
            'word_count': baseline_df['query'].str.split().str.len()
        })
        
        current_features = pd.DataFrame({
            'query_length': current_df['query'].str.len(),
            'word_count': current_df['query'].str.split().str.len()
        })
        
        drift_results = {}
        for feature in ['query_length', 'word_count']:
            drift_info = self.drift_detector.calculate_numerical_drift(
                baseline_features[feature],
                current_features[feature]
            )
            if drift_info:
                drift_results[feature] = drift_info
        
        has_drift = any(r.get('has_drift', False) for r in drift_results.values())
        
        return {
            'has_drift': has_drift,
            'drift_details': drift_results
        }
    
    def determine_retraining_strategy(
        self, 
        metrics: Dict, 
        drift_info: Dict, 
        model_metrics: Dict
    ) -> Dict:
        """
        Determine WHAT needs to be retrained
        
        Returns:
            {
                'needs_retraining': bool,
                'strategy': 'full' | 'model_only' | 'none',
                'reasons': List[str]
            }
        """
        reasons = []
        strategy = 'none'
        
        # Check for DATA DRIFT (requires full retraining)
        if drift_info.get('has_drift', False):
            reasons.append("Data drift detected - query patterns changed")
            strategy = 'full'
        
        # Check for MODEL QUALITY DEGRADATION (model selection only)
        if model_metrics['semantic_score'] < self.THRESHOLDS['min_semantic_score']:
            reasons.append(
                f"Semantic score {model_metrics['semantic_score']:.2f} below "
                f"threshold {self.THRESHOLDS['min_semantic_score']}"
            )
            if strategy != 'full':
                strategy = 'model_only'
        
        if model_metrics['hallucination_score'] > self.THRESHOLDS['max_hallucination_score']:
            reasons.append(
                f"Hallucination score {model_metrics['hallucination_score']:.2f} exceeds "
                f"threshold {self.THRESHOLDS['max_hallucination_score']}"
            )
            if strategy != 'full':
                strategy = 'model_only'
        
        # Check for OPERATIONAL ISSUES (model selection only)
        if metrics['error_rate'] > self.THRESHOLDS['max_error_rate']:
            reasons.append(f"Error rate {metrics['error_rate']:.2%} too high")
            if strategy != 'full':
                strategy = 'model_only'
        
        if metrics['p95_latency'] > self.THRESHOLDS['max_latency_p95']:
            reasons.append(f"P95 latency {metrics['p95_latency']:.2f}s too high")
            if strategy != 'full':
                strategy = 'model_only'
        
        if metrics['avg_relevance'] < self.THRESHOLDS['min_relevance_score']:
            reasons.append(f"Relevance {metrics['avg_relevance']:.2f} too low")
            if strategy != 'full':
                strategy = 'model_only'
        
        # Check last retrain date
        if strategy != 'none':
            try:
                blob = self.bucket.blob("RAG/monitoring/last_retrain.txt")
                date_str = blob.download_as_text().strip()
                last_retrain = datetime.fromisoformat(date_str)
                days_since = (datetime.utcnow() - last_retrain).days
                
                if days_since < self.THRESHOLDS['min_days_between_retrains']:
                    return {
                        'needs_retraining': False,
                        'strategy': 'none',
                        'reasons': reasons,
                        'blocked': f'Only {days_since} days since last retrain'
                    }
            except:
                pass
        
        return {
            'needs_retraining': strategy != 'none',
            'strategy': strategy,
            'reasons': reasons
        }
    
    def trigger_retraining(self, strategy: str, reason: str) -> bool:
        """
        Trigger appropriate retraining pipeline
        
        Args:
            strategy: 'full' or 'model_only'
            reason: Reason for retraining
        """
        import subprocess
        
        logger.info(f"🔄 Triggering {strategy.upper()} retraining: {reason}")
        
        try:
            if strategy == 'full':
                # Scenario 2: Full pipeline (data + model)
                logger.info("Stage 1: Triggering data pipeline...")
                subprocess.run([
                    'gh', 'workflow', 'run', 'rag-data-pipeline.yaml',
                    '-f', 'triggered_by=monitoring',
                    '-f', f'reason={reason}'
                ], check=True, capture_output=True)
                logger.info("Data pipeline triggered (model selection will auto-follow)")
                
            elif strategy == 'model_only':
                # Scenario 1: Model selection only (use existing data)
                logger.info("Triggering model selection only...")
                subprocess.run([
                    'gh', 'workflow', 'run', 'rag-training.yaml',
                    '-f', 'triggered_by=monitoring',
                    '-f', f'reason={reason}',
                    '-f', 'use_existing_data=true'
                ], check=True, capture_output=True)
                logger.info("Model selection triggered")
            
            # Update timestamp
            blob = self.bucket.blob("RAG/monitoring/last_retrain.txt")
            blob.upload_from_string(datetime.utcnow().isoformat())
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to trigger: {e.stderr}")
            return False
    
    def save_monitoring_report(self, metrics: Dict, drift: Dict, decision: Dict) -> Dict:
        """Save monitoring report to GCS"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'drift': drift,
            'decision': decision
        }
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        blob = self.bucket.blob(f"RAG/monitoring/reports/{timestamp}.json")
        blob.upload_from_string(json.dumps(report, indent=2, default=str))
        
        logger.info(f"Report saved: {blob.name}")
        return report