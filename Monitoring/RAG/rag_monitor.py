"""
rag_monitor.py - RAG monitoring with intelligent retraining decisions
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pandas as pd
from google.cloud import storage
from google.cloud import logging as cloud_logging

import sys
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
        
        # NEW: Composite score threshold for production monitoring
        'min_composite_score': 0.25,  # Minimum composite score from production inference
        'min_avg_composite_score': 0.3,  # Minimum average composite score over monitoring period
        
        'drift_p_value': 0.05,
        
        'min_days_between_retrains': 7,
        
        # NEW: Embedding space coverage threshold for retraining decisions
        'min_embedding_space_coverage': 1.0,  # Minimum % of index that should be used
    }
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.drift_detector = DriftDetector()
    
    def collect_prediction_logs(self, hours: int = 24) -> List[Dict]:
        """Collect prediction logs from Cloud Logging"""
        logging_client = cloud_logging.Client(project=self.project_id)
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        filter_str = f"""
        resource.type="cloud_run_revision"
        resource.labels.service_name="rag-service"
        timestamp >= "{start_time.isoformat()}Z"
        jsonPayload.prediction_result:*
        """
        
        logs = []
        for entry in logging_client.list_entries(filter_=filter_str, max_results=10000):
            try:
                payload = entry.payload
                if 'prediction_result' in payload:
                    pred_result = payload.get('prediction_result', {})
                    logs.append({
                        'timestamp': entry.timestamp.isoformat(),
                        'query': pred_result.get('query', ''),  # Query for drift detection
                        'latency': pred_result.get('latency', 0),
                        'response_time': pred_result.get('latency', 0),  # Alias for clarity
                        'success': pred_result.get('success', False),
                        'error': pred_result.get('error', None),
                        # NEW: Composite score and component metrics
                        'composite_score': pred_result.get('composite_score'),
                        'hallucination_score': pred_result.get('hallucination_score'),
                        'avg_retrieval_score': pred_result.get('avg_retrieval_score'),
                        # NEW: Document indices for embedding space tracking
                        'retrieved_doc_indices': pred_result.get('retrieved_doc_indices', []),
                        'retrieved_docs_metrics': pred_result.get('retrieved_docs_metrics', []),
                        # Existing metrics
                        'input_tokens': pred_result.get('input_tokens', 0),
                        'output_tokens': pred_result.get('output_tokens', 0),
                        'total_tokens': pred_result.get('total_tokens', 0),
                        'num_retrieved_docs': pred_result.get('num_retrieved_docs', 0),
                        # NEW: Index size for embedding space coverage calculation
                        'index_size': pred_result.get('index_size'),
                        # Legacy support (if old format logs exist)
                        'retrieved_docs': pred_result.get('retrieved_docs', [])
                    })
            except Exception as e:
                logger.debug(f"Error parsing log entry: {e}")
                pass
        
        logger.info(f"Collected {len(logs)} logs")
        return logs
    
    def calculate_performance_metrics(self, logs: List[Dict]) -> Dict:
        """Calculate performance metrics including composite score and embedding space usage"""
        if not logs:
            return {
                'error_rate': 0, 'avg_latency': 0, 'p95_latency': 0,
                'avg_relevance': 0, 'total_predictions': 0,
                'avg_composite_score': 0, 'min_composite_score': 0,
                'avg_hallucination_score': 0, 'avg_retrieval_score': 0,
                'embedding_space_coverage': 0, 'unique_docs_retrieved': 0
            }
        
        df = pd.DataFrame(logs)
        error_rate = (~df['success']).mean()
        
        # Response time metrics
        latencies = df[df['success']]['latency']
        avg_latency = latencies.mean() if len(latencies) > 0 else 0
        p95_latency = latencies.quantile(0.95) if len(latencies) > 0 else 0
        
        # Relevance scores (from retrieved_docs_metrics or legacy format)
        relevance_scores = []
        for idx, row in df[df['success']].iterrows():
            # Try new format first
            if row.get('retrieved_docs_metrics'):
                scores = [d.get('score', 0) for d in row['retrieved_docs_metrics'] if isinstance(d, dict)]
                if scores:
                    relevance_scores.append(sum(scores) / len(scores))
            # Fallback to legacy format
            elif row.get('retrieved_docs'):
                if isinstance(row['retrieved_docs'], list):
                    scores = [d.get('score', 0) for d in row['retrieved_docs'] if isinstance(d, dict)]
                    if scores:
                        relevance_scores.append(sum(scores) / len(scores))
            # Use avg_retrieval_score if available
            elif row.get('avg_retrieval_score') is not None:
                relevance_scores.append(row['avg_retrieval_score'])
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # NEW: Composite score metrics from production
        composite_scores = df[df['success']]['composite_score'].dropna()
        avg_composite_score = composite_scores.mean() if len(composite_scores) > 0 else None
        min_composite_score = composite_scores.min() if len(composite_scores) > 0 else None
        
        # NEW: Component metrics (hallucination and retrieval)
        hallucination_scores = df[df['success']]['hallucination_score'].dropna()
        avg_hallucination_score = hallucination_scores.mean() if len(hallucination_scores) > 0 else None
        
        retrieval_scores = df[df['success']]['avg_retrieval_score'].dropna()
        avg_retrieval_score_metric = retrieval_scores.mean() if len(retrieval_scores) > 0 else None
        
        # NEW: Embedding space usage tracking
        all_retrieved_indices = []
        for indices in df[df['success']]['retrieved_doc_indices']:
            if isinstance(indices, list):
                all_retrieved_indices.extend(indices)
        
        unique_docs_retrieved = len(set(all_retrieved_indices)) if all_retrieved_indices else 0
        # Get index_size from successful logs (handle missing values)
        index_sizes = df[df['success']]['index_size'].dropna()
        total_index_size = index_sizes.mean() if len(index_sizes) > 0 else 0
        embedding_space_coverage = (unique_docs_retrieved / total_index_size * 100) if total_index_size > 0 else 0
        
        return {
            'error_rate': float(error_rate),
            'avg_latency': float(avg_latency),
            'p95_latency': float(p95_latency),
            'avg_relevance': float(avg_relevance),
            'total_predictions': len(df),
            # NEW metrics
            'avg_composite_score': float(avg_composite_score) if avg_composite_score is not None else None,
            'min_composite_score': float(min_composite_score) if min_composite_score is not None else None,
            'avg_hallucination_score': float(avg_hallucination_score) if avg_hallucination_score is not None else None,
            'avg_retrieval_score': float(avg_retrieval_score_metric) if avg_retrieval_score_metric is not None else None,
            'embedding_space_coverage': round(embedding_space_coverage, 2),
            'unique_docs_retrieved': unique_docs_retrieved,
            'retrieved_doc_indices_sample': list(set(all_retrieved_indices))[:100]  # Sample for analysis
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
        # Note: Drift detection now uses operational metrics, not query patterns
        if drift_info.get('has_drift', False):
            drift_details = drift_info.get('drift_details', {})
            drift_features = [f for f, d in drift_details.items() if d.get('has_drift', False)]
            reasons.append(f"Data drift detected --- Query pattern changed: {', '.join(drift_features)}")
            strategy = 'full'
        
        # Check for MODEL QUALITY DEGRADATION (model selection only)
        # Check static metrics from deployment config
        if model_metrics['semantic_score'] < self.THRESHOLDS['min_semantic_score']:
            reasons.append(
                f"Semantic score {model_metrics['semantic_score']:.2f} below "
                f"threshold {self.THRESHOLDS['min_semantic_score']}"
            )
            if strategy != 'full':
                strategy = 'model_only'
        
        if model_metrics['hallucination_score'] < self.THRESHOLDS['max_hallucination_score']:
            reasons.append(
                f"Hallucination score {model_metrics['hallucination_score']:.2f} below "
                f"threshold {self.THRESHOLDS['max_hallucination_score']}"
            )
            if strategy != 'full':
                strategy = 'model_only'
        
        # NEW: Check production composite score (from actual inference)
        if metrics.get('avg_composite_score') is not None:
            if metrics['avg_composite_score'] < self.THRESHOLDS['min_avg_composite_score']:
                reasons.append(
                    f"Production composite score {metrics['avg_composite_score']:.3f} below "
                    f"threshold {self.THRESHOLDS['min_avg_composite_score']} (performance degradation detected)"
                )
                if strategy != 'full':
                    strategy = 'model_only'
        
        if metrics.get('min_composite_score') is not None:
            if metrics['min_composite_score'] < self.THRESHOLDS['min_composite_score']:
                reasons.append(
                    f"Minimum composite score {metrics['min_composite_score']:.3f} below "
                    f"threshold {self.THRESHOLDS['min_composite_score']} (severe degradation detected)"
                )
                if strategy != 'full':
                    strategy = 'model_only'

        # NEW: Check embedding space coverage (indicates if model is using full knowledge base)
        if metrics.get('embedding_space_coverage') is not None:
            if metrics['embedding_space_coverage'] < self.THRESHOLDS['min_embedding_space_coverage']:
                reasons.append(
                    f"Embedding space coverage {metrics['embedding_space_coverage']:.2f}% below "
                    f"threshold {self.THRESHOLDS['min_embedding_space_coverage']}% (model not utilizing full knowledge base)"
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
        
        # Log retraining event to Cloud Logging for alerting
        try:
            logging_client = cloud_logging.Client(project=self.project_id)
            logger_instance = logging_client.logger("rag_monitoring")
            
            payload = {
                "retraining_event": {
                    "triggered": True,
                    "strategy": strategy,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "project_id": self.project_id
                }
            }
            logger_instance.log_struct(payload, severity="WARNING")
            logger.info("Retraining event logged to Cloud Logging")
        except Exception as e:
            logger.error(f"Failed to log retraining event to Cloud Logging: {e}")
        
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
            # Log failed retraining attempt
            try:
                logging_client = cloud_logging.Client(project=self.project_id)
                logger_instance = logging_client.logger("rag_monitoring")
                payload = {
                    "retraining_event": {
                        "triggered": False,
                        "strategy": strategy,
                        "reason": reason,
                        "error": str(e.stderr),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "project_id": self.project_id
                    }
                }
                logger_instance.log_struct(payload, severity="ERROR")
            except:
                pass
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