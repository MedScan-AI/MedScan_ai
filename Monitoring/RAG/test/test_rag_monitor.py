"""
Comprehensive tests for RAG monitoring system.

Tests the RAGMonitor class with mocked Google Cloud dependencies.
"""
import json
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Monitoring.RAG.rag_monitor import RAGMonitor


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    with patch('Monitoring.RAG.rag_monitor.storage.Client') as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        yield mock_bucket


@pytest.fixture
def mock_cloud_logging():
    """Mock Google Cloud Logging client."""
    with patch('Monitoring.RAG.rag_monitor.cloud_logging.Client') as mock_client:
        yield mock_client


@pytest.fixture
def sample_logs():
    """Sample prediction logs for testing."""
    return [
        {
            'timestamp': (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            'query': 'What is diabetes?',
            'latency': 2.5,
            'response_time': 2.5,
            'success': True,
            'error': None,
            'composite_score': 0.75,
            'hallucination_score': 0.15,
            'avg_retrieval_score': 0.80,
            'retrieved_doc_indices': [0, 1, 2],
            'retrieved_docs_metrics': [
                {'score': 0.85, 'doc_id': 'doc1'},
                {'score': 0.80, 'doc_id': 'doc2'},
                {'score': 0.75, 'doc_id': 'doc3'}
            ],
            'input_tokens': 50,
            'output_tokens': 200,
            'total_tokens': 250,
            'num_retrieved_docs': 3,
            'index_size': 1000
        },
        {
            'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            'query': 'How to treat hypertension?',
            'latency': 3.0,
            'response_time': 3.0,
            'success': True,
            'error': None,
            'composite_score': 0.70,
            'hallucination_score': 0.18,
            'avg_retrieval_score': 0.75,
            'retrieved_doc_indices': [1, 3, 4],
            'retrieved_docs_metrics': [
                {'score': 0.80, 'doc_id': 'doc2'},
                {'score': 0.75, 'doc_id': 'doc4'},
                {'score': 0.70, 'doc_id': 'doc5'}
            ],
            'input_tokens': 60,
            'output_tokens': 250,
            'total_tokens': 310,
            'num_retrieved_docs': 3,
            'index_size': 1000
        },
        {
            'timestamp': (datetime.utcnow() - timedelta(hours=3)).isoformat(),
            'query': 'Symptoms of COVID-19',
            'latency': 1.5,
            'response_time': 1.5,
            'success': False,
            'error': 'Timeout error',
            'composite_score': None,
            'hallucination_score': None,
            'avg_retrieval_score': None,
            'retrieved_doc_indices': [],
            'retrieved_docs_metrics': [],
            'input_tokens': 40,
            'output_tokens': 0,
            'total_tokens': 40,
            'num_retrieved_docs': 0,
            'index_size': 1000
        }
    ]


@pytest.fixture
def sample_model_config():
    """Sample model configuration."""
    return {
        'display_name': 'rag-model-v1',
        'performance_metrics': {
            'semantic_matching_score': 0.75,
            'hallucination_score': 0.15,
            'retrieval_score': 0.80,
            'composite_score': 0.70
        }
    }


@pytest.fixture
def monitor(mock_storage_client):
    """Create RAGMonitor instance with mocked dependencies."""
    return RAGMonitor(
        project_id='test-project',
        bucket_name='test-bucket'
    )


class TestRAGMonitorInitialization:
    """Tests for RAGMonitor initialization."""
    
    def test_monitor_initialization(self, monitor):
        """Test that RAGMonitor initializes correctly."""
        assert monitor is not None
        assert monitor.project_id == 'test-project'
        assert monitor.bucket_name == 'test-bucket'
        assert hasattr(monitor, 'drift_detector')
        assert hasattr(monitor, 'THRESHOLDS')
    
    def test_thresholds_exist(self, monitor):
        """Test that all required thresholds are defined."""
        required_thresholds = [
            'max_error_rate',
            'max_latency_p95',
            'min_relevance_score',
            'min_semantic_score',
            'max_hallucination_score',
            'min_composite_score',
            'min_avg_composite_score',
            'drift_p_value',
            'min_days_between_retrains',
            'min_embedding_space_coverage'
        ]
        for threshold in required_thresholds:
            assert threshold in monitor.THRESHOLDS, f"Missing threshold: {threshold}"


class TestCollectPredictionLogs:
    """Tests for log collection."""
    
    def test_collect_prediction_logs_success(self, monitor, mock_cloud_logging, sample_logs):
        """Test successful log collection."""
        # Mock log entries
        mock_entries = []
        for log in sample_logs:
            mock_entry = MagicMock()
            mock_entry.timestamp = datetime.fromisoformat(log['timestamp'].replace('Z', ''))
            mock_entry.payload = {
                'prediction_result': {
                    'query': log['query'],
                    'latency': log['latency'],
                    'success': log['success'],
                    'error': log['error'],
                    'composite_score': log['composite_score'],
                    'hallucination_score': log['hallucination_score'],
                    'avg_retrieval_score': log['avg_retrieval_score'],
                    'retrieved_doc_indices': log['retrieved_doc_indices'],
                    'retrieved_docs_metrics': log['retrieved_docs_metrics'],
                    'input_tokens': log['input_tokens'],
                    'output_tokens': log['output_tokens'],
                    'total_tokens': log['total_tokens'],
                    'num_retrieved_docs': log['num_retrieved_docs'],
                    'index_size': log['index_size']
                }
            }
            mock_entries.append(mock_entry)
        
        mock_cloud_logging.return_value.list_entries.return_value = mock_entries
        
        logs = monitor.collect_prediction_logs(hours=24)
        
        assert len(logs) == 3
        assert logs[0]['query'] == 'What is diabetes?'
        assert logs[0]['success'] is True
        assert logs[2]['success'] is False
    
    def test_collect_prediction_logs_empty(self, monitor, mock_cloud_logging):
        """Test log collection with no logs."""
        mock_cloud_logging.return_value.list_entries.return_value = []
        
        logs = monitor.collect_prediction_logs(hours=24)
        
        assert len(logs) == 0
    
    def test_collect_prediction_logs_malformed_entry(self, monitor, mock_cloud_logging):
        """Test log collection with malformed entries."""
        mock_entry = MagicMock()
        mock_entry.timestamp = datetime.utcnow()
        mock_entry.payload = {}  # Missing prediction_result
        
        mock_cloud_logging.return_value.list_entries.return_value = [mock_entry]
        
        logs = monitor.collect_prediction_logs(hours=24)
        
        assert len(logs) == 0


class TestCalculatePerformanceMetrics:
    """Tests for performance metrics calculation."""
    
    def test_calculate_metrics_success(self, monitor, sample_logs):
        """Test metrics calculation with successful logs."""
        metrics = monitor.calculate_performance_metrics(sample_logs)
        
        assert metrics['total_predictions'] == 3
        assert metrics['error_rate'] == pytest.approx(1/3, abs=0.01)
        assert metrics['avg_latency'] > 0
        assert metrics['p95_latency'] > 0
        assert metrics['avg_relevance'] > 0
        assert metrics['avg_composite_score'] is not None
        assert metrics['min_composite_score'] is not None
        assert metrics['embedding_space_coverage'] >= 0
    
    def test_calculate_metrics_empty_logs(self, monitor):
        """Test metrics calculation with empty logs."""
        metrics = monitor.calculate_performance_metrics([])
        
        assert metrics['total_predictions'] == 0
        assert metrics['error_rate'] == 0
        assert metrics['avg_latency'] == 0
        assert metrics['p95_latency'] == 0
        assert metrics['avg_relevance'] == 0
        assert metrics['avg_composite_score'] == 0
        assert metrics['min_composite_score'] == 0
    
    def test_calculate_metrics_all_failures(self, monitor):
        """Test metrics calculation when all requests fail."""
        failed_logs = [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'query': 'test query',
                'latency': 0,
                'response_time': 0,
                'success': False,
                'error': 'Error',
                'composite_score': None,
                'hallucination_score': None,
                'avg_retrieval_score': None,
                'retrieved_doc_indices': [],
                'retrieved_docs_metrics': [],
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'num_retrieved_docs': 0,
                'index_size': 1000
            }
        ] * 5
        
        metrics = monitor.calculate_performance_metrics(failed_logs)
        
        assert metrics['error_rate'] == 1.0
        assert metrics['total_predictions'] == 5
    
    def test_calculate_metrics_legacy_format(self, monitor):
        """Test metrics calculation with legacy log format."""
        legacy_logs = [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'query': 'test query',
                'latency': 2.0,
                'response_time': 2.0,
                'success': True,
                'error': None,
                'composite_score': None,
                'hallucination_score': None,
                'avg_retrieval_score': None,
                'retrieved_doc_indices': [],
                'retrieved_docs': [
                    {'score': 0.8, 'doc_id': 'doc1'},
                    {'score': 0.7, 'doc_id': 'doc2'}
                ],
                'input_tokens': 50,
                'output_tokens': 200,
                'total_tokens': 250,
                'num_retrieved_docs': 2,
                'index_size': 1000
            }
        ]
        
        metrics = monitor.calculate_performance_metrics(legacy_logs)
        
        assert metrics['avg_relevance'] > 0
        assert metrics['avg_relevance'] == pytest.approx(0.75, abs=0.01)


class TestGetCurrentModelMetrics:
    """Tests for getting current model metrics."""
    
    def test_get_model_metrics_success(self, monitor, mock_storage_client, sample_model_config):
        """Test successful retrieval of model metrics."""
        # Mock latest.txt
        mock_latest_blob = MagicMock()
        mock_latest_blob.download_as_text.return_value = 'model-v1'
        mock_storage_client.blob.return_value = mock_latest_blob
        
        # Mock config.json
        def blob_side_effect(path):
            mock_blob = MagicMock()
            if 'latest.txt' in path:
                mock_blob.download_as_text.return_value = 'model-v1'
            elif 'config.json' in path:
                mock_blob.download_as_text.return_value = json.dumps(sample_model_config)
            return mock_blob
        
        mock_storage_client.blob.side_effect = blob_side_effect
        
        metrics = monitor.get_current_model_metrics()
        
        assert metrics['model_name'] == 'rag-model-v1'
        assert metrics['semantic_score'] == 0.75
        assert metrics['hallucination_score'] == 0.15
        assert metrics['retrieval_score'] == 0.80
    
    def test_get_model_metrics_missing_config(self, monitor, mock_storage_client):
        """Test handling of missing model config."""
        def blob_side_effect(path):
            mock_blob = MagicMock()
            if 'latest.txt' in path:
                mock_blob.download_as_text.return_value = 'model-v1'
            elif 'config.json' in path:
                mock_blob.download_as_text.side_effect = Exception("Not found")
            return mock_blob
        
        mock_storage_client.blob.side_effect = blob_side_effect
        
        metrics = monitor.get_current_model_metrics()
        
        assert metrics['model_name'] == 'unknown'
        assert metrics['semantic_score'] == 0.0
        assert metrics['hallucination_score'] == 0.0
        assert metrics['retrieval_score'] == 0.0


class TestDetectDataDrift:
    """Tests for data drift detection."""
    
    def test_detect_drift_insufficient_data(self, monitor, sample_logs):
        """Test drift detection with insufficient data."""
        # Less than 100 logs
        drift_info = monitor.detect_data_drift(sample_logs[:2])
        
        assert drift_info['has_drift'] is False
        assert drift_info['reason'] == 'insufficient_data'
    
    def test_detect_drift_no_baseline(self, monitor, mock_storage_client):
        """Test drift detection with no baseline."""
        logs = [{'query': f'query {i}'} for i in range(150)]
        
        mock_blob = MagicMock()
        mock_blob.download_as_text.side_effect = Exception("Not found")
        mock_storage_client.blob.return_value = mock_blob
        
        drift_info = monitor.detect_data_drift(logs)
        
        assert drift_info['has_drift'] is False
        assert drift_info['reason'] == 'no_baseline'
    
    def test_detect_drift_with_baseline(self, monitor, mock_storage_client):
        """Test drift detection with baseline."""
        # Create enough logs
        logs = [{'query': f'What is disease {i}?'} for i in range(150)]
        
        # Create baseline
        baseline_queries = [{'query': f'How to treat condition {i}?'} for i in range(100)]
        baseline_jsonl = '\n'.join([json.dumps(q) for q in baseline_queries])
        
        def blob_side_effect(path):
            mock_blob = MagicMock()
            if 'baseline_queries.jsonl' in path:
                mock_blob.download_as_text.return_value = baseline_jsonl
            return mock_blob
        
        mock_storage_client.blob.side_effect = blob_side_effect
        
        drift_info = monitor.detect_data_drift(logs)
        
        assert 'has_drift' in drift_info
        assert 'drift_details' in drift_info


class TestDetermineRetrainingStrategy:
    """Tests for retraining strategy determination."""
    
    def test_no_retraining_needed(self, monitor):
        """Test when no retraining is needed."""
        # Create logs with healthy metrics
        # Use more diverse document indices to meet embedding space coverage threshold (1.0%)
        # If index_size is 100, we need at least 1 unique document retrieved
        healthy_logs = [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'query': 'What is diabetes?',
                'latency': 2.0,  # Below threshold
                'response_time': 2.0,
                'success': True,
                'error': None,
                'composite_score': 0.75,  # Above threshold
                'hallucination_score': 0.10,  # Below threshold
                'avg_retrieval_score': 0.80,  # Above threshold
                'retrieved_doc_indices': list(range(i*3, i*3+3)),  # Different indices per log
                'retrieved_docs_metrics': [
                    {'score': 0.85, 'doc_id': f'doc{i*3}'},
                    {'score': 0.80, 'doc_id': f'doc{i*3+1}'},
                    {'score': 0.75, 'doc_id': f'doc{i*3+2}'}
                ],
                'input_tokens': 50,
                'output_tokens': 200,
                'total_tokens': 250,
                'num_retrieved_docs': 3,
                'index_size': 50  # Smaller index so coverage is easier to meet
            }
            for i in range(10)  # 10 requests with different document indices
        ]
        
        metrics = monitor.calculate_performance_metrics(healthy_logs)
        drift_info = {'has_drift': False}
        model_metrics = {
            'semantic_score': 0.75,  # Above threshold
            'hallucination_score': 0.25,  # Above threshold (0.2 is minimum acceptable)
            'retrieval_score': 0.80,
            'model_name': 'test-model'
        }
        
        decision = monitor.determine_retraining_strategy(metrics, drift_info, model_metrics)
        
        assert decision['needs_retraining'] is False
        assert decision['strategy'] == 'none'
    
    def test_retraining_due_to_drift(self, monitor, sample_logs):
        """Test retraining triggered by data drift."""
        metrics = monitor.calculate_performance_metrics(sample_logs)
        drift_info = {
            'has_drift': True,
            'drift_details': {
                'query_length': {'has_drift': True},
                'word_count': {'has_drift': False}
            }
        }
        model_metrics = {
            'semantic_score': 0.75,
            'hallucination_score': 0.10,
            'retrieval_score': 0.80,
            'model_name': 'test-model'
        }
        
        decision = monitor.determine_retraining_strategy(metrics, drift_info, model_metrics)
        
        assert decision['needs_retraining'] is True
        assert decision['strategy'] == 'full'
        assert len(decision['reasons']) > 0
    
    def test_retraining_due_to_low_composite_score(self, monitor):
        """Test retraining triggered by low composite score."""
        logs = [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'query': 'test',
                'latency': 2.0,
                'response_time': 2.0,
                'success': True,
                'error': None,
                'composite_score': 0.20,  # Below threshold
                'hallucination_score': 0.15,
                'avg_retrieval_score': 0.70,
                'retrieved_doc_indices': [0, 1],
                'retrieved_docs_metrics': [{'score': 0.7}],
                'input_tokens': 50,
                'output_tokens': 200,
                'total_tokens': 250,
                'num_retrieved_docs': 2,
                'index_size': 1000
            }
        ] * 10
        
        metrics = monitor.calculate_performance_metrics(logs)
        drift_info = {'has_drift': False}
        model_metrics = {
            'semantic_score': 0.75,
            'hallucination_score': 0.10,
            'retrieval_score': 0.80,
            'model_name': 'test-model'
        }
        
        decision = monitor.determine_retraining_strategy(metrics, drift_info, model_metrics)
        
        assert decision['needs_retraining'] is True
        assert decision['strategy'] == 'model_only'
    
    def test_retraining_blocked_by_recent_retrain(self, monitor, mock_storage_client, sample_logs):
        """Test retraining blocked by recent retrain."""
        metrics = monitor.calculate_performance_metrics(sample_logs)
        drift_info = {'has_drift': True, 'drift_details': {}}
        model_metrics = {
            'semantic_score': 0.75,
            'hallucination_score': 0.10,
            'retrieval_score': 0.80,
            'model_name': 'test-model'
        }
        
        # Mock recent retrain (2 days ago)
        recent_date = (datetime.utcnow() - timedelta(days=2)).isoformat()
        mock_blob = MagicMock()
        mock_blob.download_as_text.return_value = recent_date
        mock_storage_client.blob.return_value = mock_blob
        
        decision = monitor.determine_retraining_strategy(metrics, drift_info, model_metrics)
        
        assert decision['needs_retraining'] is False
        assert 'blocked' in decision
    
    def test_retraining_due_to_high_error_rate(self, monitor):
        """Test retraining triggered by high error rate."""
        logs = [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'query': 'test',
                'latency': 2.0,
                'response_time': 2.0,
                'success': False,  # All failures
                'error': 'Error',
                'composite_score': None,
                'hallucination_score': None,
                'avg_retrieval_score': None,
                'retrieved_doc_indices': [],
                'retrieved_docs_metrics': [],
                'input_tokens': 50,
                'output_tokens': 0,
                'total_tokens': 50,
                'num_retrieved_docs': 0,
                'index_size': 1000
            }
        ] * 20  # 20 failures = 100% error rate
        
        metrics = monitor.calculate_performance_metrics(logs)
        drift_info = {'has_drift': False}
        model_metrics = {
            'semantic_score': 0.75,
            'hallucination_score': 0.10,
            'retrieval_score': 0.80,
            'model_name': 'test-model'
        }
        
        decision = monitor.determine_retraining_strategy(metrics, drift_info, model_metrics)
        
        assert decision['needs_retraining'] is True
        assert decision['strategy'] == 'model_only'
        assert any('error rate' in reason.lower() for reason in decision['reasons'])


class TestTriggerRetraining:
    """Tests for triggering retraining."""
    
    @patch('subprocess.run')
    @patch('Monitoring.RAG.rag_monitor.cloud_logging.Client')
    def test_trigger_full_retraining(self, mock_logging_client, mock_subprocess, monitor, mock_storage_client):
        """Test triggering full retraining."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_blob = MagicMock()
        mock_storage_client.blob.return_value = mock_blob
        
        # Mock Cloud Logging
        mock_logger_instance = MagicMock()
        mock_logging_client.return_value.logger.return_value = mock_logger_instance
        
        success = monitor.trigger_retraining('full', 'Data drift detected')
        
        assert success is True
        assert mock_subprocess.called
        # Check that gh workflow was called
        call_args = mock_subprocess.call_args[0][0]
        assert 'gh' in call_args
        assert 'workflow' in call_args
        assert 'rag-data-pipeline.yaml' in call_args
        # Verify Cloud Logging was called
        assert mock_logging_client.called
        assert mock_logger_instance.log_struct.called
        # Verify log payload contains retraining event
        log_call = mock_logger_instance.log_struct.call_args
        assert log_call[0][0]['retraining_event']['triggered'] is True
        assert log_call[0][0]['retraining_event']['strategy'] == 'full'
    
    @patch('subprocess.run')
    @patch('Monitoring.RAG.rag_monitor.cloud_logging.Client')
    def test_trigger_model_only_retraining(self, mock_logging_client, mock_subprocess, monitor, mock_storage_client):
        """Test triggering model-only retraining."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_blob = MagicMock()
        mock_storage_client.blob.return_value = mock_blob
        
        # Mock Cloud Logging
        mock_logger_instance = MagicMock()
        mock_logging_client.return_value.logger.return_value = mock_logger_instance
        
        success = monitor.trigger_retraining('model_only', 'Low composite score')
        
        assert success is True
        call_args = mock_subprocess.call_args[0][0]
        assert 'rag-training.yaml' in call_args
        # Verify Cloud Logging was called
        assert mock_logging_client.called
        assert mock_logger_instance.log_struct.called
        log_call = mock_logger_instance.log_struct.call_args
        assert log_call[0][0]['retraining_event']['strategy'] == 'model_only'
    
    @patch('subprocess.run')
    @patch('Monitoring.RAG.rag_monitor.cloud_logging.Client')
    def test_trigger_retraining_failure(self, mock_logging_client, mock_subprocess, monitor, mock_storage_client):
        """Test handling of retraining trigger failure."""
        import subprocess
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'gh', stderr='Command failed')
        mock_blob = MagicMock()
        mock_storage_client.blob.return_value = mock_blob
        
        # Mock Cloud Logging
        mock_logger_instance = MagicMock()
        mock_logging_client.return_value.logger.return_value = mock_logger_instance
        
        success = monitor.trigger_retraining('full', 'Test reason')
        
        assert success is False
        # Verify error was logged to Cloud Logging
        assert mock_logging_client.called
        # Should be called twice: once for initial log, once for error log
        assert mock_logger_instance.log_struct.call_count >= 1


class TestSaveMonitoringReport:
    """Tests for saving monitoring reports."""
    
    def test_save_report_success(self, monitor, mock_storage_client, sample_logs):
        """Test successful report saving."""
        metrics = monitor.calculate_performance_metrics(sample_logs)
        drift_info = {'has_drift': False}
        decision = {'needs_retraining': False, 'strategy': 'none', 'reasons': []}
        
        mock_blob = MagicMock()
        mock_storage_client.blob.return_value = mock_blob
        
        report = monitor.save_monitoring_report(metrics, drift_info, decision)
        
        assert 'timestamp' in report
        assert 'metrics' in report
        assert 'drift' in report
        assert 'decision' in report
        assert mock_blob.upload_from_string.called


class TestIntegration:
    """Integration tests for the full monitoring workflow."""
    
    @patch('subprocess.run')
    def test_full_monitoring_workflow(self, mock_subprocess, monitor, mock_cloud_logging, 
                                      mock_storage_client, sample_logs, sample_model_config):
        """Test the complete monitoring workflow."""
        # Setup mocks
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        # Mock log entries
        mock_entries = []
        for log in sample_logs:
            mock_entry = MagicMock()
            mock_entry.timestamp = datetime.fromisoformat(log['timestamp'].replace('Z', ''))
            mock_entry.payload = {'prediction_result': log}
            mock_entries.append(mock_entry)
        
        mock_cloud_logging.return_value.list_entries.return_value = mock_entries
        
        # Mock storage
        def blob_side_effect(path):
            mock_blob = MagicMock()
            if 'latest.txt' in path:
                mock_blob.download_as_text.return_value = 'model-v1'
            elif 'config.json' in path:
                mock_blob.download_as_text.return_value = json.dumps(sample_model_config)
            elif 'baseline_queries.jsonl' in path:
                baseline = [{'query': f'query {i}'} for i in range(100)]
                mock_blob.download_as_text.return_value = '\n'.join([json.dumps(q) for q in baseline])
            return mock_blob
        
        mock_storage_client.blob.side_effect = blob_side_effect
        
        # Run monitoring steps
        logs = monitor.collect_prediction_logs(hours=24)
        assert len(logs) > 0
        
        metrics = monitor.calculate_performance_metrics(logs)
        assert metrics['total_predictions'] > 0
        
        model_metrics = monitor.get_current_model_metrics()
        assert model_metrics['model_name'] != 'unknown'
        
        drift_info = monitor.detect_data_drift(logs)
        assert 'has_drift' in drift_info
        
        decision = monitor.determine_retraining_strategy(metrics, drift_info, model_metrics)
        assert 'needs_retraining' in decision
        
        report = monitor.save_monitoring_report(metrics, drift_info, decision)
        assert report is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

