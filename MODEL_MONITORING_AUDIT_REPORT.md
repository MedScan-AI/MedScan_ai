# Model Monitoring & Retraining Readiness Audit Report

**Project**: MedScan AI  
**Auditor**: MLOps Audit Agent  
**Components Audited**: Vision (Computer Vision) + RAG (Retrieval Augmented Generation)

---

## Executive Summary

This audit evaluates the MedScan AI codebase for model monitoring, retraining triggers, and alerting infrastructure across **TWO separate ML components**: Vision (medical image classification) and RAG (medical Q&A system).

**Overall Status**: ⚠️ **PARTIAL IMPLEMENTATION**

- **RAG Component**: ✅ **Well Implemented** (6/7 categories complete, **recently enhanced with production quality metrics**)
- **Vision Component**: ⚠️ **Partially Implemented** (3/7 categories complete)

---

## CATEGORY 1: Performance Metrics Collection

| Sub-Component | VISION Status | RAG Status | Evidence | Gaps |
|-------------|----------------|------------|----------|------|
| Data Pipeline | ✅ | ✅ | Vision: `DataPipeline/scripts/data_preprocessing/schema_statistics.py` - tracks preprocessing metrics, data quality stats<br>RAG: `DataPipeline/scripts/RAG/analysis/main.py` - tracks scraping success, embedding generation metrics | None |
| Model Development | ✅ | ✅ | Vision: `ModelDevelopment/Vision/metrics_tracker.py` - comprehensive training metrics (accuracy, precision, recall, F1, AUC)<br>RAG: `ModelDevelopment/RAG/ModelSelection/experiment.py` - tracks retrieval accuracy, semantic scores, hallucination scores | None |
| Model Deployment | ⚠️ | ✅ | Vision: ⚠️ **PARTIAL** - Metrics tracked during training but **no inference logging found** in deployment code<br>RAG: ✅ `deployment/RAG_serve.py` - logs queries, responses, stats; `Monitoring/RAG/rag_monitor.py` collects prediction logs from Cloud Logging with **execution time tracking** (avg_latency, p95_latency) | **Vision**: Missing inference latency/execution time logging, prediction confidence scores, throughput metrics collection during deployment |
| Monitoring | ⚠️ | ✅ | Vision: ⚠️ **PARTIAL** - `ModelDevelopment/common/monitoring.py` has infrastructure but **no active vision monitoring script**<br>RAG: ✅ `Monitoring/RAG/rag_monitor.py` - real-time tracking of error rate, latency (avg, p95), relevance scores | **Vision**: Missing real-time accuracy tracking, ground truth label collection for production images |

**Category Status**: ⚠️ **Partial** (Vision missing deployment inference logging and real-time monitoring)

---

## CATEGORY 2: Data Drift Detection Infrastructure

| Sub-Component | VISION Status | RAG Status | Evidence | Gaps |
|-------------|----------------|------------|----------|------|
| Data Pipeline | ✅ | ✅ | Vision: `DataPipeline/scripts/data_preprocessing/schema_statistics.py` - `detect_drift()` method uses KS test for numerical, Chi-square for categorical<br>RAG: `DataPipeline/scripts/RAG/analysis/drift.py` - `DriftDetector` class with numerical/categorical drift detection | None |
| Model Development | ✅ | ✅ | Vision: Training data statistics saved in metadata files (`training_metadata.json`)<br>RAG: Baseline queries stored in `RAG/monitoring/baseline_queries.jsonl` | None |
| Model Deployment | ⚠️ | ✅ | Vision: ⚠️ **PARTIAL** - No incoming image characteristics logging found in deployment code<br>RAG: ✅ `Monitoring/RAG/rag_monitor.py` - logs query patterns, retrieved document characteristics | **Vision**: Missing incoming image distribution statistics (brightness, resolution, class balance) logging |
| Monitoring | ⚠️ | ✅ | Vision: ⚠️ **PARTIAL** - Drift detection exists in data pipeline but **no production drift monitoring**<br>RAG: ✅ `Monitoring/RAG/rag_monitor.py` - `detect_data_drift()` compares current queries vs baseline | **Vision**: Missing visual drift detection comparing production image stats vs training baseline |

**Category Status**: ⚠️ **Partial** (Vision missing deployment-level drift tracking and production monitoring)

---

## CATEGORY 3: Threshold & Configuration Management

| Sub-Component | VISION Status | RAG Status | Evidence | Gaps |
|-------------|----------------|------------|----------|------|
| Data Pipeline | ✅ | ✅ | Vision: `DataPipeline/config/vision_pipeline.yml` - data quality thresholds<br>RAG: `DataPipeline/config/rag_pipeline.yml` - `alerts.thresholds` section with scraping, validation, embedding thresholds | None |
| Model Development | ✅ | ✅ | Vision: `ModelDevelopment/config/vision_training.yml` - early stopping criteria, convergence thresholds<br>RAG: `ModelDevelopment/RAG/ModelSelection/select_best_model.py` - validation threshold (0.2) | None |
| Model Deployment | ✅ | ✅ | Vision: `deployment/monitoring_config.yaml` - confidence thresholds, latency SLAs<br>RAG: `Monitoring/RAG/rag_monitor.py` - `THRESHOLDS` dict with max_error_rate, max_latency_p95, min_relevance_score | None |
| Monitoring | ✅ | ✅ | Vision: `deployment/monitoring_config.yaml` - performance degradation thresholds (0.70 accuracy), drift alert thresholds<br>RAG: `Monitoring/RAG/rag_monitor.py` - drift_p_value (0.05), min_semantic_score (0.2), max_hallucination_score (0.2) | None |

**Category Status**: ✅ **Complete**

---

## CATEGORY 4: Retraining Pipeline Readiness

| Sub-Component | VISION Status | RAG Status | Evidence | Gaps |
|-------------|----------------|------------|----------|------|
| Data Pipeline | ✅ | ✅ | Vision: DVC versioning in `DataPipeline/data/`, new labeled image ingestion via Airflow DAG<br>RAG: DVC versioning, corpus update mechanism via `DataPipeline/scripts/RAG/scraper.py` | None |
| Model Development | ✅ | ✅ | Vision: `cloudbuild/vision-training.yaml` - automated training script, hyperparameter configs in YAML<br>RAG: `cloudbuild/rag-training.yaml` - automated training, HPO experiments | None |
| Model Deployment | ✅ | ✅ | Vision: `ModelDevelopment/Vision/deploy.py` - model versioning via Vertex AI Model Registry, deployment info saved<br>RAG: `ModelDevelopment/RAG/deploy.py` - model versioning, index update mechanism | None |
| Monitoring | ⚠️ | ✅ | Vision: ⚠️ **PARTIAL** - `scripts/trigger_retraining.py` exists but **no monitoring → training trigger connection**<br>RAG: ✅ `Monitoring/RAG/rag_monitor.py` - `trigger_retraining()` method connects monitoring to GitHub Actions workflows | **Vision**: Missing automated trigger mechanism from monitoring alerts to training pipeline |

**Category Status**: ⚠️ **Partial** (Vision missing monitoring-to-training trigger)

---

## CATEGORY 5: Model Comparison & Validation

| Sub-Component | VISION Status | RAG Status | Evidence | Gaps |
|-------------|----------------|------------|----------|------|
| Data Pipeline | ✅ | ✅ | Vision: Test set versioning via DVC, validation datasets in `DataPipeline/data/preprocessed/`<br>RAG: Evaluation query set in `ModelDevelopment/RAG/ModelSelection/qa.json`, ground truth answers | None |
| Model Development | ✅ | ✅ | Vision: `ModelDevelopment/Vision/select_best_model.py` - compares ResNet, ViT, Custom CNN architectures<br>RAG: `ModelDevelopment/RAG/ModelSelection/select_best_model.py` - compares multiple embedding models, evaluates MRR, nDCG | None |
| Model Deployment | ✅ | ✅ | Vision: ✅ Model versioning exists via Vertex AI Model Registry<br>RAG: ✅ Model versioning exists via Vertex AI Model Registry | None |
| Monitoring | ⚠️ | ⚠️ | Vision: ⚠️ **PARTIAL** - No champion/challenger comparison or rollback criteria found<br>RAG: ⚠️ **PARTIAL** - No online evaluation metrics or user preference tracking found | **Both**: Missing champion/challenger comparison, rollback criteria, online evaluation |

**Category Status**: ⚠️ **Partial** (Both missing monitoring-level champion/challenger comparison)

---

## CATEGORY 6: Alerting Infrastructure Readiness

| Sub-Component | VISION Status | RAG Status | Evidence | Gaps |
|-------------|----------------|------------|----------|------|
| Data Pipeline | ✅ | ✅ | Vision: `airflow/dags/medscan_vision_pipeline_gcs.py` - data quality alerts, ingestion failure alerts via email<br>RAG: `DataPipeline/config/rag_pipeline.yml` - alert thresholds, email recipients configured | None |
| Model Development | ✅ | ✅ | Vision: `ModelDevelopment/common/email_notifier.py` - training failure notifications, convergence alerts<br>RAG: `ModelDevelopment/RAG/utils/send_notification.py` - training completion, validation failure, bias violation alerts | None |
| Model Deployment | ✅ | ✅ | Vision: `deployment/monitoring_config.yaml` - service health alerts, latency spike alerts configured<br>RAG: `deployment/monitoring_config.yaml` - query failure alerts, LLM error alerts configured | None |
| Monitoring | ✅ | ✅ | Vision: `deployment/monitoring_config.yaml` - performance degradation alert configs (accuracy < 0.70), drift alert templates<br>RAG: `Monitoring/RAG/rag_monitor.py` - answer quality alerts, retrieval relevance drop alerts (via thresholds) | None |

**Category Status**: ✅ **Complete**

---

## CATEGORY 7: Logging & Observability

| Sub-Component | VISION Status | RAG Status | Evidence | Gaps |
|-------------|----------------|------------|----------|------|
| Data Pipeline | ✅ | ✅ | Vision: `DataPipeline/scripts/data_preprocessing/` - image processing logs, data transformation audit trail<br>RAG: `DataPipeline/scripts/RAG/` - document processing logs, embedding generation logs | None |
| Model Development | ✅ | ✅ | Vision: MLflow integration in training scripts, experiment tracking, epoch-by-epoch logs<br>RAG: MLflow tracking in `ModelDevelopment/RAG/ModelSelection/experiment.py`, prompt version tracking | None |
| Model Deployment | ⚠️ | ✅ | Vision: ⚠️ **PARTIAL** - No inference logs found in `ModelDevelopment/Vision/deploy.py`<br>RAG: ✅ `deployment/RAG_serve.py` - logs queries, responses; `Monitoring/RAG/rag_monitor.py` collects from Cloud Logging | **Vision**: Missing inference logs, prediction audit trail during deployment |
| Monitoring | ⚠️ | ✅ | Vision: ⚠️ **PARTIAL** - `ModelDevelopment/common/monitoring.py` has infrastructure but **no centralized dashboard**<br>RAG: ✅ `Monitoring/RAG/rag_monitor.py` - centralized monitoring with reports saved to GCS | **Vision**: Missing centralized dashboard for vision metrics visualization |

**Category Status**: ⚠️ **Partial** (Vision missing deployment inference logging and centralized dashboard)

---

## Final Summary Table

| Category | VISION | RAG | Overall |
|----------|--------|-----|---------|
| 1. Performance Metrics | 2/4 | 4/4 | 6/8 |
| 2. Data Drift Detection | 2/4 | 4/4 | 6/8 |
| 3. Threshold Configuration | 4/4 | 4/4 | 8/8 |
| 4. Retraining Pipeline | 3/4 | 4/4 | 7/8 |
| 5. Model Comparison | 3/4 | 3/4 | 6/8 |
| 6. Alerting Infrastructure | 4/4 | 4/4 | 8/8 |
| 7. Logging & Observability | 2/4 | 4/4 | 6/8 |
| **TOTAL** | **20/28** | **27/28** ⬆️ | **47/56** |

---

## Gap Analysis

### VISION Component Gaps (Priority Order):

1. **🚨 CRITICAL: Missing Production Inference Logging**
   - **Location**: `ModelDevelopment/Vision/deploy.py`
   - **Gap**: No code to log predictions, confidence scores, **execution time/latency** during inference
   - **Impact**: Cannot track production performance, detect degradation, or collect ground truth. **Cannot monitor inference latency** (unlike RAG which tracks avg_latency and p95_latency)
   - **Recommendation**: Add logging to Vertex AI endpoint or inference service to log:
     - Prediction results with confidence scores
     - **Inference execution time/latency per request** (critical for performance monitoring)
     - Image metadata (resolution, brightness stats)
     - Request timestamps

2. **🚨 CRITICAL: Missing Real-Time Monitoring Script**
   - **Location**: `Monitoring/` directory (Vision subdirectory missing)
   - **Gap**: No equivalent to `Monitoring/RAG/rag_monitor.py` for Vision models
   - **Impact**: Cannot automatically detect performance degradation or drift in production
   - **Recommendation**: Create `Monitoring/Vision/vision_monitor.py` that:
     - Collects prediction logs from Cloud Logging
     - Calculates accuracy, **execution time/latency** (avg, p95), error rates
     - Detects data drift (image statistics vs baseline)
     - Triggers retraining when thresholds breached (including latency thresholds)

3. **⚠️ HIGH: Missing Monitoring-to-Training Trigger**
   - **Location**: `scripts/trigger_retraining.py` exists but not connected to monitoring
   - **Gap**: `trigger_retraining.py` can check metrics but no automated connection from monitoring alerts
   - **Impact**: Manual intervention required to retrain when issues detected
   - **Recommendation**: Integrate `Monitoring/Vision/vision_monitor.py` with `scripts/trigger_retraining.py` or Cloud Build triggers

4. **⚠️ HIGH: Missing Production Drift Monitoring**
   - **Location**: Deployment/inference code
   - **Gap**: No tracking of incoming image characteristics (brightness, resolution, class distribution)
   - **Impact**: Cannot detect when production data distribution shifts from training data
   - **Recommendation**: Add image statistics logging in inference service, compare against training baseline

5. **⚠️ MEDIUM: Missing Centralized Dashboard**
   - **Location**: Monitoring infrastructure
   - **Gap**: No visualization dashboard for Vision metrics (RAG has reports in GCS)
   - **Impact**: Difficult to visualize trends and performance over time
   - **Recommendation**: Create GCS-based reports similar to RAG, or integrate with GCP Monitoring dashboards

### RAG Component Gaps (Priority Order):

1. **✅ RESOLVED: Production Quality Metrics** (Previously: Missing production inference quality tracking)
   - **Status**: ✅ **IMPLEMENTED** - Production composite scores, hallucination scores, and retrieval scores are now tracked and used in retraining decisions
   - **Evidence**: `deployment/RAG_serve.py` logs composite scores; `Monitoring/RAG/rag_monitor.py` uses them for retraining decisions
   - **Impact**: ✅ Can now detect production performance degradation using actual inference metrics

2. **✅ RESOLVED: Embedding Space Usage Tracking** (Previously: Missing knowledge base utilization tracking)
   - **Status**: ✅ **IMPLEMENTED** - Embedding space coverage is now tracked and used in retraining decisions
   - **Evidence**: `Monitoring/RAG/rag_monitor.py` calculates embedding space coverage; used in `determine_retraining_strategy()`
   - **Impact**: ✅ Can now detect if model is not utilizing full knowledge base effectively

3. **⚠️ MEDIUM: Missing Online Evaluation Metrics**
   - **Location**: Monitoring code
   - **Gap**: No user feedback collection or preference tracking
   - **Impact**: Cannot measure user satisfaction or improve based on real usage
   - **Recommendation**: Add feedback collection endpoint and integrate with monitoring

### Cross-Component Gaps (affects both):

1. **⚠️ MEDIUM: Missing Champion/Challenger Comparison**
   - **Location**: Monitoring infrastructure
   - **Gap**: No automated comparison of new model vs current production model
   - **Impact**: Cannot make data-driven decisions about model updates
   - **Recommendation**: Implement comparison logic that evaluates new model on held-out test set before deployment

2. **⚠️ LOW: Missing Rollback Criteria**
   - **Location**: Deployment/monitoring code
   - **Gap**: No automated rollback triggers if new model performs worse
   - **Impact**: Manual intervention required to revert bad deployments
   - **Recommendation**: Add rollback logic that monitors new model performance and reverts if metrics drop below threshold

---

## Recommendations

### Immediate Priority (Blocking):

1. **Implement Vision Production Inference Logging** ⚠️ **CRITICAL**
   - Add logging to `ModelDevelopment/Vision/deploy.py` or Vertex AI endpoint
   - Log: predictions, confidence scores, latency, image metadata
   - Store logs in Cloud Logging for monitoring collection

2. **Create Vision Monitoring Script** ⚠️ **CRITICAL**
   - Create `Monitoring/Vision/vision_monitor.py` similar to RAG monitor
   - Implement: log collection, metric calculation, drift detection, retraining triggers
   - Schedule via cron or Cloud Scheduler

3. **Connect Vision Monitoring to Retraining** ⚠️ **HIGH**
   - Integrate `Monitoring/Vision/vision_monitor.py` with `scripts/trigger_retraining.py`
   - Or use Cloud Build triggers similar to RAG implementation
   - Enable automated retraining when thresholds breached

### High Priority:

4. **Add Production Drift Tracking for Vision**
   - Log image statistics (brightness, resolution, class balance) during inference
   - Compare against training baseline in monitoring script
   - Alert when drift detected

5. **Create Vision Metrics Dashboard**
   - Generate monitoring reports similar to RAG (save to GCS)
   - Or create GCP Monitoring dashboard
   - Visualize trends over time

### Nice to Have:

7. **Add User Feedback Collection (RAG)**
   - Implement feedback endpoint in `deployment/RAG_serve.py`
   - Track user satisfaction scores
   - Integrate with monitoring for continuous improvement

8. **Implement Automated Rollback**
   - Add rollback logic to deployment code
   - Monitor new model performance post-deployment
   - Automatically revert if metrics drop below threshold

9. **Add Champion/Challenger Comparison**
   - Implement comparison logic in monitoring
   - Evaluate new model on test set before deployment
   - Provide decision metrics for model updates

---

## Implementation Notes

### Vision Monitoring Implementation Example:

```python
# Monitoring/Vision/vision_monitor.py (to be created)
class VisionMonitor:
    def collect_prediction_logs(self, hours: int = 24):
        # Collect from Cloud Logging (similar to RAG)
        # Filter: resource.type="aiplatform.googleapis.com/Endpoint"
        # Extract: predictions, confidence, latency, image_metadata
    
    def calculate_performance_metrics(self, logs):
        # Calculate: accuracy, precision, recall, latency (avg, p95)
        # Requires: ground truth labels (from manual review or feedback)
    
    def detect_data_drift(self, logs):
        # Compare image statistics (brightness, resolution) vs training baseline
        # Use KS test or similar statistical tests
    
    def trigger_retraining(self, reason: str):
        # Trigger Cloud Build or GitHub Actions workflow
        # Similar to RAG implementation
```

### Inference Logging Implementation Example:

```python
# Add to ModelDevelopment/Vision/deploy.py or inference service
import logging
import time
from google.cloud import logging as cloud_logging

def log_prediction(image_path, prediction, confidence, metadata):
    logging_client = cloud_logging.Client()
    logger = logging_client.logger("vision_predictions")
    
    # Calculate execution time
    start_time = time.time()
    # ... inference happens here ...
    execution_time = time.time() - start_time
    
    logger.log_struct({
        "prediction": prediction,
        "confidence": confidence,
        "latency": execution_time,  # Execution time in seconds (matches RAG format)
        "latency_ms": execution_time * 1000,  # Also log in milliseconds
        "image_metadata": metadata,
        "timestamp": datetime.utcnow().isoformat()
    })
```

---

## Recent Enhancements (Latest Update)

### RAG Component Improvements:

1. **Production Quality Metrics Tracking** ✅
   - Added composite score calculation from production inference
   - Tracks hallucination scores and retrieval scores per prediction
   - Logs all metrics to Cloud Logging via `deployment/RAG_serve.py`
   - Metrics collected and displayed in `Monitoring/RAG/run_monitoring.py`

2. **Embedding Space Coverage Monitoring** ✅
   - Tracks which documents are being retrieved from the knowledge base
   - Calculates percentage of index utilization
   - Detects if model is over-relying on a small subset of documents
   - Used in retraining decisions to identify knowledge base utilization issues

3. **Enhanced Retraining Strategy** ✅
   - `determine_retraining_strategy()` now uses production composite scores
   - Checks both average and minimum composite scores against thresholds
   - Uses embedding space coverage to detect bias in document retrieval
   - Provides more intelligent retraining decisions based on actual production performance

4. **Comprehensive Logging** ✅
   - `RAG_serve.py` now logs: composite_score, hallucination_score, avg_retrieval_score, retrieved_doc_indices, index_size
   - All metrics stored in Cloud Logging for monitoring collection
   - Privacy-preserving (no query/response text in logs)

## Conclusion

The **RAG component** has comprehensive monitoring and retraining infrastructure (26/28 requirements met) with **recent enhancements for production quality tracking**. The **Vision component** has strong data pipeline and training infrastructure but lacks critical production monitoring capabilities (19/28 requirements met).

**Key Strengths:**
- ✅ Excellent threshold and configuration management (both components)
- ✅ Comprehensive alerting infrastructure (both components)
- ✅ Strong data pipeline validation and drift detection (both components)
- ✅ RAG has complete monitoring-to-retraining pipeline
- ✅ **NEW**: RAG now tracks production quality metrics (composite scores, hallucination, retrieval) from actual inference
- ✅ **NEW**: RAG now monitors embedding space coverage to detect knowledge base utilization issues
- ✅ **NEW**: RAG retraining decisions now use production metrics for intelligent strategy selection

**Critical Gaps:**
- ❌ Vision missing production inference logging
- ❌ Vision missing real-time monitoring script
- ❌ Vision missing monitoring-to-training trigger connection

**Overall Readiness**: 84% (47/56 requirements met)

**RAG Component Readiness**: ~93% (26/28 requirements met, with recent enhancements improving production monitoring quality)

With the immediate priority fixes for Vision, the system will achieve **~95% readiness** for production model monitoring and automated retraining.

