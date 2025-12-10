# RAG Service Monitoring - Terraform Configuration

This Terraform configuration sets up comprehensive monitoring for the RAG service (`rag-service`) on Cloud Run, including custom log-based metrics, dashboards, and alert policies.

## Overview

This configuration creates:

- **7 Custom Log-Based Metrics**: Track RAG-specific quality metrics from prediction logs
- **Monitoring Dashboard**: 11 widgets showing Cloud Run production metrics + RAG quality metrics
- **11 Alert Policies**: 5 for Cloud Run production issues + 5 for RAG quality degradation + 1 for retraining
- **Email Notification Channel**: For alert notifications

## Prerequisites

1. **GCP Project**: `medscanai-476500` (or update `project_id` variable)
2. **Cloud Run Service**: `rag-service` must exist and be logging prediction metrics
3. **Permissions**: 
   - `roles/monitoring.metricWriter` (for creating custom metrics)
   - `roles/monitoring.dashboardEditor` (for creating dashboards)
   - `roles/monitoring.alertPolicyEditor` (for creating alert policies)
   - `roles/monitoring.notificationChannelEditor` (for creating notification channels)
4. **Terraform**: Version >= 1.0
5. **Google Provider**: Version ~> 5.0

## Quick Start

1. **Copy example variables file**:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. **Edit `terraform.tfvars`**:
   ```hcl
   project_id = "medscanai-476500"
   region = "us-central1"
   service_name = "rag-service"
   monitoring_email = "your-email@example.com"
   ```

3. **Initialize Terraform**:
   ```bash
   terraform init
   ```

4. **Review changes**:
   ```bash
   terraform plan
   ```

5. **Apply configuration**:
   ```bash
   terraform apply
   ```

## Custom Log-Based Metrics

The following metrics are created from RAG prediction logs:

1. **rag_composite_score** (GAUGE): Composite quality score (0-1, higher is better)
2. **rag_hallucination_score** (GAUGE): Hallucination score (0-1, higher is better - lower means more hallucinations)
3. **rag_retrieval_score** (GAUGE): Average retrieval score (0-1, higher is better)
4. **rag_low_composite_score** (DELTA): Count of low-quality predictions (composite_score < 0.25)
5. **rag_tokens_used** (DELTA): Total tokens consumed per prediction
6. **rag_docs_retrieved** (GAUGE): Number of documents retrieved per query
7. **rag_retraining_triggered** (DELTA): Count of retraining events triggered by monitoring

### Retraining Event Logging

The monitoring code (`Monitoring/RAG/rag_monitor.py`) logs retraining events to Cloud Logging when retraining is triggered. The log format is:

```json
{
  "retraining_event": {
    "triggered": true,
    "strategy": "full" | "model_only",
    "reason": "Reason for retraining",
    "timestamp": "2024-01-01T12:00:00Z",
    "project_id": "medscanai-476500"
  }
}
```

These logs are written to the `rag_monitoring` log and trigger the retraining alert.

### Log Format Expected

The RAG service must log predictions in the following format:

```json
{
  "prediction_result": {
    "success": true,
    "latency": 1.234,
    "composite_score": 0.75,
    "hallucination_score": 0.85,
    "avg_retrieval_score": 0.70,
    "num_retrieved_docs": 5,
    "total_tokens": 1500,
    "index_size": 1000,
    "retrieved_doc_indices": [1, 2, 3, 4, 5]
  }
}
```

## Dashboard Widgets

The dashboard includes 11 widgets organized in 5 rows:

### Row 1-2: Cloud Run Production Metrics (6 widgets)
1. Request Rate
2. Error Rate
3. Request Latency (P50, P95, P99)
4. CPU Utilization
5. Memory Utilization
6. Instance Count

### Row 3: RAG Quality Metrics
7. Composite Score Distribution (Mean, P50, P95 with threshold lines at 0.25 and 0.3)
8. Answer Groundedness Trend (Mean, P50, P95 with threshold line at 0.2)

### Row 4: RAG Quality Metrics (continued)
9. Retrieval Score Trend (Mean with threshold line at 0.6)
10. Low Quality Predictions Count

### Row 5: RAG Operational Metrics
11. Tokens Usage

## Alert Policies

**Total: 11 alert policies** (5 production + 5 quality + 1 retraining)

### Cloud Run Production Alerts (5 alerts)

1. **High Error Rate**: Error rate > 15% for 5 minutes
2. **High Latency**: P95 latency > 10 seconds for 5 minutes
3. **Service Unavailable**: No requests in 5 hours
4. **High CPU**: CPU > 80% for 5 minutes
5. **High Memory**: Memory > 85% for 5 minutes

### RAG Quality Alerts (5 alerts)

6. **Low Composite Score (Average)**: Avg composite score < 0.3 for 15 minutes
7. **Critical Composite Score Drop**: Min composite score < 0.25 for 5 minutes
8. **Low Hallucination Score**: Avg hallucination score < 0.2 for 15 minutes
   - **Note**: Hallucination score is inverted - lower is worse (more hallucinations)
9. **Low Retrieval Quality**: Avg retrieval score < 0.6 for 15 minutes
10. **Low Quality Prediction Spike**: > 10 low-quality predictions in 5 minutes

### RAG Retraining Alert (1 alert)

11. **Retraining Triggered**: Alert fires immediately when retraining pipeline is triggered
   - Monitors `rag_monitoring` log for retraining events
   - Includes strategy (full/model_only) and reason in alert details
   - Check Cloud Logging for full retraining event details

### Alert Thresholds

Thresholds match `Monitoring/RAG/rag_monitor.py`:
- `max_error_rate`: 0.15 (15%)
- `max_latency_p95`: 10.0 seconds
- `min_relevance_score`: 0.6
- `max_hallucination_score`: 0.2 (minimum acceptable)
- `min_composite_score`: 0.25
- `min_avg_composite_score`: 0.3

## Important Notes

### Hallucination Score Logic

**CRITICAL**: Hallucination score is inverted - higher is better, lower is worse.

- **Threshold**: 0.2 (minimum acceptable)
- **Alert triggers when**: `hallucination_score < 0.2`
- **Alert comparison**: `COMPARISON_LT` (less than)
- **Dashboard**: Shows threshold line at 0.2, values below are bad

### Lifecycle Protection

The configuration uses a `null_resource` with a destroy provisioner to ensure proper cleanup order. When resources are destroyed, the `null_resource.delete_alert_policies_before_metrics` resource ensures alert policies are deleted before custom metrics to prevent dependency issues.

To destroy resources:

1. Run `terraform destroy`
2. The destroy provisioner will automatically handle cleanup order
3. Alert policies will be deleted before metrics to avoid dependency errors

### Importing Existing Resources

If alert policies or dashboards already exist, import them before running `terraform apply`:

```bash
# Find existing alert policy IDs
gcloud alpha monitoring policies list --project=medscanai-476500 --format="table(name,displayName)"

# Import each alert policy (replace POLICY_ID)
terraform import 'google_monitoring_alert_policy.high_error_rate[0]' projects/medscanai-476500/alertPolicies/POLICY_ID

# Find existing dashboard ID
gcloud monitoring dashboards list --project=medscanai-476500 --format="table(name,displayName)"

# Import dashboard (replace DASHBOARD_ID)
terraform import 'google_monitoring_dashboard.rag_dashboard[0]' projects/medscanai-476500/dashboards/DASHBOARD_ID
```

## Outputs

After applying, Terraform will output:

- `dashboard_url`: Direct link to the monitoring dashboard
- `alert_policy_ids`: Map of alert policy IDs
- `custom_metrics`: List of custom metric names

## Troubleshooting

### Metrics Not Appearing

1. **Verify logs are being written**: Check Cloud Logging for `rag-service` logs
2. **Check log format**: Ensure logs match the expected format above
3. **Wait for metric collection**: Custom metrics may take 5-10 minutes to appear
4. **Verify filters**: Check that log filters match your actual log structure

### Alerts Not Triggering

1. **Check alert policy status**: Verify alerts are enabled in GCP Console
2. **Verify notification channel**: Ensure email notification channel is created
3. **Test thresholds**: Manually trigger conditions to test alerts
4. **Check metric data**: Ensure metrics have data points

### Dashboard Not Loading

1. **Verify dashboard exists**: Check GCP Console > Monitoring > Dashboards
2. **Check permissions**: Ensure you have `monitoring.dashboards.get` permission
3. **Wait for metrics**: Dashboard needs metric data to display

## Reference Files

- `Monitoring/RAG/rag_monitor.py` - Existing monitoring logic and thresholds
- `deployment/RAG_serve.py` - Logging format (lines 653-740)
- `deploymentVisionInference/terraform/monitoring.tf` - Reference implementation

## Maintenance

### Updating Thresholds

To update alert thresholds:

1. Edit `monitoring.tf`
2. Update threshold values in alert policy conditions
3. Run `terraform plan` to review changes
4. Run `terraform apply` to update

### Adding New Metrics

To add new custom metrics:

1. Add `google_logging_metric` resource in `monitoring.tf`
2. Add corresponding dashboard widget (if needed)
3. Add alert policy (if needed)
4. Update `outputs.tf` to include new metric
5. Run `terraform apply`

## Support

For issues or questions:
1. Check existing monitoring code: `Monitoring/RAG/rag_monitor.py`
2. Review GCP Monitoring documentation
3. Check Terraform Google Provider documentation

