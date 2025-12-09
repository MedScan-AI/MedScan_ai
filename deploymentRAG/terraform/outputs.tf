output "dashboard_url" {
  description = "URL to the monitoring dashboard"
  value       = var.enable_monitoring ? "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.rag_dashboard[0].id}?project=${var.project_id}" : "Monitoring disabled"
}

output "alert_policy_ids" {
  description = "IDs of created alert policies"
  value = var.enable_monitoring ? {
    high_error_rate          = google_monitoring_alert_policy.high_error_rate[0].id
    high_latency             = google_monitoring_alert_policy.high_latency[0].id
    service_unavailable      = google_monitoring_alert_policy.service_unavailable[0].id
    high_cpu                 = google_monitoring_alert_policy.high_cpu[0].id
    high_memory              = google_monitoring_alert_policy.high_memory[0].id
    low_composite_score      = google_monitoring_alert_policy.low_composite_score[0].id
    critical_composite_score = google_monitoring_alert_policy.critical_composite_score[0].id
    low_hallucination_score  = google_monitoring_alert_policy.low_hallucination_score[0].id
    low_retrieval_quality    = google_monitoring_alert_policy.low_retrieval_quality[0].id
    low_quality_spike        = google_monitoring_alert_policy.low_quality_spike[0].id
    retraining_triggered     = google_monitoring_alert_policy.retraining_triggered[0].id
  } : {}
}

output "custom_metrics" {
  description = "Names of created custom log-based metrics"
  value = var.enable_monitoring ? [
    google_logging_metric.rag_composite_score[0].name,
    google_logging_metric.rag_hallucination_score[0].name,
    google_logging_metric.rag_retrieval_score[0].name,
    google_logging_metric.rag_low_composite_score[0].name,
    google_logging_metric.rag_tokens_used[0].name,
    google_logging_metric.rag_docs_retrieved[0].name,
    google_logging_metric.rag_retraining_triggered[0].name
  ] : []
}

