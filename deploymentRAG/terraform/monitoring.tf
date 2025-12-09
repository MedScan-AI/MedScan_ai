################################################################################
# Google Cloud Monitoring for RAG Service
################################################################################

# IMPORTANT: Preventing Recreation of Existing Monitoring Resources
#
# If alert policies or dashboards already exist, you MUST import them into
# Terraform state to prevent recreation. All monitoring resources have
# lifecycle { prevent_destroy = true } to protect against accidental deletion.
#
# To import existing monitoring resources:
#
# 1. Find existing alert policy IDs:
#    gcloud alpha monitoring policies list --project=YOUR_PROJECT_ID --format="table(name,displayName)"
#
# 2. Import each alert policy (replace POLICY_ID with actual ID):
#    terraform import 'google_monitoring_alert_policy.high_error_rate[0]' projects/PROJECT_ID/alertPolicies/POLICY_ID
#
# 3. Find existing dashboard ID:
#    gcloud monitoring dashboards list --project=YOUR_PROJECT_ID --format="table(name,displayName)"
#
# 4. Import the dashboard (replace DASHBOARD_ID with actual ID):
#    terraform import 'google_monitoring_dashboard.rag_dashboard[0]' projects/PROJECT_ID/dashboards/DASHBOARD_ID
#
################################################################################

# Notification channel for alerts (email)
resource "google_monitoring_notification_channel" "email" {
  count        = (var.monitoring_email != "" && var.create_notification_channel) ? 1 : 0
  display_name = "RAG Service - Email Alerts"
  type         = "email"

  labels = {
    email_address = var.monitoring_email
  }
}

################################################################################
# Custom Log-Based Metrics
################################################################################

# Metric 1: RAG Composite Score (DISTRIBUTION for numeric extraction)
resource "google_logging_metric" "rag_composite_score" {
  count       = var.enable_monitoring ? 1 : 0
  name        = "rag_composite_score"
  description = "Composite score from RAG predictions (0-1, higher is better)"

  filter = join(" ", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.service_name}\"",
    "jsonPayload.prediction_result.composite_score:*"
  ])

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "1"
  }

  value_extractor = "EXTRACT(jsonPayload.prediction_result.composite_score)"

  bucket_options {
    linear_buckets {
      num_finite_buckets = 100
      width              = 0.01
      offset             = 0.0
    }
  }
}

# Metric 2: RAG Hallucination Score (DISTRIBUTION for numeric extraction)
# NOTE: Higher is better, lower is worse (inverted logic)
resource "google_logging_metric" "rag_hallucination_score" {
  count       = var.enable_monitoring ? 1 : 0
  name        = "rag_hallucination_score"
  description = "Hallucination score from RAG predictions (0-1, higher is better - lower means more hallucinations)"

  filter = join(" ", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.service_name}\"",
    "jsonPayload.prediction_result.hallucination_score:*"
  ])

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "1"
  }

  value_extractor = "EXTRACT(jsonPayload.prediction_result.hallucination_score)"

  bucket_options {
    linear_buckets {
      num_finite_buckets = 100
      width              = 0.01
      offset             = 0.0
    }
  }
}

# Metric 3: RAG Retrieval Score (DISTRIBUTION for numeric extraction)
resource "google_logging_metric" "rag_retrieval_score" {
  count       = var.enable_monitoring ? 1 : 0
  name        = "rag_retrieval_score"
  description = "Average retrieval score from RAG predictions (0-1, higher is better)"

  filter = join(" ", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.service_name}\"",
    "jsonPayload.prediction_result.avg_retrieval_score:*"
  ])

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "1"
  }

  value_extractor = "EXTRACT(jsonPayload.prediction_result.avg_retrieval_score)"

  bucket_options {
    linear_buckets {
      num_finite_buckets = 100
      width              = 0.01
      offset             = 0.0
    }
  }
}

# Metric 4: RAG Low Composite Score (DELTA - count)
resource "google_logging_metric" "rag_low_composite_score" {
  count       = var.enable_monitoring ? 1 : 0
  name        = "rag_low_composite_score"
  description = "Count of low-quality predictions (composite_score < 0.25)"

  filter = join(" ", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.service_name}\"",
    "jsonPayload.prediction_result.composite_score < 0.25"
  ])

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

# Metric 5: RAG Tokens Used (DISTRIBUTION for numeric extraction)
resource "google_logging_metric" "rag_tokens_used" {
  count       = var.enable_monitoring ? 1 : 0
  name        = "rag_tokens_used"
  description = "Total tokens used in RAG predictions"

  filter = join(" ", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.service_name}\"",
    "jsonPayload.prediction_result.total_tokens:*"
  ])

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "1"
  }

  value_extractor = "EXTRACT(jsonPayload.prediction_result.total_tokens)"

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 50
      growth_factor      = 1.1
      scale              = 1.0
    }
  }
}

# Metric 6: RAG Documents Retrieved (DISTRIBUTION for numeric extraction)
resource "google_logging_metric" "rag_docs_retrieved" {
  count       = var.enable_monitoring ? 1 : 0
  name        = "rag_docs_retrieved"
  description = "Number of documents retrieved per query"

  filter = join(" ", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.service_name}\"",
    "jsonPayload.prediction_result.num_retrieved_docs:*"
  ])

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "1"
  }

  value_extractor = "EXTRACT(jsonPayload.prediction_result.num_retrieved_docs)"

  bucket_options {
    linear_buckets {
      num_finite_buckets = 50
      width              = 1.0
      offset             = 0.0
    }
  }
}

# Metric 7: RAG Retraining Triggered (DELTA - count)
resource "google_logging_metric" "rag_retraining_triggered" {
  count       = var.enable_monitoring ? 1 : 0
  name        = "rag_retraining_triggered"
  description = "Count of retraining events triggered by monitoring"

  filter = join(" ", [
    "logName=\"projects/${var.project_id}/logs/rag_monitoring\"",
    "jsonPayload.retraining_event.triggered=true"
  ])

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

################################################################################
# Cloud Run Production Alerts
################################################################################

# Alert Policy: High Error Rate
resource "google_monitoring_alert_policy" "high_error_rate" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - High Error Rate"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Error rate > 15%"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.15

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }

      denominator_filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_count\""
      denominator_aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: High Latency (P95 > 10 seconds)
resource "google_monitoring_alert_policy" "high_latency" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - High Latency (P95 > 10s)"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "P95 latency > 10 seconds"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10000

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: Service Unavailable (No requests in 5 hours)
resource "google_monitoring_alert_policy" "service_unavailable" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - Service Unavailable (5h window)"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "No requests in 5 hours (service may be down)"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_count\""
      duration        = "18000s" # 5 hours
      comparison      = "COMPARISON_LT"
      threshold_value = 1

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: High CPU Usage
resource "google_monitoring_alert_policy" "high_cpu" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - High CPU Usage"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "CPU utilization > 80%"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_50"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: High Memory Usage
resource "google_monitoring_alert_policy" "high_memory" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - High Memory Usage"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Memory utilization > 85%"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.85

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_50"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

################################################################################
# RAG Quality Alerts
################################################################################

# Alert Policy: Low Composite Score (Average)
resource "google_monitoring_alert_policy" "low_composite_score" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - Low Composite Score (Average)"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Avg composite score < 0.3 for 15 minutes"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/rag_composite_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
      duration        = "900s" # 15 minutes
      comparison      = "COMPARISON_LT"
      threshold_value = 0.3

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: Critical Composite Score Drop
resource "google_monitoring_alert_policy" "critical_composite_score" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - Critical Composite Score Drop"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Min composite score < 0.25 for 5 minutes"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/rag_composite_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
      duration        = "300s" # 5 minutes
      comparison      = "COMPARISON_LT"
      threshold_value = 0.25

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: Low Hallucination Score Alert (CORRECTED - lower is worse)
resource "google_monitoring_alert_policy" "low_hallucination_score" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - Low Hallucination Score"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Avg hallucination score < 0.2 for 15 minutes (model generating more hallucinations)"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/rag_hallucination_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
      duration        = "900s"          # 15 minutes
      comparison      = "COMPARISON_LT" # Less than - lower is worse
      threshold_value = 0.2

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: Low Retrieval Quality
resource "google_monitoring_alert_policy" "low_retrieval_quality" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - Low Retrieval Quality"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Avg retrieval score < 0.6 for 15 minutes"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/rag_retrieval_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
      duration        = "900s" # 15 minutes
      comparison      = "COMPARISON_LT"
      threshold_value = 0.6

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: Low Quality Prediction Spike
resource "google_monitoring_alert_policy" "low_quality_spike" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - Low Quality Prediction Spike"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "> 10 low-quality predictions in 5 minutes"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/rag_low_composite_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
      duration        = "300s" # 5 minutes
      comparison      = "COMPARISON_GT"
      threshold_value = 10

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

# Alert Policy: Retraining Triggered
resource "google_monitoring_alert_policy" "retraining_triggered" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "RAG Service - Retraining Triggered"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Retraining pipeline triggered"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/rag_retraining_triggered\" AND resource.type=\"global\""
      duration        = "60s" # Alert immediately when retraining is triggered
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = (var.monitoring_email != "" && var.create_notification_channel) ? [google_monitoring_notification_channel.email[0].id] : []

  documentation {
    content   = "RAG model retraining has been triggered by the monitoring system. Check the retraining_event log for strategy (full/model_only) and reason."
    mime_type = "text/markdown"
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [enabled]
  }
}

################################################################################
# Monitoring Dashboard
################################################################################

resource "google_monitoring_dashboard" "rag_dashboard" {
  count = var.enable_monitoring ? 1 : 0
  dashboard_json = jsonencode({
    displayName = "RAG Service - Monitoring Dashboard"
    mosaicLayout = {
      columns = 12
      tiles = [
        # Row 1-2: Cloud Run Production Metrics (6 widgets)
        {
          xPos   = 0
          yPos   = 0
          width  = 6
          height = 4
          widget = {
            title = "Request Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_count\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.response_code_class"]
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Requests/sec"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 0
          width  = 6
          height = 4
          widget = {
            title = "Error Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Errors/sec"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 0
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Request Latency (P50, P95, P99)"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_50"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "P50"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_95"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "P95"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_99"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "P99"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Latency (ms)"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "CPU Utilization"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_50"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "CPU %"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 0
          yPos   = 8
          width  = 6
          height = 4
          widget = {
            title = "Memory Utilization"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_50"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Memory %"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 8
          width  = 6
          height = 4
          widget = {
            title = "Instance Count"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/container/instance_count\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_MEAN"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Instances"
                scale = "LINEAR"
              }
            }
          }
        },
        # Row 3: RAG Quality Metrics
        {
          xPos   = 0
          yPos   = 12
          width  = 6
          height = 4
          widget = {
            title = "Composite Score Distribution"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"logging.googleapis.com/user/rag_composite_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_MEAN"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "Mean"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"logging.googleapis.com/user/rag_composite_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_50"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "P50"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"logging.googleapis.com/user/rag_composite_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_95"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "P95"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Composite Score (0-1)"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 12
          width  = 6
          height = 4
          widget = {
            title = "Answer Groundedness Trend"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"logging.googleapis.com/user/rag_hallucination_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_MEAN"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "Mean"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"logging.googleapis.com/user/rag_hallucination_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_50"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "P50"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"logging.googleapis.com/user/rag_hallucination_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_95"
                      }
                    }
                  }
                  plotType       = "LINE"
                  legendTemplate = "P95"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Hallucination Score (higher is better)"
                scale = "LINEAR"
              }
            }
          }
        },
        # Row 4: RAG Quality Metrics (continued)
        {
          xPos   = 0
          yPos   = 16
          width  = 6
          height = 4
          widget = {
            title = "Retrieval Score Trend"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/rag_retrieval_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_DELTA"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Retrieval Score (0-1)"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 16
          width  = 6
          height = 4
          widget = {
            title = "Low Quality Predictions Count"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/rag_low_composite_score\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_DELTA"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Count"
                scale = "LINEAR"
              }
            }
          }
        },
        # Row 5: RAG Operational Metrics
        {
          xPos   = 0
          yPos   = 20
          width  = 6
          height = 4
          widget = {
            title = "Tokens Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/rag_tokens_used\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_DELTA"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Tokens"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })

  lifecycle {
    prevent_destroy = true
  }
}

