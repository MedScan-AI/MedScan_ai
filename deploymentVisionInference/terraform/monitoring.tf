################################################################################
# Google Cloud Monitoring for Vision Inference API
################################################################################

# Enable Monitoring API (optional)
resource "google_project_service" "monitoring_api" {
  count            = var.enable_apis ? 1 : 0
  service          = "monitoring.googleapis.com"
  disable_on_destroy = false
}

# Notification channel for alerts (email)
resource "google_monitoring_notification_channel" "email" {
  count        = var.monitoring_email != "" ? 1 : 0
  display_name = "Vision Inference API - Email Alerts"
  type         = "email"

  labels = {
    email_address = var.monitoring_email
  }

  # depends_on removed - Terraform will infer dependencies
}

# Alert Policy: High Error Rate
resource "google_monitoring_alert_policy" "high_error_rate" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Vision Inference API - High Error Rate"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Error rate > 5%"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields    = ["resource.label.service_name"]
      }

      denominator_filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_count\""
      denominator_aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields    = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = var.monitoring_email != "" ? [google_monitoring_notification_channel.email[0].id] : []

  # depends_on removed - Terraform will infer dependencies
}

# Alert Policy: High Latency (P95 > 5 seconds)
resource "google_monitoring_alert_policy" "high_latency" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Vision Inference API - High Latency (P95 > 5s)"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "P95 latency > 5 seconds"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5000

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
        group_by_fields    = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = var.monitoring_email != "" ? [google_monitoring_notification_channel.email[0].id] : []

  # depends_on removed - Terraform will infer dependencies
}

# Alert Policy: Service Unavailable (No requests in 5 minutes)
resource "google_monitoring_alert_policy" "service_unavailable" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Vision Inference API - Service Unavailable"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "No requests in 5 minutes (service may be down)"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_count\""
      duration        = "300s"
      comparison      = "COMPARISON_LT"
      threshold_value = 1

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields    = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = var.monitoring_email != "" ? [google_monitoring_notification_channel.email[0].id] : []

  # depends_on removed - Terraform will infer dependencies
}

# Alert Policy: High CPU Usage
resource "google_monitoring_alert_policy" "high_cpu" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Vision Inference API - High CPU Usage"
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
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields    = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = var.monitoring_email != "" ? [google_monitoring_notification_channel.email[0].id] : []

  # depends_on removed - Terraform will infer dependencies
}

# Alert Policy: High Memory Usage
resource "google_monitoring_alert_policy" "high_memory" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Vision Inference API - High Memory Usage"
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
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields    = ["resource.label.service_name"]
      }
    }
  }

  notification_channels = var.monitoring_email != "" ? [google_monitoring_notification_channel.email[0].id] : []

  # depends_on removed - Terraform will infer dependencies
}

# Monitoring Dashboard
resource "google_monitoring_dashboard" "vision_inference_dashboard" {
  count        = var.enable_monitoring ? 1 : 0
  dashboard_json = jsonencode({
    displayName = "Vision Inference API - Monitoring Dashboard"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
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
                      alignmentPeriod     = "60s"
                      perSeriesAligner    = "ALIGN_RATE"
                      crossSeriesReducer  = "REDUCE_SUM"
                      groupByFields       = ["metric.label.response_code_class"]
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
                      alignmentPeriod     = "60s"
                      perSeriesAligner    = "ALIGN_RATE"
                      crossSeriesReducer  = "REDUCE_SUM"
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
                        alignmentPeriod     = "60s"
                        perSeriesAligner    = "ALIGN_DELTA"
                        crossSeriesReducer  = "REDUCE_PERCENTILE_50"
                      }
                    }
                  }
                  plotType = "LINE"
                  legendTemplate = "P50"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
                      aggregation = {
                        alignmentPeriod     = "60s"
                        perSeriesAligner    = "ALIGN_DELTA"
                        crossSeriesReducer  = "REDUCE_PERCENTILE_95"
                      }
                    }
                  }
                  plotType = "LINE"
                  legendTemplate = "P95"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
                      aggregation = {
                        alignmentPeriod     = "60s"
                        perSeriesAligner    = "ALIGN_DELTA"
                        crossSeriesReducer  = "REDUCE_PERCENTILE_99"
                      }
                    }
                  }
                  plotType = "LINE"
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
                      alignmentPeriod     = "60s"
                      perSeriesAligner    = "ALIGN_MEAN"
                      crossSeriesReducer  = "REDUCE_MEAN"
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
                      alignmentPeriod     = "60s"
                      perSeriesAligner    = "ALIGN_MEAN"
                      crossSeriesReducer  = "REDUCE_MEAN"
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
                      alignmentPeriod     = "60s"
                      perSeriesAligner    = "ALIGN_MEAN"
                      crossSeriesReducer  = "REDUCE_SUM"
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
        }
      ]
    }
  })

  # depends_on removed - Terraform will infer dependencies
}
