# ── modules/feature_store ────────────────────────────────────────────────────
# Lab 2 Task 3:
#
#   aws_sagemaker_feature_group   customer-level churn features
#
# 16 definitions: 2 keys (customer_id, event_time), 13 features, 1 label.
# event_time MUST be Fractional (Unix epoch seconds). A String event time
# paired with a numeric value makes PutRecord "succeed" while silently
# dropping every record.

locals {
  name_prefix = "${var.project}-${var.environment}"

  # Order is preserved in the feature group definition.
  feature_definitions = [
    { name = "customer_id", type = "String" },
    { name = "event_time", type = "Fractional" },
    { name = "days_since_last_purchase", type = "Fractional" },
    { name = "customer_tenure_days", type = "Fractional" },
    { name = "purchase_frequency_30d", type = "Fractional" },
    { name = "purchase_frequency_90d", type = "Fractional" },
    { name = "purchase_frequency_180d", type = "Fractional" },
    { name = "avg_order_value", type = "Fractional" },
    { name = "total_spend_90d", type = "Fractional" },
    { name = "total_lifetime_value", type = "Fractional" },
    { name = "avg_basket_size_6m", type = "Fractional" },
    { name = "category_diversity_score", type = "Fractional" },
    { name = "online_to_store_ratio", type = "Fractional" },
    { name = "loyalty_tier", type = "String" },
    { name = "churn_risk_score", type = "Fractional" },
    { name = "churn_label", type = "Integral" },
  ]
}

resource "aws_sagemaker_feature_group" "customer" {
  feature_group_name             = "${local.name_prefix}-customer-features"
  description                    = "Customer churn features from the observation window, label from the outcome window"
  record_identifier_feature_name = "customer_id"
  event_time_feature_name        = "event_time"
  role_arn                       = var.role_arn

  dynamic "feature_definition" {
    for_each = local.feature_definitions
    content {
      feature_name = feature_definition.value.name
      feature_type = feature_definition.value.type
    }
  }

  online_store_config {
    enable_online_store = var.enable_online_store
  }

  # The offline store builds its own <account>/sagemaker/<region>/offline-store/
  # tree under this prefix, so it must not share features/customers/ with the
  # feature engineering job's Parquet output.
  offline_store_config {
    disable_glue_table_creation = false

    s3_storage_config {
      s3_uri = "s3://${var.bucket_name}/${var.offline_store_prefix}"
    }
  }
}
