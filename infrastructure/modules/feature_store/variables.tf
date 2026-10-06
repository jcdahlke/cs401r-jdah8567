# Every variable needs a description.

variable "project" {
  description = "Project name, used as the first element of every resource name"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "bucket_name" {
  description = "Name of the data bucket that backs the offline store"
  type        = string
}

variable "offline_store_prefix" {
  description = "S3 prefix for the offline store; kept apart from the job's features/customers/ output"
  type        = string
  default     = "features/offline-store/"
}

variable "role_arn" {
  description = "Execution role for the Feature Group (the DataEngineer role ARN)"
  type        = string
}

variable "enable_online_store" {
  description = "Enable the low-latency online store for real-time inference in Labs 3-4"
  type        = bool
  default     = true
}
