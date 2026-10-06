# Every variable needs a description.

variable "project" {
  description = "Project name, used as the first element of every resource name"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "availability_zone" {
  description = "Availability Zone of the private subnet the Glue workers run in"
  type        = string
  default     = "us-east-1a"
}

variable "vpc_id" {
  description = "VPC the Glue security group is created in"
  type        = string
}

variable "private_subnet_id" {
  description = "Private subnet the Glue NETWORK connection places job workers in"
  type        = string
}

variable "bucket_name" {
  description = "Name of the data bucket (raw/, processed/, features/, artifacts/)"
  type        = string
}

variable "data_engineer_role_arn" {
  description = "ARN of the DataEngineer role the crawler and jobs run as"
  type        = string
}

variable "feature_group_name" {
  description = "SageMaker Feature Group the feature engineering job ingests into"
  type        = string
}

variable "glue_version" {
  description = "AWS Glue runtime version for the ETL jobs"
  type        = string
  default     = "4.0"
}

variable "worker_type" {
  description = "Glue worker type for the ETL jobs"
  type        = string
  default     = "G.1X"
}

variable "number_of_workers" {
  description = "Number of Glue workers per job run"
  type        = number
  default     = 2
}
