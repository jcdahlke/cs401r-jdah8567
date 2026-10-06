# ── modules/glue ─────────────────────────────────────────────────────────────
# Lab 2 data pipeline:
#
#   aws_glue_catalog_database   catalog the crawler registers tables in
#   aws_glue_crawler            scans raw/customers/ -> table "customers"
#   aws_security_group          self-referencing, required by Glue in a VPC
#   aws_glue_connection         NETWORK connection into the private subnet
#   aws_s3_object x2            job scripts uploaded to artifacts/glue/
#   aws_glue_job x2             transform and feature-engineer
#
# Every name is built from var.project and var.environment.

locals {
  name_prefix = "${var.project}-${var.environment}"
  # Glue catalog names cannot contain hyphens: northstar_dev
  database_name = replace(local.name_prefix, "-", "_")
  scripts_dir   = "${path.module}/../../../glue-scripts"
  script_prefix = "artifacts/glue"
}

data "aws_region" "current" {}

# ── Catalog and crawler ──────────────────────────────────────────────────────

resource "aws_glue_catalog_database" "this" {
  name        = local.database_name
  description = "NorthStar data catalog - tables discovered by the raw crawler"
}

resource "aws_glue_crawler" "raw" {
  name          = "${local.name_prefix}-raw-crawler"
  role          = var.data_engineer_role_arn
  database_name = aws_glue_catalog_database.this.name
  description   = "Discovers the schema of raw/customers/ CSV files"

  # No table_prefix, so the table is named after the prefix: "customers".
  s3_target {
    path = "s3://${var.bucket_name}/raw/customers/"
  }

  schema_change_policy {
    update_behavior = "UPDATE_IN_DATABASE"
    delete_behavior = "LOG"
  }
}

# ── Networking for Glue inside the VPC ───────────────────────────────────────

resource "aws_security_group" "glue" {
  name        = "${local.name_prefix}-glue-sg"
  description = "Glue job workers - self-referencing ingress required by Glue"
  vpc_id      = var.vpc_id

  # Glue requires an all-ports ingress rule whose source is this same
  # security group. A VPC CIDR rule does not satisfy the check.
  ingress {
    description = "All traffic between Glue workers"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    description = "All outbound traffic (S3, Feature Store, AWS APIs via NAT)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-glue-sg" }
}

resource "aws_glue_connection" "vpc" {
  name            = "${local.name_prefix}-glue-vpc"
  description     = "Places Glue job workers in the private subnet"
  connection_type = "NETWORK"

  physical_connection_requirements {
    availability_zone      = var.availability_zone
    subnet_id              = var.private_subnet_id
    security_group_id_list = [aws_security_group.glue.id]
  }
}

# ── Job scripts ──────────────────────────────────────────────────────────────

resource "aws_s3_object" "transform_script" {
  bucket = var.bucket_name
  key    = "${local.script_prefix}/transform.py"
  source = "${local.scripts_dir}/transform.py"
  etag   = filemd5("${local.scripts_dir}/transform.py")
}

resource "aws_s3_object" "feature_engineer_script" {
  bucket = var.bucket_name
  key    = "${local.script_prefix}/feature_engineer.py"
  source = "${local.scripts_dir}/feature_engineer.py"
  etag   = filemd5("${local.scripts_dir}/feature_engineer.py")
}

# ── ETL jobs ─────────────────────────────────────────────────────────────────

resource "aws_glue_job" "transform" {
  name              = "${local.name_prefix}-transform"
  description       = "raw/customers/ CSV -> processed/customers/ Parquet"
  role_arn          = var.data_engineer_role_arn
  glue_version      = var.glue_version
  worker_type       = var.worker_type
  number_of_workers = var.number_of_workers
  connections       = [aws_glue_connection.vpc.name]

  command {
    name            = "glueetl"
    python_version  = "3"
    script_location = "s3://${var.bucket_name}/${aws_s3_object.transform_script.key}"
  }

  # Must match getResolvedOptions(...) in glue-scripts/transform.py
  default_arguments = {
    "--job-language"                     = "python"
    "--enable-continuous-cloudwatch-log" = "true"
    "--database_name"                    = aws_glue_catalog_database.this.name
    "--table_name"                       = "customers"
    "--output_path"                      = "s3://${var.bucket_name}/processed/customers/"
  }
}

resource "aws_glue_job" "feature_engineer" {
  name              = "${local.name_prefix}-feature-engineer"
  description       = "processed/customers/ -> features/customers/ Parquet and Feature Store"
  role_arn          = var.data_engineer_role_arn
  glue_version      = var.glue_version
  worker_type       = var.worker_type
  number_of_workers = var.number_of_workers
  connections       = [aws_glue_connection.vpc.name]

  command {
    name            = "glueetl"
    python_version  = "3"
    script_location = "s3://${var.bucket_name}/${aws_s3_object.feature_engineer_script.key}"
  }

  # Must match getResolvedOptions(...) in glue-scripts/feature_engineer.py
  default_arguments = {
    "--job-language"                     = "python"
    "--enable-continuous-cloudwatch-log" = "true"
    "--input_path"                       = "s3://${var.bucket_name}/processed/customers/"
    "--output_path"                      = "s3://${var.bucket_name}/features/customers/"
    "--feature_group_name"               = var.feature_group_name
    "--region"                           = data.aws_region.current.name
  }
}
