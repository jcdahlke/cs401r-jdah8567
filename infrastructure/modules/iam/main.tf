# ── modules/iam ──────────────────────────────────────────────────────────────
# Required resources (Task B1). Exactly one of each:
#
#   aws_iam_role                     MLEngineer, trusted by sagemaker.amazonaws.com
#   aws_iam_policy
#   aws_iam_role_policy_attachment
#
# Least privilege is graded in later labs, so start narrow: grant only the S3
# prefixes and SageMaker actions this role actually needs. A wildcard policy
# here will cost you points in Lab 2.

# TODO: implement the three resources above.

locals {
  name_prefix = "${var.project}-${var.environment}"
}

resource "aws_iam_role" "ml_engineer" {
  name = "${local.name_prefix}-MLEngineer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "ml_engineer" {
  name        = "${local.name_prefix}-MLEngineerPolicy"
  description = "Least-privilege permissions for the MLEngineer role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SageMakerCore"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob",
          "sagemaker:CreateEndpoint", "sagemaker:DescribeEndpoint", "sagemaker:DeleteEndpoint",
          "sagemaker:CreateEndpointConfig", "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateMlflowApp", "sagemaker:DescribeMlflowApp", "sagemaker:ListMlflowApps",
          "sagemaker:CreatePresignedMlflowAppUrl",
          "sagemaker:RegisterModel", "sagemaker:DescribeModelPackage", "sagemaker:ListModelPackages",
        ]
        Resource = "*"
      },
      {
        Sid    = "StudioSelfService"
        Effect = "Allow"
        Action = [
          "sagemaker:DescribeDomain", "sagemaker:ListDomains",
          "sagemaker:DescribeUserProfile", "sagemaker:ListUserProfiles",
          "sagemaker:DescribeSpace", "sagemaker:ListSpaces", "sagemaker:CreateSpace",
          "sagemaker:UpdateSpace", "sagemaker:DeleteSpace",
          "sagemaker:DescribeApp", "sagemaker:ListApps", "sagemaker:CreateApp", "sagemaker:DeleteApp",
          "sagemaker:CreatePresignedDomainUrl",
        ]
        Resource = [
          "arn:aws:sagemaker:*:*:domain/*", "arn:aws:sagemaker:*:*:user-profile/*",
          "arn:aws:sagemaker:*:*:space/*", "arn:aws:sagemaker:*:*:app/*",
        ]
      },
      {
        # Object actions ONLY on artifacts/ and features/. Do not widen this to
        # the bucket ARN: a trailing * would also match raw/ and processed/.
        Sid    = "S3ArtifactsAndFeatures"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = [
          "arn:aws:s3:::${local.name_prefix}-data-*/artifacts/*",
          "arn:aws:s3:::${local.name_prefix}-data-*/features/*",
        ]
      },
      {
        Sid      = "S3BucketList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = "arn:aws:s3:::${local.name_prefix}-data-*"
      },
      {
        Sid      = "CloudWatchLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      },
      {
        Sid      = "ECRRead"
        Effect   = "Allow"
        Action   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:GetAuthorizationToken"]
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ml_engineer" {
  role       = aws_iam_role.ml_engineer.name
  policy_arn = aws_iam_policy.ml_engineer.arn
}

resource "aws_iam_role" "data_engineer" {
  name = "${local.name_prefix}-DataEngineer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = ["glue.amazonaws.com", "lambda.amazonaws.com", "sagemaker.amazonaws.com"] }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "data_engineer" {
  name        = "${local.name_prefix}-DataEngineerPolicy"
  description = "Least-privilege permissions for the DataEngineer role (Glue ETL and Feature Store ingestion)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Databases, tables, crawlers, jobs, runs - and glue:GetConnection,
        # which Glue needs to resolve the VPC NETWORK connection.
        Sid      = "GlueFullAccess"
        Effect   = "Allow"
        Action   = "glue:*"
        Resource = "*"
      },
      {
        # Glue workers run in the private subnet and need ENIs there.
        Sid    = "GlueVpcNetworkInterfaces"
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface", "ec2:DeleteNetworkInterface",
          "ec2:DescribeNetworkInterfaces", "ec2:DescribeSubnets", "ec2:DescribeVpcs",
          "ec2:DescribeSecurityGroups", "ec2:DescribeVpcEndpoints", "ec2:DescribeRouteTables",
          "ec2:DescribeVpcAttribute", "ec2:DescribeAvailabilityZones",
        ]
        Resource = "*"
      },
      {
        # Glue tags every ENI it creates.
        Sid      = "GlueEniTagging"
        Effect   = "Allow"
        Action   = ["ec2:CreateTags", "ec2:DeleteTags"]
        Resource = "arn:aws:ec2:*:*:network-interface/*"
      },
      {
        # Read/write ONLY the data-plane prefixes. artifacts/ is deliberately absent.
        Sid    = "S3DataPrefixesReadWrite"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = [
          "arn:aws:s3:::${local.name_prefix}-data-*/raw/*",
          "arn:aws:s3:::${local.name_prefix}-data-*/processed/*",
          "arn:aws:s3:::${local.name_prefix}-data-*/features/*",
        ]
      },
      {
        # The Feature Store offline store writes objects with an ACL.
        Sid      = "S3FeatureStoreObjectAcl"
        Effect   = "Allow"
        Action   = ["s3:PutObjectAcl"]
        Resource = "arn:aws:s3:::${local.name_prefix}-data-*/features/*"
      },
      {
        # Glue must fetch its own job scripts. Read-only.
        Sid      = "S3GlueScriptsReadOnly"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "arn:aws:s3:::${local.name_prefix}-data-*/artifacts/glue/*"
      },
      {
        # GetBucketAcl: Feature Store checks it before accepting the offline store URI.
        Sid      = "S3BucketLevel"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation", "s3:GetBucketAcl"]
        Resource = "arn:aws:s3:::${local.name_prefix}-data-*"
      },
      {
        Sid    = "FeatureStoreWrite"
        Effect = "Allow"
        Action = [
          "sagemaker:PutRecord", "sagemaker:CreateFeatureGroup", "sagemaker:DescribeFeatureGroup",
        ]
        Resource = "arn:aws:sagemaker:*:*:feature-group/*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = [
          "arn:aws:logs:*:*:log-group:/aws-glue/*",
          "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
        ]
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "data_engineer" {
  role       = aws_iam_role.data_engineer.name
  policy_arn = aws_iam_policy.data_engineer.arn
}

resource "aws_iam_role" "model_monitor" {
  name = "${local.name_prefix}-ModelMonitor"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "model_monitor" {
  name        = "${local.name_prefix}-ModelMonitorPolicy"
  description = "Observe-only permissions for the ModelMonitor role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchMetricsAndAlarms"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData", "cloudwatch:GetMetricStatistics",
          "cloudwatch:PutMetricAlarm", "cloudwatch:DescribeAlarms",
        ]
        Resource = "*"
      },
      {
        # Read-only visibility into drift runs. No CreateProcessingJob.
        Sid      = "SageMakerProcessingReadOnly"
        Effect   = "Allow"
        Action   = ["sagemaker:ListProcessingJobs", "sagemaker:DescribeProcessingJob"]
        Resource = "*"
      },
      {
        Sid      = "S3ArtifactsReadOnly"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "arn:aws:s3:::${local.name_prefix}-data-*/artifacts/*"
      },
      {
        Sid      = "S3BucketList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = "arn:aws:s3:::${local.name_prefix}-data-*"
      },
      {
        Sid      = "CloudWatchLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "model_monitor" {
  role       = aws_iam_role.model_monitor.name
  policy_arn = aws_iam_policy.model_monitor.arn
}
