# ── modules/storage ──────────────────────────────────────────────────────────
# Required resources (Task B1). Only these belong in this module:
#
#   aws_s3_bucket
#   aws_s3_bucket_public_access_block
#   aws_s3_bucket_versioning
#   aws_s3_bucket_server_side_encryption_configuration
#   aws_s3_object  x4                 the raw/ processed/ features/ artifacts/ prefixes
#
# ONE bucket with four prefixes, not four buckets. Later labs derive the name
# as ${project}-${environment}-data-${account_id}, so keep that shape.
#
# The four aws_s3_object resources create the prefixes. S3 has no real
# directories; an empty object with a trailing slash is how a prefix is made
# to exist before anything is written to it.

# TODO: implement the resources above.

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "data" {
  bucket = "${var.project}-${var.environment}-data-${data.aws_caller_identity.current.account_id}"

  # A versioned bucket cannot be deleted while it holds object versions.
  # force_destroy lets `terraform destroy` empty it first.
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 has no real folders; an empty object whose key ends in "/" makes the
# prefix visible in the console.
resource "aws_s3_object" "prefixes" {
  for_each = toset(var.prefixes)

  bucket  = aws_s3_bucket.data.id
  key     = each.value
  content = ""

  depends_on = [
    aws_s3_bucket_public_access_block.data,
    aws_s3_bucket_versioning.data,
    aws_s3_bucket_server_side_encryption_configuration.data,
  ]
}
