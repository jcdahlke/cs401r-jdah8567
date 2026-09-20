# ── modules/sagemaker ────────────────────────────────────────────────────────
# Required resources (Task B1). Only these belong in this module:
#
#   aws_sagemaker_domain
#   aws_sagemaker_user_profile
#
# A brand-new AWS account has no service-linked role for Studio, and the
# Domain fails to create with a service-linked role error. Fix it once in the
# console (IAM -> Roles -> Create Role -> AWS Service -> SageMaker -> SageMaker
# Studio) and re-apply. See the New Account Bootstrap note in the lab.
#
# The Domain is also the slowest resource here by a wide margin -- several
# minutes to create and to delete. Factor that into your apply/destroy timings
# for Task B2.

# The Domain needs an execution role (default_user_settings.execution_role)
# and the security group from modules/vpc; both arrive as variables. See the
# module call in environments/dev/main.tf.

# TODO: implement the two resources above.

resource "aws_sagemaker_domain" "this" {
  domain_name = "${var.project}-${var.environment}-domain"
  auth_mode   = "IAM"
  vpc_id      = var.vpc_id
  subnet_ids  = var.subnet_ids

  default_user_settings {
    execution_role  = var.execution_role_arn
    security_groups = var.security_group_ids

    sharing_settings {
      notebook_output_option = "Disabled"
    }

    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = var.instance_type
      }
    }
  }

  # Studio creates an EFS home-directory volume that Terraform never sees.
  # With the default (Retain) it survives DeleteDomain, pins the subnet, and
  # makes `terraform destroy` hang. Delete it with the domain.
  retention_policy {
    home_efs_file_system = "Delete"
  }
}

resource "aws_sagemaker_user_profile" "ml_engineer" {
  domain_id         = aws_sagemaker_domain.this.id
  user_profile_name = "MLEngineer"

  user_settings {
    execution_role = var.execution_role_arn
  }
}
