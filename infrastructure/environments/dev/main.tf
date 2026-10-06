# ── environments/dev ─────────────────────────────────────────────────────────
# Wire the four modules together here. Each module call passes var.project and
# var.environment down; nothing in modules/ hardcodes a name.
#
# Uncomment each block as you implement the module it calls.

module "vpc" {
  source              = "../../modules/vpc"
  project             = var.project
  environment         = var.environment
  vpc_cidr            = var.vpc_cidr
  public_subnet_cidr  = var.public_subnet_cidr
  availability_zone   = var.availability_zone
  private_subnet_cidr = var.private_subnet_cidr
}

module "storage" {
  source      = "../../modules/storage"
  project     = var.project
  environment = var.environment
}

module "iam" {
  source      = "../../modules/iam"
  project     = var.project
  environment = var.environment
}

module "sagemaker" {
  source             = "../../modules/sagemaker"
  project            = var.project
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = [module.vpc.private_subnet_id]
  security_group_ids = [module.vpc.security_group_id]
  execution_role_arn = module.iam.ml_engineer_role_arn
  instance_type      = var.sagemaker_instance_type
}

module "glue" {
  source                 = "../../modules/glue"
  project                = var.project
  environment            = var.environment
  availability_zone      = var.availability_zone
  vpc_id                 = module.vpc.vpc_id
  private_subnet_id      = module.vpc.private_subnet_id
  bucket_name            = module.storage.bucket_name
  data_engineer_role_arn = module.iam.data_engineer_role_arn
  feature_group_name     = module.feature_store.feature_group_name
}

module "feature_store" {
  source      = "../../modules/feature_store"
  project     = var.project
  environment = var.environment
  bucket_name = module.storage.bucket_name
  role_arn    = module.iam.data_engineer_role_arn

  # The role's trust policy and S3 permissions must exist before
  # CreateFeatureGroup validates them.
  depends_on = [module.iam]
}
