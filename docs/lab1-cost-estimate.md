# Lab 1 Monthly Cost Estimate (steady state, us-east-1)

Steady state means the Lab 1 platform left in place for a month with one engineer using Studio on working days. Prices are us-east-1 list prices from the course cost guide (`aws-account-setup.md`) and AWS's published pricing; each figure traces to the assumption beside it.

| Component | Monthly Estimate | Key Assumptions | One Optimization |
|-----------|------------------|-----------------|------------------|
| SageMaker Studio (JupyterLab space) | $2.20 | 2 hrs/day × 22 working days = 44 hrs at $0.05/hr (`ml.t3.medium`) | Enable idle auto-shutdown at 60 minutes (quantified below) |
| S3 storage (data bucket) | $1.15 | 50 GB at $0.023/GB-month, including old object versions across the four prefixes | Move cold `raw/` data to S3 Standard-IA after 90 days: 30 GB × ($0.023 − $0.0125) saves $0.32/month |
| Internet Gateway | $0.90 | The gateway itself is free; 10 GB/month leaves the VPC to the internet at $0.09/GB, ignoring the free 100 GB/month allowance | Keep Studio-to-S3 traffic in-region (no transfer charge) and stay inside the free allowance: saves the full $0.90 |
| DynamoDB (state lock) | $0.00 | On-demand, about 100 Terraform runs/month × 4 lock operations = 400 requests at $1.25 per million writes = $0.0005 | Drop the table and use S3-native locking (`use_lockfile`, the option Terraform's deprecation warning points to): removes one resource, saves under $0.01 |
| S3 state bucket | $0.01 | About 100 versions × 50 KB = 5 MB (under $0.001) plus about 1,000 requests at $0.005 per 1,000 writes | Expire non-current state versions after 30 days: caps version growth, saves under $0.01 |
| **Total** | **$4.26** | | |

Not in the table: Studio also keeps a home-directory volume on EFS at about $0.30/GB-month. At 5 GB that is $1.50 a month, bringing the realistic total to about $5.76, which sits inside the course guide's $3–6 estimate for Lab 1. The Terraform `retention_policy` deletes that volume when the domain is destroyed.
