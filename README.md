# NorthStar AI Platform (CS 401R)

Infrastructure-as-code for the NorthStar Retail ML platform, built lab by lab
with Terraform on AWS (`us-east-1`). Lab 1 laid the foundation (VPC, S3 data
bucket, MLEngineer role, SageMaker Studio). **Lab 2** hardens the network,
adds the DataEngineer and ModelMonitor roles, and builds the data pipeline:
raw customer transactions -> cleaned Parquet -> engineered churn features in
SageMaker Feature Store, ready for model training in Lab 3.

## Repository layout

```
infrastructure/
  modules/
    vpc/            VPC, public + private subnets, IGW, NAT Gateway + EIP, route tables, SageMaker SG
    storage/        data bucket (versioned, SSE-S3, public access blocked), prefixes, lifecycle rules
    iam/            MLEngineer, DataEngineer, ModelMonitor roles + least-privilege policies
    sagemaker/      SageMaker Domain (private subnet, VpcOnly) + MLEngineer user profile
    glue/           Glue catalog DB, raw crawler, VPC connection + SG, both ETL jobs, script uploads
    feature_store/  SageMaker Feature Group northstar-dev-customer-features (16 definitions)
  environments/
    dev/            real AWS (remote S3 state); wires all six modules together
    local/          LocalStack (vpc, storage, iam only; NAT and lifecycle rules disabled)
glue-scripts/
  transform.py         raw/customers/ CSV -> processed/customers/ Parquet
  feature_engineer.py  processed/customers/ -> features/customers/ Parquet + Feature Store
scripts/
  bootstrap-state.sh   creates the Terraform state bucket and lock table (Lab 1)
  check-secrets.sh     pre-commit secret scan
  verify-lab1.sh       Lab 1 rubric checks
  verify-lab2.sh       Lab 2 rubric checks (infra, data quality, Feature Store)
  teardown-lab2.sh     full Lab 2 teardown, including resources Terraform does not own
docs/                  ADRs, diagrams, data contract, apply/verify/destroy evidence
```

## Lab 2 additions

| Area | Change |
|------|--------|
| Networking | Private subnet `northstar-dev-private-1` (10.0.1.0/24), NAT Gateway in the public subnet, private route table `0.0.0.0/0 -> NAT`. SageMaker Domain moved into the private subnet with `app_network_access_type = "VpcOnly"`. |
| IAM | `northstar-dev-DataEngineer` (trusted by Glue, Lambda, SageMaker): read/write `raw/`, `processed/`, `features/`; read-only `artifacts/glue/`; Glue, VPC ENI, Feature Store write. `northstar-dev-ModelMonitor` (trusted by SageMaker): CloudWatch metrics/alarms, read-only processing jobs and `artifacts/`, no S3 writes. |
| S3 lifecycle | `expire-raw-data` (90 d), `expire-raw-versions` (30 d noncurrent), `expire-processed-versions` (30 d noncurrent), `expire-feature-versions` (60 d noncurrent), `expire-datacapture` (7 d). |
| `modules/glue` | Catalog DB `northstar_dev`; crawler `northstar-dev-raw-crawler` on `raw/customers/` -> table `customers`; NETWORK connection + self-referencing SG so jobs run in the private subnet; jobs `northstar-dev-transform` and `northstar-dev-feature-engineer` (Glue 4.0); scripts uploaded to `artifacts/glue/` on apply. |
| `modules/feature_store` | Feature Group `northstar-dev-customer-features`: record ID `customer_id`, event time `event_time` (Fractional), online store on, offline store at `features/offline-store/`. 13 features + `churn_label` (Integral). |

### Feature engineering and the temporal split
Features are computed only from purchases on or before `FEATURE_CUTOFF`
(T = 2026-04-01). `churn_label` is 1 if the customer made no purchase in the
outcome window (T, 2026-06-30]. Each customer contributes one labeled row.
`churn_risk_score` is a rule-based recency baseline, not the label.

See `docs/lab2-data-contract.md` for the `processed/customers/` contract and
`docs/lab2-data-lineage.png` for the lineage diagram.

## Running the data pipeline end to end

Prerequisites: AWS CLI authenticated to the lab account, Terraform >= 1.5,
Git Bash (Windows) or any bash shell. Run from the repo root unless noted.

```bash
# 1. Deploy infrastructure (about 15 minutes, mostly the SageMaker Domain)
cd infrastructure/environments/dev
terraform init
terraform apply
cd ../../..

# 2. Upload the raw data
ACCT=$(aws sts get-caller-identity --query Account --output text)
aws s3 cp northstar-raw-sample.csv \
  s3://northstar-dev-data-$ACCT/raw/customers/northstar-raw-sample.csv

# 3. Crawl raw/ (repeat get-crawler until it returns "READY")
aws glue start-crawler --name northstar-dev-raw-crawler
aws glue get-crawler --name northstar-dev-raw-crawler --query Crawler.State
aws glue get-table --database-name northstar_dev --name customers

# 4. Transform: raw -> processed (wait for SUCCEEDED)
aws glue start-job-run --job-name northstar-dev-transform
aws glue get-job-runs --job-name northstar-dev-transform \
  --query 'JobRuns[0].[JobRunState,ErrorMessage]'

# 5. Feature engineering: processed -> features + Feature Store (wait for SUCCEEDED)
aws glue start-job-run --job-name northstar-dev-feature-engineer
aws glue get-job-runs --job-name northstar-dev-feature-engineer \
  --query 'JobRuns[0].[JobRunState,ErrorMessage]'

# 6. Verify everything against the rubric
pip install pandas pyarrow
bash scripts/verify-lab2.sh | tee docs/lab2-verify-output.txt
```

The sample CSV is not committed (it is in `.gitignore`); download it from the
Lab 2 starter kit on Canvas.

### Local validation (LocalStack)

```bash
make local-validate LOCAL_OUT=docs/lab2-localstack-output.txt
```

Applies the vpc, storage, and iam modules to LocalStack with the NAT Gateway
and lifecycle rules disabled, and confirms all three IAM roles exist.

## Teardown

The NAT Gateway bills about $0.045/hour whether or not it is used, so tear
down as soon as verification is captured:

```bash
bash scripts/teardown-lab2.sh
```

`terraform destroy` alone is not enough for Lab 2: Glue ENIs, the Studio EFS
volume, SageMaker NFS security groups, S3 object versions, the
`sagemaker_featurestore` Glue database, and Feature Store lineage entities are
created outside Terraform state. The script removes them in order, runs
`terraform destroy` (output in `docs/lab2-destroy-output.txt`), and confirms
no billable resources remain. Rebuild later with `terraform apply`.
