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

## Lab 3 additions: model development

Lab 3 builds two of NorthStar's AI systems on the Lab 2 features:
- **Track A**, a churn model: XGBoost trained on the Feature Store offline store via Athena.
- **Track B**, retention-offer generation: RAG over `northstar-policy-docs/` with Bedrock.

Design and results:
- [`docs/lab3-model-design.md`](docs/lab3-model-design.md)
- [`docs/lab3-evaluation-report.md`](docs/lab3-evaluation-report.md)

| Path | What it is |
|---|---|
| `churn_training_skeleton.py` | Churn training: Athena query → XGBoost → baseline + bootstrap lift CI → slices → MLflow App run → Model Registry (`--register`, `PendingManualApproval`) |
| `rag/offer_rag.py` | RAG pipeline: header chunking → Titan Text Embeddings v2 → FAISS → tier card from POL-LOY-011 §4 → Claude Haiku 4.5 |
| `rag/run_eval.py` | Runs the RAGAS harness (`evaluation_harness.py`) N times for the `baseline` / `improved` / `tiercard` configs |
| `notebooks/lab3-demo.ipynb` | End-to-end Track B demo, committed with outputs |
| `docs/lab3-evidence/` | Graded evidence: metrics, slices, plots, RAGAS results, examples and failures |

### Train and register the churn model

Run locally with your own AWS credentials, which the Athena query needs:

```bash
pip install xgboost scikit-learn "sagemaker<3" mlflow sagemaker-mlflow
python churn_training_skeleton.py --artifacts-bucket northstar-dev-data-<account> \
  --max-depth 3 --eta 0.05 --num-round 400 \
  --mlflow-arn <MLflow App ARN> --run-name xgb-d3-eta0.05 \
  --output-data-dir docs/lab3-evidence [--register]
```

Metrics land in `--output-data-dir` (`evaluation_metrics.json`). Runs are logged to the MLflow App experiment `northstar-churn`. `--register` adds a `PendingManualApproval` version to `northstar-churn-models`.

### Run and evaluate the offer generator

```bash
pip install faiss-cpu langchain-text-splitters ragas datasets "langchain-community<0.4" langchain-aws
python rag/run_eval.py                  # 2 runs of each config, 4 test cases each
python rag/run_eval.py --only tiercard  # one config only
```

Per-run CSVs go to `evaluation_results/`. The graded copy is `docs/lab3-evidence/rag_eval.csv` (the final `tiercard` config); the other configs are saved as `rag_eval_<config>.csv`. The RAGAS judge is Claude Haiku 4.5 on Bedrock (no OpenAI key needed).

### Lab 3 teardown

```bash
bash scripts/teardown-lab3.sh   # endpoints, endpoint configs, running jobs, MLflow Tracking Server check
```

Output: `docs/lab3-teardown-output.txt`. The MLflow **App** is kept deliberately: it costs nothing and Labs 4 and 6 use it.

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
