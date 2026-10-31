#!/usr/bin/env bash
# Lab 4 Task 3 -- copy lineage from the training job onto a model package, then
# save the package description to docs/lab4-model-package.json.
#
#   usage (repo root, Git Bash):  bash scripts/lab4-lineage.sh <package-version>
#
# The RegisterModel step records only the artifact and a pointer to the metrics.
# The commit SHA (COMMIT_SHA env var) and the training data URI (input channel)
# live on the TRAINING JOB, so we find that job from the package's model
# artifact path and copy them across as CustomerMetadataProperties.
set -euo pipefail
export MSYS_NO_PATHCONV=1 AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
PY=$(command -v python3 || command -v python)

N=${1:?usage: bash scripts/lab4-lineage.sh <package-version>}
ACCT=$(aws sts get-caller-identity --query Account --output text)
ARN="arn:aws:sagemaker:us-east-1:${ACCT}:model-package/northstar-churn-models/${N}"

MODEL_URI=$(aws sagemaker describe-model-package --model-package-name "$ARN" \
  --query 'InferenceSpecification.Containers[0].ModelDataUrl' --output text)
# s3://<bucket>/artifacts/training/<training-job-name>/output/model.tar.gz
JOB=$(echo "$MODEL_URI" | awk -F/ '{print $(NF-2)}')
echo "package v$N  <-  training job $JOB"

COMMIT8=$(aws sagemaker describe-training-job --training-job-name "$JOB" \
  --query 'Environment.COMMIT_SHA' --output text)
DATA_URI=$(aws sagemaker describe-training-job --training-job-name "$JOB" \
  --query 'InputDataConfig[0].DataSource.S3DataSource.S3Uri' --output text)
TRAIN_START=$(aws sagemaker describe-training-job --training-job-name "$JOB" \
  --query 'TrainingStartTime' --output text)
BUCKET=$(echo "$DATA_URI" | cut -d/ -f3); PREFIX=$(echo "$DATA_URI" | cut -d/ -f4-)
DATA_TS=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$PREFIX" \
  --query 'max_by(Contents, &LastModified).LastModified' --output text)
COMMIT_FULL=$(git rev-parse --verify --quiet "${COMMIT8}^{commit}" || echo "$COMMIT8")
echo "commit $COMMIT_FULL | data $DATA_URI @ $DATA_TS"

# Metrics pointer: whichever ModelMetrics entry ends in evaluation_metrics.json,
# else the path the training job uploads to (artifacts/evaluations/<sha8>/).
METRICS_URI=$(aws sagemaker describe-model-package --model-package-name "$ARN" \
  --query 'ModelMetrics' --output json | "$PY" -c '
import json, sys
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "S3Uri" and isinstance(v, str): yield v
            else: yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)
raw = sys.stdin.read().strip()
uris = list(walk(json.loads(raw))) if raw and raw != "null" else []
print(next((u for u in uris if u.endswith("evaluation_metrics.json")), ""))
' | tr -d '\r')
if [ -z "$METRICS_URI" ]; then
  METRICS_URI="s3://${BUCKET}/artifacts/evaluations/${COMMIT8}/evaluation_metrics.json"
fi
echo "metrics $METRICS_URI"

aws s3 cp "$METRICS_URI" - | \
  COMMIT_FULL="$COMMIT_FULL" COMMIT8="$COMMIT8" DATA_URI="$DATA_URI" DATA_TS="$DATA_TS" \
  JOB="$JOB" TRAIN_START="$TRAIN_START" METRICS_URI="$METRICS_URI" MODEL_URI="$MODEL_URI" \
  "$PY" -c '
import json, os, sys
m = json.load(sys.stdin)
e = os.environ
meta = {
    "commit_sha": e["COMMIT_FULL"],
    "commit_sha_short": e["COMMIT8"],
    "training_data_s3_uri": e["DATA_URI"],
    "training_data_timestamp": e["DATA_TS"],
    "training_job_name": e["JOB"],
    "training_start_time": e["TRAIN_START"],
    "model_artifact_s3_uri": e["MODEL_URI"],
    "evaluation_metrics_s3_uri": e["METRICS_URI"],
}
for k in ["auc_roc", "baseline_auc", "auc_vs_baseline", "lift_ci_low", "lift_ci_high",
          "precision_top10", "recall_top10", "positive_rate_val", "n_val_samples",
          "tier_recall_gap_platinum_bronze", "fairness_flag"]:
    if m.get(k) is not None:
        meta["eval_" + k] = str(m[k])
json.dump(meta, open("lineage-meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
'

aws sagemaker update-model-package --model-package-arn "$ARN" \
  --customer-metadata-properties file://lineage-meta.json >/dev/null
rm -f lineage-meta.json
aws sagemaker describe-model-package --model-package-name "$ARN" > docs/lab4-model-package.json
echo "wrote docs/lab4-model-package.json"
grep -A3 '"CustomerMetadataProperties"' docs/lab4-model-package.json
