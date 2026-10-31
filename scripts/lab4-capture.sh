#!/usr/bin/env bash
# Lab 4 capture block (from the handout). Run from the repo root while the
# northstar-cicd stack is still up:
#     bash scripts/lab4-capture.sh <final-green-codebuild-build-id>
set -uo pipefail
export MSYS_NO_PATHCONV=1 AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
PY=$(command -v python3 || command -v python)
GREEN=${1:?usage: bash scripts/lab4-capture.sh <final-green-build-id>}
TMPLOG=lab4-build.tmp.log

save_log () {   # usage: save_log <codebuild-build-id> <output-file>
  read -r G S <<< "$(aws codebuild batch-get-builds --ids "$1" \
      --query 'builds[0].logs.[groupName,streamName]' --output text)"
  aws logs filter-log-events --log-group-name "$G" --log-stream-names "$S" --output json \
    | "$PY" -c 'import json,sys; sys.stdout.write("".join(e["message"] if e["message"].endswith("\n") else e["message"]+"\n" for e in json.load(sys.stdin)["events"]))' \
    > "$2"
}

{
  echo "===== CodePipeline executions: start, status, commit, execution id ====="
  aws codepipeline list-pipeline-executions --pipeline-name northstar-pipeline \
    --query 'pipelineExecutionSummaries[].[startTime,status,sourceRevisions[0].revisionId,pipelineExecutionId]' --output text
  echo; echo "===== CodeBuild builds, newest first: start, status, commit, build id ====="
  for B in $(aws codebuild list-builds-for-project --project-name northstar-ci --query 'ids[]' --output text); do
    aws codebuild batch-get-builds --ids "$B" --query 'builds[0].[startTime,buildStatus,resolvedSourceVersion,id]' --output text
    aws codebuild batch-get-builds --ids "$B" --query 'builds[0].phases[].[phaseType,phaseStatus]' --output text | sed 's/^/    /'
    save_log "$B" "$TMPLOG" && grep -E '^=+ .*(passed|failed|error)' "$TMPLOG" | sed 's/^/    pytest: /'
  done
  echo; echo "===== SageMaker Pipeline executions: start, status, name ====="
  aws sagemaker list-pipeline-executions --pipeline-name northstar-churn-pipeline \
    --query 'PipelineExecutionSummaries[].[StartTime,PipelineExecutionStatus,PipelineExecutionDisplayName]' --output text
} 2>&1 | tee docs/lab4-pipeline-runs.txt
rm -f "$TMPLOG"

save_log "$GREEN" docs/lab4-build-green.log
aws sagemaker list-model-packages --model-package-group-name northstar-churn-models \
  --sort-by CreationTime --sort-order Descending --output json > docs/lab4-model-registry.json

echo; echo "===== sanity checks ====="
wc -l docs/lab4-build-green.log docs/lab4-pipeline-runs.txt docs/lab4-model-registry.json
grep -cE "test_(data|features|model|fairness)\.py::.* PASSED" docs/lab4-build-green.log | sed 's/^/PASSED lines: /'
grep -E "champion v[0-9]+|FAIRNESS|Succeeded|Registered:|passed|failed" docs/lab4-build-green.log | head -20
