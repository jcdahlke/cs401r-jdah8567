# Lab 4 — XOps Assessment: CI/CD, MLOps Configuration, Maturity

NorthStar churn model, repository `cs401r-jdah8567`. All evidence files cited
below are committed under `docs/` at the `lab4-submit` tag.

## Pipeline Stage Mapping

**Implementation:** AWS CodePipeline + CodeBuild (`pipeline.yaml`, `buildspec.yml`).

**Rationale.** Everything else on the platform (S3, Glue, SageMaker, IAM roles
from Terraform) already lives in this AWS account, so an AWS-native pipeline
needs no long-lived AWS credentials stored in GitHub. The build runs under a
scoped IAM role (`northstar-codebuild-role`) and passes only the MLEngineer role
to SageMaker. CodeBuild also emits the `AWS/CodeBuild FailedBuilds` metric on
its own, so the failure alarm fires even when `pre_build` fails and
`post_build` is skipped. The trade-off is a slower edit-and-run loop than
GitHub Actions and a GitHub connection that has to be authorized by hand.

Five CI phases, three CodePipeline stages:

| # | CI phase | CodePipeline stage | Where it happens | Enforcement |
|---|----------|--------------------|------------------|-------------|
| 1 | **Source** | `Source` | CodeConnections (`northstar-github`) → GitHub `main` | Every run builds a specific commit; CodeBuild records it as `resolvedSourceVersion` |
| 2 | **Test** | `Build` | `buildspec.yml` `pre_build`: `pytest tests/test_data.py` (schema, grain, leakage, distributions, freshness, run against the live S3 data) then `pytest tests/test_features.py` | Any failure ends the build in `PRE_BUILD`. `BUILD` never runs, so nothing trains or registers |
| 3 | **Build** | `Build` | `buildspec.yml` `build`: upserts and runs SageMaker Pipeline `northstar-churn-pipeline`. Step `TrainChurnModel` runs `models/churn/train_sagemaker.py` → `churn_model.train()` on managed spot `ml.m5.large`, pinned to the commit SHA | `train()` keeps Lab 3's raising `evaluate_model()`, so a model that misses precision/recall/lift fails the training step |
| 4 | **Evaluate** | `Build` | `buildspec.yml` `build`: `pytest tests/test_model.py tests/test_fairness.py -rw` against the trained artifact and its `evaluation_metrics.json` | Precision@10% ≥ 0.50, recall@10% ≥ 0.25, lift-CI low > 0, champion regression (AUC ≥ champion − 0.02), prediction contract. Fairness is flagged, not blocking |
| 5 | **Register** | `Build` (SageMaker step `RegisterChurnModel`) | Package created in `northstar-churn-models` as `PendingManualApproval` | Registration runs before the CodeBuild gate. If any later test fails, `post_build` sets that package to `Rejected`, so a red build leaves nothing approvable |
| — | Human promotion | `ManualApproval` | SNS-notified approval action (`northstar-model-approvals`) | A person approves the package in the registry |

**Failure path.** CloudWatch alarm `northstar-ci-build-failure` watches
`FailedBuilds` and publishes to SNS topic `northstar-model-approvals`, which has
a confirmed email subscriber. Proof is in `docs/lab4-failure-demo.txt`: commit
`12a5fd4` ended with `PRE_BUILD FAILED` and no `BUILD` phase, the registry
stayed at v3, the alarm went `OK → ALARM` at 15:56:36 with *Successfully executed
action*, and the email is in `docs/lab4-alarm-email.png`. The revert `9d4d45a`
went green, registered v4 `PendingManualApproval`, and the alarm returned to `OK`.

*Operational note:* `DetectChanges: true` is set on the Source action, but
pushes to this repository did not start executions on their own. The demo runs
were started with `aws codepipeline start-pipeline-execution`. Every run still
builds the head of `main`, and each build's commit SHA appears in
`docs/lab4-pipeline-runs.txt`.

## Champion-Challenger

The champion is the newest **Approved** package in `northstar-churn-models`.
A challenger replaces it if and only if **all** of the following hold, each
evaluated on that challenger's held-out 30% validation split:

1. **Absolute gates:** `precision_top10 ≥ 0.50` **and** `recall_top10 ≥ 0.25` **and** `lift_ci_low > 0` (the 95% bootstrap CI on AUC lift over the recency-only baseline excludes zero).
2. **No regression:** `auc_roc_challenger ≥ auc_roc_champion − 0.02`. This is enforced in CI by `test_candidate_not_worse_than_champion`; failing it fails the build and auto-rejects the package.
3. **Material improvement:** `auc_roc_challenger − auc_roc_champion ≥ 0.010` **or** `precision_top10_challenger − precision_top10_champion ≥ 0.03`.

If all three hold, the challenger is approved and becomes the champion. If not,
the champion stays and the challenger is set to `Rejected`. A tie keeps the
champion.

Rule 2 is what a challenger must clear to be *considered*. Rule 3 is what it
must clear to *replace* the champion. The 0.010 AUC margin is about the
run-to-run spread from data-order changes measured in Lab 3 (0.7276 to 0.7431
across reorderings), so a challenger has to beat the champion by more than
noise. Example: v4 (AUC 0.7848) passes rules 1 and 2 against champion v3 but
not rule 3. The same code on the same data does not justify a swap, so v3
stays champion.

## Retraining Triggers

Both triggers start the same SageMaker Pipeline (`northstar-churn-pipeline`)
that CI uses, so a retrained model passes through the same gates and
`PendingManualApproval` registration as a code change. Designed, not deployed.

**(a) Scheduled: weekly.**
- **Schedule:** Amazon **EventBridge Scheduler**, `cron(0 3 ? * MON *)` in `America/Denver` (03:00 every Monday).
- **Step 1:** starts an AWS **Glue Workflow** that chains `northstar-dev-raw-crawler` → `northstar-dev-transform` → `northstar-dev-feature-engineer`. This also keeps `features/` inside the 48-hour freshness gate.
- **Step 2:** an **EventBridge rule** on the Glue `Job State Change` event (`jobName = northstar-dev-feature-engineer`, `state = SUCCEEDED`) targets the SageMaker Pipeline directly (EventBridge supports `StartPipelineExecution` as a target). It passes `CommitSha` = the commit of the currently Approved package.

**(b) Performance-based: model quality drops on production data.**
- **Measurement:** a **SageMaker Model Monitor** model-quality schedule (daily, `cron(0 4 ? * * *)`) joins endpoint data capture (Lab 5/6) with ground-truth `churn_label`s. The labels arrive once each 90-day holdout window closes. The job publishes `auc` and `precision` to **CloudWatch**.
- **Alarm:** a **CloudWatch alarm** fires when `auc < 0.74` (roughly 0.045 below the current 0.7848) **or** `precision_top10 < 0.50` (the promotion gate) for **2 consecutive daily evaluations**.
- **Response:** an **EventBridge rule** on that alarm's `CloudWatch Alarm State Change` → `ALARM` event starts `northstar-churn-pipeline`. The same alarm notifies `northstar-model-approvals` over SNS.
- **Earlier warning:** a Model Monitor *data-quality* (feature drift) violation on `days_since_last_purchase` or `purchase_frequency_90d` raises an SNS warning only, not a retrain. Drift is a reason to look, not proof that quality dropped.

## DataOps Maturity

**Current level: 2**

**Evidence:**
- The data platform is code. Terraform modules (`infrastructure/modules/glue`, `storage`, `feature_store`, `vpc`, `iam`) recreated all 41 Lab 2 resources from scratch this session.
- The Glue jobs are version-controlled: `glue-scripts/transform.py` and `glue-scripts/feature_engineer.py`.
- The producer/consumer agreement is written down: `docs/lab2-data-contract.md`.
- Data quality is an automated gate. `tests/test_data.py` checks raw, processed and feature data on every CI run: schema, transaction grain, label leakage, bounds, window monotonicity, a cutoff-only recomputation of the time-sensitive features, and 48-hour freshness. Its result is the `test_data.py` `PASSED` lines in `docs/lab4-build-green.log`.
- Feature logic is unit-tested: `data/feature_engineering.py` with 26 tests in `tests/test_features.py`.

**Gap to Level 3:**
- Pipeline *execution* is still manual. Nothing in `infrastructure/modules/glue` defines a trigger or workflow. The crawler and both jobs were started by hand with `aws glue start-job-run`, so data freshness depends on a person remembering to do it.
- Quality checks run in CI, not inside the data pipeline. A bad Glue run would publish to `features/` and only be caught at the next code push.
- The feature logic is duplicated: Spark in `glue-scripts/feature_engineer.py` and pandas in `data/feature_engineering.py`. Only the CI parity test keeps the two in sync.

**Top priority investment:** Add an `aws_glue_workflow` with chained `aws_glue_trigger`s (scheduled crawler → conditional transform → conditional feature job) to `infrastructure/modules/glue`. Add a **Glue Data Quality** ruleset (DQDL) after the transform, so a failing ruleset stops publication before `features/` is overwritten.

## MLOps Maturity

**Current level: 2**

**Evidence:**
- Training is fully automated and gated, from commit to model:
  - `buildspec.yml` runs `models/churn/pipeline_definition.py`, which trains `churn_model.train()` on spot capacity and registers to `northstar-churn-models` as `PendingManualApproval`.
  - `tests/test_model.py` enforces the precision, recall and lift-CI gates plus a registry-backed champion regression test. The `champion v3 auc=...` warning in `docs/lab4-build-green.log` shows it compared against v3.
  - `tests/test_fairness.py` makes the tier recall gap visible to the approver (Platinum 0.143 vs Bronze 0.446).
- Failure halts and notifies: see `docs/lab4-failure-demo.txt`.
- Lineage is recorded. `docs/lab4-model-package.json` carries the commit SHA, the training data URI and timestamp, and the evaluation metrics.
- Experiments are tracked in the MLflow App: `docs/lab4-mlflow-runs.txt`.

**Gap to Level 3:**
- No production feedback loop. Nothing measures the model after registration: there is no endpoint data capture and no Model Monitor (Lab 6).
- Retraining has no automated trigger. The triggers above are designed, not deployed, and source changes did not fire the pipeline on their own.
- Promotion is a manual approval with no deployment step.

**Top priority investment:** Deploy the performance trigger:
- a **SageMaker Model Monitor** model-quality schedule on the Lab 5 endpoint, with a CloudWatch alarm on `auc < 0.74`;
- an **EventBridge** rule from that alarm to `StartPipelineExecution` on `northstar-churn-pipeline`.

Together these close the loop from production degradation back to a gated retrain.
