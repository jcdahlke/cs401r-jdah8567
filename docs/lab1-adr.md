## ADR-001: NorthStar Platform Foundation

### Status
Accepted

### Context
NorthStar Retail loses about 18% of its 2.1M active customers each year. At roughly $340 of lifetime value per customer, that is the $128.5M annual churn problem (2.1M × 18% × $340) the CDO has asked the AI team to attack. Three systems share one platform: a weekly churn model that must publish 90-day churn scores by Monday 6 AM ET, an LLM/RAG offer generator that must respond within 2 seconds, and a customer service agent that must hit 99.5% availability from 8 AM to 10 PM. All three read the same customer and transaction data, which contains PII governed by GDPR, CCPA, and a 24-month raw-retention policy.

The network boundary, storage layout, and identity model are the hardest things to change once data flows. Re-permissioning a bucket that pipelines already write to, or moving a running service between subnets, costs far more than getting the structure right while the platform holds no customer data. The platform must also fit a $200 credit budget during the course, so the foundation has to be cheap to run and quick to destroy.

### Decision
**Network.** One VPC (`northstar-dev-vpc`, 10.0.0.0/16) in us-east-1 with a single public subnet (10.0.100.0/24) in us-east-1a, an internet gateway, and one route table sending 0.0.0.0/0 to the gateway. The security group accepts inbound traffic only from the VPC CIDR and allows all outbound traffic. Studio sits in a public subnet because it must pull container images and reach S3, and a NAT gateway would cost about $32–45 per month, a large share of a $200 budget, for a lab with no real customer data. Lab 2 moves Studio to private subnets.

**Storage.** One bucket, `northstar-dev-data-{account-id}`, with four prefixes that follow the data lifecycle: `raw/`, `processed/`, `features/`, `artifacts/`. One bucket lets IAM grant each system only the stage it owns, while churn scores that feed offer targeting (the top 10% highest-risk customers) stay in one place instead of being copied between buckets. Versioning protects the source data churn labels depend on from a bad ETL run, encryption is SSE-S3 (AES-256), and all four public-access blocks are on because the contents will be customer PII.

**Identity.** One role, `northstar-dev-MLEngineer`, trusted only by `sagemaker.amazonaws.com`. It can run training jobs, endpoints, and the model registry, and can read and write only `artifacts/` and `features/`. It cannot write `raw/` or `processed/`, so a model developer cannot alter the data the churn labels are computed from. Object actions and `ListBucket` are separate policy statements, because a trailing wildcard on the bucket ARN would silently grant writes to `raw/`. The DataEngineer and ModelMonitor roles arrive in Lab 2.

**Development environment.** A SageMaker Studio domain with IAM authentication, notebook output sharing disabled (notebooks can contain customer rows), and `ml.t3.medium` as the default instance. The stack exists twice: built by hand to learn each resource, then in Terraform with S3 remote state and a DynamoDB lock so it rebuilds with one command.

### Consequences

#### What this makes easy
- Adding the DataEngineer role in Lab 2 is one new policy scoped to `raw/`, `processed/`, and `features/`, with no bucket restructuring.
- The 19-resource stack rebuilds with one `terraform apply` in about 10 minutes (the Studio domain dominates) and destroys in 1–2 minutes, keeping Lab 1 near the $3–6 estimate.
- The weekly churn job can rerun from `features/` without touching raw data, protecting the Monday 6 AM ET deadline.

#### What this makes harder
- The public subnet leaves outbound traffic open, so a compromised notebook could send customer data anywhere. That is acceptable with simulated data and unacceptable with real PII under GDPR and CCPA.
- One availability zone cannot meet the agent's 99.5% target. Across 14 business hours a day for 30 days (420 hours), 0.5% allows only about 2.1 hours of downtime a month, and one AZ outage can exceed that.
- One bucket concentrates risk: a mistaken bucket policy exposes all three systems' data at once, and prefix isolation is only as strong as the ARN patterns in the policy.

#### What would cause you to revisit this decision
- Real customer PII lands in `raw/`, which forces private subnets and VPC endpoints.
- The offer system needs a live endpoint with the 2-second response target, which requires multiple AZs.
- Monthly spend approaches the $200 credit limit, or the 24-month raw-retention rule needs S3 lifecycle policies.

### Alternative Considered
Separate buckets, or separate AWS accounts, for each of the three systems. This gives a smaller blast radius and clean cost attribution, which the CFO wants for cost per AI interaction. I rejected it because the systems are coupled by data: churn scores decide which customers the offer system targets, and the agent needs the same customer records. Separate accounts would copy PII into three places, triple the retention and deletion work under GDPR and CCPA, and add cross-account role plumbing that a $200 course budget cannot justify. Tags and prefix-scoped IAM give most of the cost attribution without the duplication.

### AWS Service Selection
- **Networking isolation:** a dedicated VPC with a security group rather than the default VPC, because the platform needs a boundary it controls before any PII arrives.
- **Storage:** S3 rather than EFS or a copy in Snowflake, because it is durable at about $0.023 per GB-month, supports prefix-scoped IAM, and is the native input for SageMaker training and Glue.
- **Identity:** IAM roles assumed by services rather than IAM users with access keys, so no long-lived credentials live in notebooks or code.
- **ML development environment:** SageMaker Studio rather than self-managed EC2 Jupyter, because it provides managed kernels at about $0.05 per hour plus built-in access to training, the model registry, and experiment tracking.
