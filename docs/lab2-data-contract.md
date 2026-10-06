## Data Contract: processed/customers

| Field | Value |
|-------|-------|
| Dataset | `s3://northstar-dev-data-<account-id>/processed/customers/` |
| Format | Parquet (Snappy), written in overwrite mode |
| Contract version | 1.0 |
| Owner | NorthStar data engineering (role `northstar-dev-DataEngineer`) |

### Producer
Team / process: Glue ETL job `northstar-dev-transform`, running as `northstar-dev-DataEngineer`.

It reads the catalog table `northstar_dev.customers` (registered by crawler `northstar-dev-raw-crawler` from `raw/customers/`), then:
1. trims whitespace and converts empty strings to nulls,
2. parses `purchase_date` from both `yyyy-MM-dd` and `MM/dd/yyyy`,
3. casts every column to the schema below and drops rows with a null `customer_id`,
4. imputes nulls (numeric columns -> column median, string columns -> `'unknown'`),
5. removes duplicate `transaction_id` rows (keeps latest `purchase_date`, then highest `order_value`).

The job asserts the first three quality guarantees below before writing; if any fails, the job fails and nothing is published.

### Consumers
- Feature engineering job `northstar-dev-feature-engineer` (aggregates to one row per customer and writes `features/customers/` + Feature Store)
- (Future) Direct model training in Lab 3 (`northstar-dev-MLEngineer`)

### Grain
One row per transaction. A customer appears on many rows. Collapsing to one row per customer is the consumer's responsibility (feature engineering), never the producer's.

### Schema
| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `transaction_id` | string | No | Natural key, `TXN-` + 12 alphanumeric characters. Unique in this dataset. |
| `customer_id` | string | No | `CUST-` + 8 digits. Repeats across rows by design. Whitespace trimmed. |
| `purchase_date` | date | No | Calendar date of purchase, stored as a Parquet DATE (ISO 8601, `yyyy-MM-dd`). |
| `order_value` | double | No | Gross order value in USD. Nulls in raw data imputed with the column median. |
| `num_items` | int | No | Number of line items in the order. Nulls imputed with the rounded median. |
| `payment_method` | string | No | One of `credit_card`, `debit_card`, `gift_card`, `cash`, or `unknown`. |
| `channel` | string | No | `online`, `store`, or `unknown`. |
| `store_id` | string | No | `STORE-` + 3 digits for in-store orders, `ONLINE` for online orders, or `unknown`. |
| `product_category` | string | No | Primary category: `Apparel`, `Beauty`, `Electronics`, `Footwear`, `Grocery`, `Home`, `Outdoor`, `Toys`, or `unknown` (imputed). |

### Quality Guarantees
Each guarantee is a measurable assertion a consumer can check against the published Parquet.

| # | Guarantee | How to check |
|---|-----------|--------------|
| 1 | `customer_id` is never null | `count(*) where customer_id is null` = 0 |
| 2 | No duplicate `transaction_id` rows (a `customer_id` repeating across rows is expected, not a defect) | `count(distinct transaction_id)` = `count(*)` |
| 3 | `purchase_date` is a valid ISO 8601 date, never null, between 2025-04-01 and 2026-06-30 inclusive | `min`/`max` of `purchase_date`; null count = 0 |
| 4 | `order_value` is between 15.00 and 620.00 USD | `min(order_value)` >= 15.00 and `max(order_value)` <= 620.00 |
| 5 | `num_items` is an integer between 1 and 9 | `min(num_items)` >= 1 and `max(num_items)` <= 9 |
| 6 | No nulls in any column (all imputed or dropped upstream) | total null count across all columns = 0 |
| 7 | Categorical columns contain only the values listed in the schema | `distinct channel`, `payment_method`, `product_category` are subsets of the allowed sets |
| 8 | Transaction-level grain is preserved | `count(*)` > `count(distinct customer_id)` (expected ~157,600 rows across ~11,400 customers) |
| 9 | Row count is within 2% of expected for the current sample (~157,600 rows) | `count(*)` between 154,450 and 160,750 |

### SLA
- Data is available in `processed/customers/` within 2 hours of landing in `raw/customers/`.
- A run that fails any quality guarantee publishes nothing; the previous version of the dataset remains in place (S3 versioning is enabled, noncurrent versions retained 30 days by lifecycle rule `expire-processed-versions`).
- Failures are visible as `JobRunState: FAILED` on `northstar-dev-transform` with the failed assertion in CloudWatch Logs (`/aws-glue/jobs/`).

### Versioning
- Schema changes require a new S3 prefix (e.g., `processed/customers/v2/`); the existing prefix keeps its current schema until all consumers have migrated.
- Breaking changes (removing or renaming a column, changing a type, changing the grain, tightening nullability) require consumer notification 5 business days in advance.
- Non-breaking changes (adding a new nullable column, widening an allowed value set) are announced in this document and bump the minor contract version (1.0 -> 1.1).
