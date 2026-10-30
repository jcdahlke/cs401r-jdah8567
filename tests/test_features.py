"""
NorthStar Retail — Feature Engineering Unit Tests
CS 401R Lab 4 Starter Kit

These tests validate your Lab 2 feature engineering pipeline — the code that
transforms raw transaction and customer data into the 11 model features
used by the churn prediction model.

Run locally:
    pytest tests/test_features.py -v

Design principle: Unit test the feature logic in isolation using small,
controlled DataFrames — do not depend on the full synthetic dataset.
"""

from datetime import datetime, timedelta

import os

import numpy as np
import pandas as pd
import pytest

# ── Import your feature engineering module ─────────────────────────────────────
# Lab 2 implements feature engineering in PySpark (glue-scripts/feature_engineer.py),
# which is awkward to unit test. For Lab 4 you extract the pure computation logic into
# testable functions in data/feature_engineering.py that operate on pandas DataFrames,
# then have the Glue job and these tests share that logic.
#
# Your pandas functions must compute EXACTLY what your Glue job computes --
# same windows (purchase_date > T - N days), same definitions. In particular,
# avg_basket_size_6m is ITEMS per order over 180 days (Lab 2 schema), not
# dollars. Check the port against your Glue output on the full dataset.
#
# That refactor IS the assignment here: code you cannot test in isolation is code you
# cannot gate a pipeline on.
try:
    from data.feature_engineering import (
        compute_recency_features,
        compute_frequency_features,
        compute_monetary_features,
        compute_behavioral_features,
        build_churn_feature_set,
        assign_loyalty_tier,
    )
    FEATURES_MODULE_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as e:
    FEATURES_MODULE_AVAILABLE = False
    _IMPORT_ERROR = e

# A missing module is a FAILURE, not a skip. Silently skipping means `pytest tests/`
# goes green having tested nothing, and a CI gate that cannot fail is not a gate.
# Set ALLOW_MISSING_FEATURES_MODULE=1 only while you are still building it.
if not FEATURES_MODULE_AVAILABLE and os.environ.get("ALLOW_MISSING_FEATURES_MODULE") != "1":
    pytest.fail(
        f"Could not import data.feature_engineering ({_IMPORT_ERROR!r}). Lab 4 Task 1 "
        "requires the Lab 2 feature logic extracted into testable pandas functions in "
        "data/feature_engineering.py. If the file exists, check that pytest.ini is at the "
        "repo root and that the module itself imports cleanly. Set "
        "ALLOW_MISSING_FEATURES_MODULE=1 to defer this while developing.",
        pytrace=False,
    )

pytestmark = pytest.mark.skipif(
    not FEATURES_MODULE_AVAILABLE,
    reason="feature module unavailable and ALLOW_MISSING_FEATURES_MODULE=1 was set"
)

# Must match FEATURE_CUTOFF in the Lab 2 feature job. Windows are measured
# backwards from here, never from today().
SNAPSHOT_DATE = datetime(2026, 4, 1)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_customers():
    """Customer roster. Matches the customer_id format Lab 2 generates."""
    return pd.DataFrame({
        "customer_id": ["CUST-10000001", "CUST-10000002", "CUST-10000003"],
        "loyalty_tier": ["Bronze", "Silver", "Platinum"],
        "churn_label": [1, 0, 0],
    })


@pytest.fixture
def sample_transactions():
    """Small transaction set at PROCESSED grain.

    Columns match Lab 2's transform output exactly:
        transaction_id, customer_id, purchase_date, order_value,
        num_items, payment_method, channel, store_id, product_category

    Three deliberately different customers:
      CUST-10000001  lapsing  - last purchase 60d before T, 2 buys, one category, store only
      CUST-10000002  active   - last purchase 5d before T, 8 buys, two categories, mixed channel
      CUST-10000003  high value - last purchase 30d before T, large baskets, four categories
    """
    T = SNAPSHOT_DATE
    rows = []

    def tx(cid, days_ago, value, items, channel, category, n):
        return {
            "transaction_id": f"TXN-{cid[-4:]}{n:08d}",
            "customer_id": cid,
            "purchase_date": (T - timedelta(days=days_ago)).isoformat(),
            "order_value": value,
            "num_items": items,
            "payment_method": "credit_card",
            "channel": channel,
            "store_id": "ONLINE" if channel == "online" else "STORE-042",
            "product_category": category,
        }

    # Customer 1 - lapsing: 2 purchases, both inside 90d, single category, store only
    rows.append(tx("CUST-10000001", 60, 89.99, 2, "store", "Apparel", 1))
    rows.append(tx("CUST-10000001", 85, 45.00, 1, "store", "Apparel", 2))

    # Customer 2 - active: 8 purchases inside 90d, alternating channel, two categories
    for i in range(8):
        rows.append(tx("CUST-10000002", 5 + i * 10, 55.00 + i * 5, 2,
                       "online" if i % 2 == 0 else "store",
                       "Footwear" if i % 2 == 0 else "Outdoor", 10 + i))

    # Customer 3 - high value: large baskets across four categories
    for i, cat in enumerate(["Electronics", "Home", "Beauty", "Grocery"]):
        rows.append(tx("CUST-10000003", 30 + i * 25, 365.00, 5,
                       "online" if i % 2 == 0 else "store", cat, 20 + i))

    return pd.DataFrame(rows)


class TestRecencyFeatures:

    def test_days_since_last_purchase_correct(self, sample_customers, sample_transactions):
        features = compute_recency_features(sample_transactions, SNAPSHOT_DATE)
        cust1 = features[features["customer_id"] == "CUST-10000001"].iloc[0]
        # Last purchase 60 days ago
        assert 58 <= cust1["days_since_last_purchase"] <= 62, (
            f"Expected ~60, got {cust1['days_since_last_purchase']}"
        )

    def test_recent_buyer_low_recency(self, sample_customers, sample_transactions):
        features = compute_recency_features(sample_transactions, SNAPSHOT_DATE)
        cust2 = features[features["customer_id"] == "CUST-10000002"].iloc[0]
        assert cust2["days_since_last_purchase"] <= 10, (
            f"Recent buyer should have low recency, got {cust2['days_since_last_purchase']}"
        )

    def test_no_negative_recency(self, sample_transactions):
        features = compute_recency_features(sample_transactions, SNAPSHOT_DATE)
        assert (features["days_since_last_purchase"] >= 0).all(), (
            "days_since_last_purchase should never be negative"
        )

    def test_customer_with_no_transactions_returns_sentinel(self, sample_customers):
        """Customer with no transactions should get a sentinel value (e.g., 999)."""
        empty_txns = pd.DataFrame(columns=["customer_id", "purchase_date", "order_value"])
        customer_ids = pd.DataFrame({"customer_id": ["CUST-NODATA"]})

        # Boundary case: a customer with ZERO purchases gets the 999 sentinel
        # (maximally dormant) rather than NaN or 0 (which would read as "just bought").
        features = compute_recency_features(empty_txns, SNAPSHOT_DATE, customer_ids)
        assert len(features) == 1
        assert features.iloc[0]["customer_id"] == "CUST-NODATA"
        assert features.iloc[0]["days_since_last_purchase"] == 999
        assert features.iloc[0]["customer_tenure_days"] == 0


# ── Frequency Feature Tests ────────────────────────────────────────────────────

class TestFrequencyFeatures:

    def test_frequency_90d_correct(self, sample_transactions):
        features = compute_frequency_features(sample_transactions, SNAPSHOT_DATE)
        cust2 = features[features["customer_id"] == "CUST-10000002"].iloc[0]
        # Customer 2 has 8 transactions, all in last 80 days (within 90-day window)
        assert cust2["purchase_frequency_90d"] == 8, (
            f"Expected 8 purchases in 90 days, got {cust2['purchase_frequency_90d']}"
        )

    def test_frequency_90d_excludes_older_transactions(self, sample_transactions):
        """Customer 1 has 2 purchases — one at 60 days (in window), one at 85 days (in window)."""
        features = compute_frequency_features(sample_transactions, SNAPSHOT_DATE)
        cust1 = features[features["customer_id"] == "CUST-10000001"].iloc[0]
        assert cust1["purchase_frequency_90d"] == 2

    def test_frequency_is_non_negative(self, sample_transactions):
        features = compute_frequency_features(sample_transactions, SNAPSHOT_DATE)
        assert (features["purchase_frequency_90d"] >= 0).all()
        assert (features["purchase_frequency_180d"] >= 0).all()

    def test_frequency_180d_gte_frequency_90d(self, sample_transactions):
        """180-day frequency must always be >= 90-day frequency."""
        features = compute_frequency_features(sample_transactions, SNAPSHOT_DATE)
        violations = features[features["purchase_frequency_180d"] < features["purchase_frequency_90d"]]
        assert len(violations) == 0, (
            f"{len(violations)} customers have 180d_freq < 90d_freq (impossible)"
        )


# ── Monetary Feature Tests ─────────────────────────────────────────────────────

class TestMonetaryFeatures:

    def test_avg_basket_size_correct(self, sample_transactions):
        """avg_basket_size_6m = items bought in the last 180 days / orders in
        the last 180 days (Lab 2: "Mean items per order, last 180 days").
        Customer 3's four orders each have num_items=5, so 5.0. The dollar
        figure, 365, is avg_order_value."""
        features = compute_monetary_features(sample_transactions, SNAPSHOT_DATE)
        cust3 = features[features["customer_id"] == "CUST-10000003"].iloc[0]
        assert cust3["avg_basket_size_6m"] == pytest.approx(5.0), (
            f"Expected 5.0 items per order, got {cust3['avg_basket_size_6m']}"
        )
        assert cust3["avg_order_value"] == pytest.approx(365.0)

    def test_total_spend_90d_is_sum_not_average(self, sample_transactions):
        features = compute_monetary_features(sample_transactions, SNAPSHOT_DATE)
        cust2 = features[features["customer_id"] == "CUST-10000002"].iloc[0]
        # Customer 2: 8 transactions of $55-$90 in 90 days
        assert cust2["total_spend_90d"] > 400, (
            f"total_spend_90d should be sum, not average. Got {cust2['total_spend_90d']}"
        )

    def test_monetary_features_non_negative(self, sample_transactions):
        features = compute_monetary_features(sample_transactions, SNAPSHOT_DATE)
        assert (features["avg_basket_size_6m"] >= 0).all()
        assert (features["total_spend_90d"] >= 0).all()


# ── Behavioral Feature Tests ───────────────────────────────────────────────────

class TestBehavioralFeatures:

    def test_category_diversity_score_range(self, sample_transactions):
        """Category diversity should be between 0 and 1."""
        features = compute_behavioral_features(sample_transactions)
        assert (features["category_diversity_score"] >= 0).all()
        assert (features["category_diversity_score"] <= 1).all()

    def test_single_category_buyer_has_low_diversity(self, sample_transactions):
        features = compute_behavioral_features(sample_transactions)
        cust1 = features[features["customer_id"] == "CUST-10000001"].iloc[0]
        # Customer 1 only bought Apparel
        assert cust1["category_diversity_score"] < 0.3, (
            f"Single-category buyer should have low diversity, got {cust1['category_diversity_score']}"
        )

    def test_multi_category_buyer_has_high_diversity(self, sample_transactions):
        features = compute_behavioral_features(sample_transactions)
        cust3 = features[features["customer_id"] == "CUST-10000003"].iloc[0]
        # Customer 3 bought four of the eight categories: 4 / 8 = 0.5
        assert cust3["category_diversity_score"] > 0.3, (
            f"Multi-category buyer should have higher diversity"
        )


    def test_online_to_store_ratio_non_negative(self, sample_transactions):
        features = compute_behavioral_features(sample_transactions)
        assert (features["online_to_store_ratio"] >= 0).all()


# ── Engagement Feature Tests ───────────────────────────────────────────────────


class TestFullFeatureSet:

    REQUIRED_FEATURES = [
        "days_since_last_purchase",
        "purchase_frequency_90d",
        "purchase_frequency_180d",
        "avg_basket_size_6m",
        "total_spend_90d",
        "category_diversity_score",
        "online_to_store_ratio",
        "customer_tenure_days",
    ]

    def test_all_required_features_present(self, sample_customers, sample_transactions):
        """build_churn_feature_set must produce every feature listed above."""
        features = build_churn_feature_set(
            customers=sample_customers,
            transactions=sample_transactions,
            snapshot_date=SNAPSHOT_DATE,
        )
        missing = [f for f in self.REQUIRED_FEATURES if f not in features.columns]
        assert not missing, f"Missing features: {missing}"

    def test_no_null_features_for_known_customers(self, sample_customers, sample_transactions):
        """
        TODO: After filling missing values, no feature should be NaN for customers
        who have at least one transaction. Document your null-filling strategy.
        """
        features = build_churn_feature_set(
            customers=sample_customers,
            transactions=sample_transactions,
            snapshot_date=SNAPSHOT_DATE,
        )
        # Check customers with transactions
        active_customer_ids = sample_transactions["customer_id"].unique()
        active_features = features[features["customer_id"].isin(active_customer_ids)]
        null_counts = active_features[self.REQUIRED_FEATURES].isna().sum()
        # Null strategy (documented in data/feature_engineering.py): customers
        # with history never produce NaN; roster-only customers get recency=999
        # and 0 for every count, sum, average and ratio.
        assert null_counts.sum() == 0, f"NaN features for active customers: {null_counts[null_counts > 0].to_dict()}"

    def test_output_has_customer_id_column(self, sample_customers, sample_transactions):
        features = build_churn_feature_set(
            customers=sample_customers,
            transactions=sample_transactions,
            snapshot_date=SNAPSHOT_DATE,
        )
        assert "customer_id" in features.columns, "Output must include customer_id for joining"


# ── Boundary and edge cases ───────────────────────────────────────────────────

def _txn(cid, days_ago, value=50.0, items=2, channel="store", category="Apparel", n=1):
    return {
        "transaction_id": f"TXN-EDGE{n:06d}", "customer_id": cid,
        "purchase_date": (SNAPSHOT_DATE - timedelta(days=days_ago)).isoformat(),
        "order_value": value, "num_items": items, "payment_method": "credit_card",
        "channel": channel, "store_id": "STORE-042", "product_category": category,
    }


class TestBoundaryAndEdgeCases:

    def test_zero_purchase_customer_full_feature_set(self, sample_transactions):
        """Boundary: a roster customer with 0 purchases gets the sentinel recency
        and zeros everywhere else -- no NaN, no divide-by-zero."""
        roster = pd.DataFrame({"customer_id": ["CUST-10000001", "CUST-ZERO"]})
        features = build_churn_feature_set(customers=roster,
                                           transactions=sample_transactions,
                                           snapshot_date=SNAPSHOT_DATE)
        zero = features[features["customer_id"] == "CUST-ZERO"].iloc[0]
        assert zero["days_since_last_purchase"] == 999
        for col in ["purchase_frequency_30d", "purchase_frequency_90d",
                    "purchase_frequency_180d", "total_spend_90d",
                    "total_lifetime_value", "avg_basket_size_6m",
                    "avg_order_value", "category_diversity_score",
                    "online_to_store_ratio", "customer_tenure_days"]:
            assert zero[col] == 0, f"{col} should be 0 for a zero-purchase customer, got {zero[col]}"
        assert zero["loyalty_tier"] == "Bronze"

    def test_single_transaction_customer(self):
        """Edge: exactly one purchase. Recency == tenure, every window that
        contains it counts 1, basket size equals that order's items."""
        txns = pd.DataFrame([_txn("CUST-ONE", 10, value=120.0, items=3,
                                  channel="online", category="Home")])
        f = build_churn_feature_set(transactions=txns, snapshot_date=SNAPSHOT_DATE).iloc[0]
        assert f["days_since_last_purchase"] == 10
        assert f["customer_tenure_days"] == 10
        assert (f["purchase_frequency_30d"], f["purchase_frequency_90d"],
                f["purchase_frequency_180d"]) == (1, 1, 1)
        assert f["avg_basket_size_6m"] == pytest.approx(3.0)
        assert f["avg_order_value"] == pytest.approx(120.0)
        assert f["total_lifetime_value"] == pytest.approx(120.0)
        assert f["category_diversity_score"] == pytest.approx(1 / 8)
        assert f["online_to_store_ratio"] == pytest.approx(1.0)

    def test_single_old_transaction_has_zero_basket_not_nan(self):
        """Edge: one purchase older than 180 days. The 6-month basket has a zero
        denominator and must be guarded to 0, exactly like the Glue job."""
        txns = pd.DataFrame([_txn("CUST-OLD", 200, items=4)])
        f = build_churn_feature_set(transactions=txns, snapshot_date=SNAPSHOT_DATE).iloc[0]
        assert f["purchase_frequency_180d"] == 0
        assert f["avg_basket_size_6m"] == 0.0
        assert f["total_lifetime_value"] == pytest.approx(50.0)

    def test_window_boundaries_match_glue_job(self):
        """Boundary: 'last N days' means purchase_date > T - N days (d < N),
        the same rule as glue-scripts/feature_engineer.py. A purchase exactly
        90 days before T is outside the 90d window; 89 days is inside."""
        txns = pd.DataFrame([_txn("CUST-B", 89, n=1), _txn("CUST-B", 90, n=2),
                             _txn("CUST-B", 29, n=3), _txn("CUST-B", 30, n=4)])
        f = compute_frequency_features(txns, SNAPSHOT_DATE).iloc[0]
        assert f["purchase_frequency_30d"] == 1
        assert f["purchase_frequency_90d"] == 3
        assert f["purchase_frequency_180d"] == 4

    def test_post_cutoff_purchases_are_ignored(self, sample_transactions):
        """Leakage guard: holdout purchases (after T) must not move any feature."""
        later = pd.DataFrame([_txn("CUST-10000001", -10, value=999.0, items=9,
                                   channel="online", category="Electronics", n=99)])
        before = build_churn_feature_set(transactions=sample_transactions,
                                         snapshot_date=SNAPSHOT_DATE)
        after = build_churn_feature_set(transactions=pd.concat([sample_transactions, later]),
                                        snapshot_date=SNAPSHOT_DATE)
        pd.testing.assert_frame_equal(before, after)

    def test_unknown_category_not_counted(self):
        """The transform imputes 'unknown' for missing categories; counting it
        would inflate diversity (and can push it past 1.0)."""
        txns = pd.DataFrame([_txn("CUST-U", 5, category="unknown", n=1),
                             _txn("CUST-U", 6, category="Apparel", n=2)])
        f = compute_behavioral_features(txns, SNAPSHOT_DATE).iloc[0]
        assert f["category_diversity_score"] == pytest.approx(1 / 8)

    def test_loyalty_tier_thresholds(self):
        """Tier boundaries are inclusive on the lower edge: 500 is Silver."""
        df = pd.DataFrame({"total_lifetime_value": [0.0, 499.99, 500.0, 1999.99,
                                                    2000.0, 4999.99, 5000.0]})
        tiers = assign_loyalty_tier(df)["loyalty_tier"].tolist()
        assert tiers == ["Bronze", "Bronze", "Silver", "Silver", "Gold", "Gold", "Platinum"]

    def test_churn_label_comes_from_holdout_only(self, sample_transactions):
        """Normal case for the label: anyone with a purchase in (T, SNAPSHOT]
        is retained; everyone else churned."""
        later = pd.DataFrame([_txn("CUST-10000002", -20, n=98)])
        f = build_churn_feature_set(transactions=pd.concat([sample_transactions, later]),
                                    snapshot_date=SNAPSHOT_DATE,
                                    holdout_end=pd.Timestamp("2026-06-30"))
        labels = dict(zip(f["customer_id"], f["churn_label"]))
        assert labels == {"CUST-10000001": 1, "CUST-10000002": 0, "CUST-10000003": 1}
