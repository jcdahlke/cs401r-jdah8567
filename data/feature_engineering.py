"""
NorthStar Retail -- feature engineering as testable pandas functions (Lab 4).

This is a line-for-line pandas port of the Spark logic in
glue-scripts/feature_engineer.py (compute_rfm_features, assign_loyalty_tier,
compute_churn_proxy, attach_churn_label). Same constants, same window rule,
same definitions, so tests/test_data.py can recompute features from
processed/ and check they equal what the Glue job wrote to features/.

Window rule (identical to the Glue job):
    d = (T - purchase_date) in whole days
    "in the last N days"  <=>  d < N       (i.e. purchase_date > T - N days)
Only purchases with purchase_date <= T are ever used for features. Anything
after T is holdout and feeds the label only.

Null-filling strategy (customers in the roster with NO purchase on or before T):
    days_since_last_purchase  -> RECENCY_SENTINEL (999): "never bought" must
                                 look maximally dormant, not maximally fresh
    customer_tenure_days      -> 0    (no history yet)
    every count / sum         -> 0    (they genuinely bought nothing)
    avg_order_value,
    avg_basket_size_6m        -> 0    (guarded divide, same as the Glue job)
    category_diversity_score,
    online_to_store_ratio     -> 0    (no categories, no orders)
Customers with at least one purchase never produce a NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Must match glue-scripts/feature_engineer.py
FEATURE_CUTOFF = pd.Timestamp("2026-04-01")
SNAPSHOT = pd.Timestamp("2026-06-30")
N_CATEGORIES = 8.0
TIER_BRONZE_MAX = 500.0
TIER_SILVER_MAX = 2000.0
TIER_GOLD_MAX = 5000.0
RISK_HIGH_LO, RISK_MED_LO, RISK_LOW_HI = 0.7, 0.4, 0.4

RECENCY_SENTINEL = 999

FEATURE_COLUMNS = [
    "days_since_last_purchase", "customer_tenure_days",
    "purchase_frequency_30d", "purchase_frequency_90d", "purchase_frequency_180d",
    "avg_order_value", "total_spend_90d", "total_lifetime_value",
    "avg_basket_size_6m", "category_diversity_score", "online_to_store_ratio",
]
ROUNDED_COLUMNS = ["avg_order_value", "total_lifetime_value", "total_spend_90d",
                   "avg_basket_size_6m", "category_diversity_score",
                   "online_to_store_ratio"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _cutoff(snapshot_date) -> pd.Timestamp:
    return pd.Timestamp(snapshot_date if snapshot_date is not None else FEATURE_CUTOFF).normalize()


def _history(transactions: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Observation window only (purchase_date <= T), with d = days before T."""
    T = _cutoff(snapshot_date)
    tx = transactions.copy()
    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"]).dt.normalize()
    tx = tx[tx["purchase_date"] <= T]
    tx["_d"] = (T - tx["purchase_date"]).dt.days
    return tx


def _ids(customer_ids) -> pd.Index | None:
    if customer_ids is None:
        return None
    if isinstance(customer_ids, pd.DataFrame):
        customer_ids = customer_ids["customer_id"]
    return pd.Index(pd.Series(customer_ids).unique(), name="customer_id")


def _finish(out: pd.DataFrame, customer_ids, fill: dict) -> pd.DataFrame:
    """Reindex onto the roster (if given), apply the null strategy, flatten."""
    ids = _ids(customer_ids)
    if ids is not None:
        out = out.reindex(out.index.union(ids))
    out.index.name = "customer_id"
    out = out.fillna(fill).astype("float64")
    return out.reset_index()


# ── feature groups ───────────────────────────────────────────────────────────

def compute_recency_features(transactions, snapshot_date=None, customer_ids=None,
                             sentinel=RECENCY_SENTINEL) -> pd.DataFrame:
    """days_since_last_purchase, customer_tenure_days (both measured to T)."""
    h = _history(transactions, snapshot_date)
    g = h.groupby("customer_id")["_d"]
    out = pd.DataFrame({
        "days_since_last_purchase": g.min(),   # T - last purchase
        "customer_tenure_days": g.max(),       # T - first purchase
    })
    return _finish(out, customer_ids,
                   {"days_since_last_purchase": sentinel, "customer_tenure_days": 0})


def compute_frequency_features(transactions, snapshot_date=None, customer_ids=None) -> pd.DataFrame:
    """Order counts in the 30/90/180 days before T."""
    h = _history(transactions, snapshot_date)
    out = pd.DataFrame({
        f"purchase_frequency_{n}d": (h["_d"] < n).groupby(h["customer_id"]).sum()
        for n in (30, 90, 180)
    })
    return _finish(out, customer_ids, {c: 0 for c in out.columns})


def compute_monetary_features(transactions, snapshot_date=None, customer_ids=None) -> pd.DataFrame:
    """avg_order_value, total_spend_90d, total_lifetime_value, avg_basket_size_6m.

    avg_basket_size_6m is ITEMS per order over the last 180 days, not dollars,
    with a guarded denominator (0 if no orders in that window).
    """
    h = _history(transactions, snapshot_date)
    by = h.groupby("customer_id")
    in90, in180 = h["_d"] < 90, h["_d"] < 180
    items180 = h["num_items"].where(in180, 0).groupby(h["customer_id"]).sum()
    orders180 = in180.groupby(h["customer_id"]).sum()
    out = pd.DataFrame({
        "avg_order_value": by["order_value"].mean(),
        "total_spend_90d": h["order_value"].where(in90, 0.0).groupby(h["customer_id"]).sum(),
        "total_lifetime_value": by["order_value"].sum(),
        "avg_basket_size_6m": (items180 / orders180.replace(0, np.nan)).fillna(0.0),
    })
    return _finish(out, customer_ids, {c: 0.0 for c in out.columns})


def compute_behavioral_features(transactions, snapshot_date=None, customer_ids=None) -> pd.DataFrame:
    """category_diversity_score (distinct non-'unknown' categories / 8, capped
    at 1) and online_to_store_ratio (fraction of orders with channel == online)."""
    h = _history(transactions, snapshot_date)
    known = h["product_category"].where(h["product_category"] != "unknown")
    ncat = known.groupby(h["customer_id"]).nunique()
    out = pd.DataFrame({
        "category_diversity_score": np.minimum(ncat / N_CATEGORIES, 1.0),
        "online_to_store_ratio": (h["channel"] == "online").astype(float)
                                 .groupby(h["customer_id"]).mean(),
    })
    return _finish(out, customer_ids, {c: 0.0 for c in out.columns})


# ── derived columns ──────────────────────────────────────────────────────────

def assign_loyalty_tier(df: pd.DataFrame) -> pd.DataFrame:
    v = df["total_lifetime_value"]
    out = df.copy()
    out["loyalty_tier"] = np.select(
        [v < TIER_BRONZE_MAX, v < TIER_SILVER_MAX, v < TIER_GOLD_MAX],
        ["Bronze", "Silver", "Gold"], default="Platinum")
    return out


def compute_churn_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based churn_risk_score in [0, 1] (the Lab 2 heuristic, not the label)."""
    d, f30 = df["days_since_last_purchase"], df["purchase_frequency_30d"]
    score = np.select(
        [(d > 60) & (f30 == 0), d > 30],
        [RISK_HIGH_LO + 0.3 * np.minimum((d - 60) / 305.0, 1.0),
         RISK_MED_LO + 0.3 * np.minimum((d - 30) / 30.0, 1.0)],
        default=RISK_LOW_HI * np.minimum(d / 30.0, 1.0))
    out = df.copy()
    out["churn_risk_score"] = np.round(np.clip(score, 0.0, 1.0), 4)
    return out


def attach_churn_label(features: pd.DataFrame, transactions: pd.DataFrame,
                       snapshot_date=None, holdout_end=SNAPSHOT) -> pd.DataFrame:
    """churn_label = 1 if NO purchase in (T, holdout_end]. Never from a feature."""
    T = _cutoff(snapshot_date)
    dates = pd.to_datetime(transactions["purchase_date"]).dt.normalize()
    active = set(transactions.loc[(dates > T) & (dates <= pd.Timestamp(holdout_end)),
                                  "customer_id"])
    out = features.copy()
    out["churn_label"] = (~out["customer_id"].isin(active)).astype(int)
    return out


# ── full feature set ─────────────────────────────────────────────────────────

def build_churn_feature_set(customers=None, transactions=None, snapshot_date=None,
                            holdout_end=None) -> pd.DataFrame:
    """One row per customer: customer_id, the 11 model features, loyalty_tier,
    churn_risk_score, and churn_label if holdout_end is given.

    The roster is `customers` (if given) plus everyone with a purchase on or
    before T. Roster customers with no history get the null strategy in the
    module docstring.
    """
    T = _cutoff(snapshot_date)
    roster = _ids(customers)
    hist_ids = _ids(_history(transactions, T)["customer_id"])
    ids = hist_ids if roster is None else roster.union(hist_ids)

    parts = [f(transactions, T, ids) for f in (compute_recency_features,
                                               compute_frequency_features,
                                               compute_monetary_features,
                                               compute_behavioral_features)]
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="customer_id", how="left")
    out[ROUNDED_COLUMNS] = out[ROUNDED_COLUMNS].round(4)
    out = compute_churn_proxy(assign_loyalty_tier(out))
    if holdout_end is not None:
        out = attach_churn_label(out, transactions, T, holdout_end)
    return out.sort_values("customer_id").reset_index(drop=True)
