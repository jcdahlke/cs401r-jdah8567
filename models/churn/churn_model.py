"""Your Lab 3 churn model, packaged for the Lab 4 pipeline.

YOU WRITE THIS FILE. The pipeline's training job runs train_sagemaker.py
(beside this file; do not edit it), which imports this module and calls
train(). Everything the job needs from you goes through the three names below.

Contract (train_sagemaker.py checks it in the first seconds of the job):

    FEATURE_COLUMNS   list[str]   model input columns, in order
    LABEL             str         "churn_label"
    train(df, seed=42) -> (xgboost.Booster, dict)

`df` is the full labeled feature table: every column of your Lab 2 output,
including `loyalty_tier` for slice evaluation. It is already loaded. Do not
query Athena or the Feature Store from here; the training container stages
the data for you and has no business making service calls for its input.

The metrics dict must include the four numbers the CI gate in
tests/test_model.py reads. Your Lab 3 evaluate_model() already returns them:

    precision_top10, recall_top10      gated at 0.50 / 0.25
    lift_ci_low, lift_ci_high          bootstrap 95% CI on AUC lift over the
                                       recency-only baseline; lift_ci_low > 0

Include anything else you reported in Lab 3 (auc_roc, baseline_auc,
positive_rate_val, slice results). It is all written to evaluation_metrics.json
and attached to the registered model.

What runs in the container: the SageMaker XGBoost 1.7-1 image ships pandas,
numpy, scikit-learn, xgboost and boto3. It does NOT ship mlflow or sagemaker,
so do not import them at the top of this file. MLflow tracking stays in your
Lab 3 script.

Only this directory (models/churn/) is uploaded to the training job, so the
code must live here. `import churn_training_skeleton` will fail inside the
container even though it works on your laptop. Copy the functions you need.

Test it locally before pushing:

    python models/churn/train_sagemaker.py \
        --train-channel <dir containing your features CSV or Parquet> \
        --model-dir /tmp/model --output-data-dir /tmp/out
"""

# Keeps Lab 3 annotations like `-> tuple[np.ndarray, np.ndarray]` from being
# evaluated, whatever Python the training container runs.
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Must match the columns your Lab 3 model was trained on, in the same order.
FEATURE_COLUMNS = [
    "days_since_last_purchase",
    "customer_tenure_days",
    "purchase_frequency_30d",
    "purchase_frequency_90d",
    "purchase_frequency_180d",
    "avg_order_value",
    "total_spend_90d",
    "total_lifetime_value",
    "avg_basket_size_6m",
    "category_diversity_score",
    "online_to_store_ratio",
]

LABEL = "churn_label"

# Copied from churn_training_skeleton.py (Lab 3). Not imported: only
# models/churn/ is uploaded to the training job.
BASELINE_FEATURE = "days_since_last_purchase"
SLICE_COLUMN = "loyalty_tier"
TIER_ORDER = ["Bronze", "Silver", "Gold", "Platinum"]

# Lab 3's promoted configuration: MLflow run "xgb-d3-eta0.05-final"
# (docs/lab3-mlflow-runs.txt). scale_pos_weight is computed from the
# training split, as in Lab 3.
HYPERPARAMS = {
    "max_depth": 3,
    "eta": 0.05,
    "num_round": 400,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

EVAL_THRESHOLDS = {
    "precision_top10": 0.50,
    "recall_top10": 0.25,
}
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
FAIRNESS_GAP_PP = 0.10   # flag (not block) a Platinum-vs-Bronze recall gap above this


# ── Lab 3 functions ────────────────────────────────────────────────────────────

def preprocess_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Lab 3 preprocess_features(): sentinel -1 for missing features."""
    X = df[FEATURE_COLUMNS].fillna(-1).to_numpy(dtype=float)
    y = df[LABEL].to_numpy().astype(int)
    print(f"  Positive class rate: {y.mean():.1%}   Feature matrix: {X.shape}")
    return X, y


def train_model(X_train, y_train, X_val, y_val, seed: int = 42) -> xgb.Booster:
    """Lab 3 train_model(), hyperparameters from HYPERPARAMS."""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS)
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "objective": "binary:logistic",
        # "auc" LAST: XGBoost early-stops on the last metric (Lab 3 defect 49).
        "eval_metric": ["logloss", "auc"],
        "max_depth": HYPERPARAMS["max_depth"],
        "eta": HYPERPARAMS["eta"],
        "min_child_weight": HYPERPARAMS["min_child_weight"],
        "subsample": HYPERPARAMS["subsample"],
        "colsample_bytree": HYPERPARAMS["colsample_bytree"],
        "scale_pos_weight": spw,
        "seed": seed,
        "verbosity": 1,
    }
    print(f"Training XGBoost: {params}")
    return xgb.train(
        params=params, dtrain=dtrain,
        num_boost_round=HYPERPARAMS["num_round"],
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=20, verbose_eval=50,
    )


def train_recency_baseline(X_train, y_train, X_val) -> np.ndarray:
    """Lab 3 baseline: XGBoost on days_since_last_purchase ALONE."""
    i = FEATURE_COLUMNS.index(BASELINE_FEATURE)
    dtr = xgb.DMatrix(X_train[:, [i]], label=y_train, feature_names=[BASELINE_FEATURE])
    dva = xgb.DMatrix(X_val[:, [i]], feature_names=[BASELINE_FEATURE])
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 4, "eta": 0.1, "subsample": 0.9,
        "colsample_bytree": 0.9, "scale_pos_weight": spw, "seed": 42,
    }
    return xgb.train(params, dtr, num_boost_round=200).predict(dva)


def bootstrap_lift_ci(y_true, proba, baseline_proba, n: int = BOOTSTRAP_N,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> tuple[float, float]:
    """Paired percentile bootstrap CI for (model AUC - baseline AUC)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    idx = np.arange(len(y_true))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        ys = y_true[s]
        if len(np.unique(ys)) < 2:
            continue
        diffs.append(roc_auc_score(ys, proba[s]) - roc_auc_score(ys, baseline_proba[s]))
    if len(diffs) < n // 2:
        raise ValueError("Too few usable bootstrap replicates to form a confidence interval.")
    d = np.sort(diffs)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def evaluate_model(model: xgb.Booster, X_val, y_val, baseline_proba) -> dict:
    """Lab 3 evaluate_model(). RAISES on a failed gate, so a model that misses
    a gate fails the training step and is never registered."""
    y_pred_proba = model.predict(xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS))
    auc_roc = roc_auc_score(y_val, y_pred_proba)

    top10_mask = y_pred_proba >= np.percentile(y_pred_proba, 90)
    precision_top10 = y_val[top10_mask].mean() if top10_mask.sum() > 0 else 0.0
    recall_top10 = y_val[top10_mask].sum() / y_val.sum() if y_val.sum() > 0 else 0.0

    baseline_auc = roc_auc_score(y_val, baseline_proba)
    lift_ci_low, lift_ci_high = bootstrap_lift_ci(y_val, y_pred_proba, baseline_proba)

    metrics = {
        "auc_roc": round(float(auc_roc), 4),
        "precision_top10": round(float(precision_top10), 4),
        "recall_top10": round(float(recall_top10), 4),
        "auc_vs_baseline": round(float(auc_roc - baseline_auc), 4),
        "baseline_auc": round(float(baseline_auc), 4),
        "baseline_auc_roc": round(float(baseline_auc), 4),
        "lift_ci_low": round(float(lift_ci_low), 4),
        "lift_ci_high": round(float(lift_ci_high), 4),
        "positive_rate_val": round(float(y_val.mean()), 4),
        "n_val_samples": int(len(y_val)),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print("\n── Evaluation Results ──────────────────────────────")
    failures = []
    for metric, threshold in EVAL_THRESHOLDS.items():
        value = metrics[metric]
        print(f"  {metric:25s}: {value:.4f}  (threshold {threshold:.2f})  "
              f"{'PASS' if value >= threshold else 'FAIL'}")
        if value < threshold:
            failures.append(f"{metric}={value:.4f} < {threshold}")
    print(f"  {'auc_lift 95% CI':25s}: [{lift_ci_low:.4f}, {lift_ci_high:.4f}]  "
          f"(must exclude 0)  {'PASS' if lift_ci_low > 0 else 'FAIL'}")
    if lift_ci_low <= 0:
        failures.append(f"auc_lift 95% CI [{lift_ci_low:.4f}, {lift_ci_high:.4f}] includes zero")
    if failures:
        raise ValueError(f"Model failed evaluation thresholds: {'; '.join(failures)}. "
                         f"Do not promote to Model Registry.")
    return metrics


def evaluate_slices(model: xgb.Booster, X_val, y_val, slice_column: pd.Series) -> dict:
    """Lab 3 evaluate_slices(): per-tier AUC and recall@top10% against ONE
    global top-10% cutoff (the business has a single contact budget)."""
    y_pred_proba = model.predict(xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS))
    cutoff = np.percentile(y_pred_proba, 90)
    aggregate_recall = (y_pred_proba >= cutoff)[y_val == 1].mean()
    tiers = np.asarray(slice_column, dtype=object)
    present = list(pd.unique(tiers))
    order = [t for t in TIER_ORDER if t in present] + [t for t in present if t not in TIER_ORDER]

    results = {}
    for tier in order:
        mask = tiers == tier
        yt, yp = y_val[mask], y_pred_proba[mask]
        n, n_ch = int(mask.sum()), int(yt.sum())
        auc = roc_auc_score(yt, yp) if 0 < n_ch < n else None
        rec = (yp >= cutoff)[yt == 1].mean() if n_ch > 0 else None
        results[str(tier)] = {
            "n": n, "n_churners": n_ch,
            "auc": None if auc is None else round(float(auc), 4),
            "recall_at_10": None if rec is None else round(float(rec), 4),
            "flag_low_recall": bool(rec is not None and rec < aggregate_recall - 0.10),
            "too_small": bool(n_ch < 30),
        }
        print(f"  {tier:9s} n={n:5d} churners={n_ch:4d} "
              f"AUC={'n/a' if auc is None else f'{auc:.3f}'} "
              f"R@10={'n/a' if rec is None else f'{rec:.3f}'}")
    return {"aggregate_recall_at_10": round(float(aggregate_recall), 4),
            "global_cutoff": round(float(cutoff), 6), "tiers": results}


# ── Lab 4 entry point ──────────────────────────────────────────────────────────

def train(df: pd.DataFrame, seed: int = 42):
    """Train the churn model on `df` and return (booster, metrics).

    Same steps, split and hyperparameters as the Lab 3 script's main().
    Rows arrive sorted by customer_id (train_sagemaker.read_channel), so the
    split is reproducible.
    """
    df = df.dropna(subset=[LABEL]).reset_index(drop=True)
    X, y = preprocess_features(df)
    tier = df[SLICE_COLUMN].to_numpy(dtype=object)

    X_train, X_val, y_train, y_val, tier_train, tier_val = train_test_split(
        X, y, tier, test_size=0.30, random_state=seed, stratify=y)
    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,}")

    model = train_model(X_train, y_train, X_val, y_val, seed=seed)
    baseline_proba = train_recency_baseline(X_train, y_train, X_val)
    metrics = evaluate_model(model, X_val, y_val, baseline_proba)

    # Fairness: per-tier slice results ride along in the metrics so
    # tests/test_fairness.py (and the approver) can read them from
    # evaluation_metrics.json. Flagged, never blocking.
    slices = evaluate_slices(model, X_val, y_val, tier_val)
    t = slices["tiers"]
    recalls = {k: v["recall_at_10"] for k, v in t.items() if v["recall_at_10"] is not None}
    gap = None
    if "Platinum" in recalls and "Bronze" in recalls:
        gap = round(abs(recalls["Platinum"] - recalls["Bronze"]), 4)
    metrics.update({
        "slice_metrics": slices,
        "recall_top10_by_tier": recalls,
        "tier_recall_gap_platinum_bronze": gap,
        "tier_recall_gap_max_min": round(max(recalls.values()) - min(recalls.values()), 4)
                                   if recalls else None,
        "fairness_flag": bool(gap is not None and gap > FAIRNESS_GAP_PP),
        "hyperparameters": dict(HYPERPARAMS),
    })
    return model, metrics
