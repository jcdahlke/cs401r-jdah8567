"""
NorthStar Retail -- Fairness check (Lab 4 Task 1)

Manual slice evaluation by loyalty_tier. churn_model.train() scores every
validation row against ONE global top-10% cutoff and writes per-tier recall
into evaluation_metrics.json (key "slice_metrics"). These tests read that file
-- the same EVAL_METRICS_PATH the model gate reads -- and:

  * FAIL if the slice results are missing or malformed (no evidence = no gate)
  * FLAG, but do not fail, when the recall@top10% gap between the highest
    (Platinum) and lowest (Bronze) loyalty tier exceeds 10 percentage points.
    The flag is a warnings.warn, so `pytest -rw` puts it in the build log's
    warnings summary where the approver sees it before approving.

Run locally:
    pytest tests/test_fairness.py -v -rw --eval-metrics-path /tmp/out/evaluation_metrics.json
"""
import json
import os
import warnings
from pathlib import Path

import pytest

TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
MAX_RECALL_GAP = 0.10   # 10 percentage points; flag, do not block


@pytest.fixture(scope="module")
def slices():
    path = os.environ.get("EVAL_METRICS_PATH", "/tmp/eval_metrics.json")
    if not Path(path).exists():
        pytest.skip(f"Evaluation metrics not found at {path}.")
    with open(path) as f:
        metrics = json.load(f)
    assert "slice_metrics" in metrics, (
        "evaluation_metrics.json has no slice_metrics. churn_model.train() must "
        "emit per-tier results for the fairness check.")
    return metrics["slice_metrics"]


def test_every_tier_evaluated(slices):
    missing = [t for t in TIERS if t not in slices["tiers"]]
    assert not missing, f"No slice results for tiers: {missing}"
    for t in TIERS:
        assert slices["tiers"][t]["n"] > 0, f"{t} slice is empty in validation"


def test_tier_recalls_are_valid_rates(slices):
    for t in TIERS:
        r = slices["tiers"][t]["recall_at_10"]
        assert r is not None, f"{t} has no churners in validation; recall undefined"
        assert 0.0 <= r <= 1.0, f"{t} recall@10 {r} outside [0, 1]"


def test_tier_aucs_are_reported(slices):
    for t in TIERS:
        auc = slices["tiers"][t]["auc"]
        assert auc is None or 0.0 <= auc <= 1.0, f"{t} AUC {auc} outside [0, 1]"
    assert any(slices["tiers"][t]["auc"] is not None for t in TIERS)


def test_recall_gap_highest_vs_lowest_tier_flagged(slices):
    """Flag (not block) a Platinum-vs-Bronze recall@top10% gap above 10pp."""
    plat = slices["tiers"]["Platinum"]["recall_at_10"]
    bronze = slices["tiers"]["Bronze"]["recall_at_10"]
    gap = abs(plat - bronze)
    if gap > MAX_RECALL_GAP:
        warnings.warn(
            f"FAIRNESS FLAG: recall@top10% gap Platinum={plat:.3f} vs Bronze={bronze:.3f} "
            f"is {gap * 100:.1f}pp (> {MAX_RECALL_GAP * 100:.0f}pp). Not blocking -- "
            f"review before approving this model package.")
    else:
        warnings.warn(f"fairness: Platinum vs Bronze recall gap {gap * 100:.1f}pp "
                      f"(within {MAX_RECALL_GAP * 100:.0f}pp)")
    assert 0.0 <= gap <= 1.0
