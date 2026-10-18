"""Run the provided RAGAS harness on all 4 test cases, several times, with and without
the customer profile in `contexts` (the before/after improvement for the rubric).

Run from the repo root:
    python rag/run_eval.py                 # 2 runs baseline + 2 runs improved
    python rag/run_eval.py --runs 3
    python rag/run_eval.py --only improved # re-run just one config

Writes:
    evaluation_results/rag_eval_<config>_run<N>.csv   one file per run
    evaluation_results/rag_eval_all.csv               every run, every case
    docs/lab3-evidence/rag_eval.csv                   final evidence (improved config)
    docs/lab3-evidence/rag_eval_baseline.csv          before-change evidence
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))

import evaluation_harness as eh
from offer_rag import answer_customer_rag

METRICS = ["faithfulness", "answer_relevancy", "context_recall"]
CONFIGS = {
    "baseline": lambda c, q: answer_customer_rag(c, q, include_profile=False),
    "improved": lambda c, q: answer_customer_rag(c, q, include_profile=True),
}

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=2)
parser.add_argument("--only", choices=list(CONFIGS), default=None)
args = parser.parse_args()

results_dir = ROOT / "evaluation_results"
evidence_dir = ROOT / "docs" / "lab3-evidence"
results_dir.mkdir(exist_ok=True)
evidence_dir.mkdir(parents=True, exist_ok=True)

judge = eh.DEFAULT_BEDROCK_MODEL_ID
frames = []
for config in ([args.only] if args.only else list(CONFIGS)):
    for run in range(1, args.runs + 1):
        print(f"\n===== config={config}  run={run}/{args.runs} =====")
        df = eh.RAGEvaluator(rag_invoke_fn=CONFIGS[config], bedrock_model_id=judge).run(eh.OFFER_TEST_CASES)
        df.insert(0, "case", [tc.customer_id for tc in eh.OFFER_TEST_CASES])
        df.insert(0, "run", run)
        df.insert(0, "config", config)
        df["judge_model"] = judge
        df.to_csv(results_dir / f"rag_eval_{config}_run{run}.csv", index=False)
        frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_path = results_dir / "rag_eval_all.csv"
if args.only and all_path.exists():  # keep the other config's earlier results
    old = pd.read_csv(all_path)
    all_df = pd.concat([old[old["config"] != args.only], all_df], ignore_index=True)
all_df.to_csv(all_path, index=False)

for config in all_df["config"].unique():
    sub = all_df[all_df["config"] == config]
    name = "rag_eval.csv" if config == "improved" else f"rag_eval_{config}.csv"
    sub.to_csv(evidence_dir / name, index=False)

# ---- summary you can paste into the writeup ----
print(f"\nJudge model: {judge}")
print("\nPer-case scores (each run):")
print(all_df[["config", "run", "case"] + METRICS].round(3).to_string(index=False))
per_run = all_df.groupby(["config", "run"])[METRICS].mean()
print("\nMean per run:")
print(per_run.round(3).to_string())
print("\nMean and range across runs:")
for config, g in per_run.groupby(level=0):
    parts = [f"{m}={g[m].mean():.3f} [{g[m].min():.3f}-{g[m].max():.3f}]" for m in METRICS]
    print(f"  {config:9s} " + "  ".join(parts))
nan = all_df[METRICS].isna().sum().sum()
if nan:
    print(f"\nWARNING: {nan} NaN score(s) = failed judge calls (usually throttling), not zeros.")
    print("Re-run that config with --only <config>; do not report NaN as a result.")
