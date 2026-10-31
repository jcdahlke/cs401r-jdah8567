"""Lab 4 Task 3 -- export MLflow App runs to docs/lab4-mlflow-runs.txt.

    pip install mlflow sagemaker-mlflow      # same as Lab 3, laptop only
    python scripts/lab4_mlflow_export.py <mlflow-app-arn> [experiment-name]

Writes the WHOLE table (no truncated params.* columns), as the handout asks.
"""
import sys

import mlflow
import pandas as pd

arn = sys.argv[1]
experiment = sys.argv[2] if len(sys.argv) > 2 else "northstar-churn"
mlflow.set_tracking_uri(arn)
runs = mlflow.search_runs(experiment_names=[experiment])
cols = ["run_id", "tags.mlflow.runName", "start_time"] + \
       [c for c in runs.columns if c.startswith(("params.", "metrics."))]
pd.set_option("display.width", 10000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
with open("docs/lab4-mlflow-runs.txt", "w") as f:
    f.write(f"tracking_uri: {mlflow.get_tracking_uri()}\n")
    f.write(f"experiment: {experiment}\n\n{runs[cols].to_string()}\n")
print(f"{len(runs)} runs written to docs/lab4-mlflow-runs.txt")
print(runs[["tags.mlflow.runName"] + [c for c in cols if c.startswith("params.")]].to_string())
