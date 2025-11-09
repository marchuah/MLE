# ============================================================
# MODEL MONITORING SCRIPT — for Airflow DAG
# ============================================================

import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import glob

# === Directories ===
MODEL_DIR = "/opt/airflow/utils/model_bank"
DATA_DIR  = "/opt/airflow/datamart/gold/feature_store"
REPORT_DIR = "/opt/airflow/datamart/gold/monitor_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# === Load reference (training) data ===
train_path = os.path.join(DATA_DIR, "training_snapshot.parquet")  # saved during training
oot_pattern = os.path.join(DATA_DIR, "gold_feature_store_2024_*.parquet")

# pick the most recent OOT data
oot_files = sorted(glob.glob(oot_pattern))
if not oot_files:
    raise FileNotFoundError("No OOT files found for monitoring.")
latest_oot = oot_files[-1]

print(f"Monitoring latest OOT snapshot: {latest_oot}")
df_train = pd.read_parquet(train_path)
df_oot   = pd.read_parquet(latest_oot)

# === Feature alignment ===
feature_cols = [c for c in df_train.columns if c.startswith("f_") or c not in ["label", "Customer_ID"]]
df_train = df_train[feature_cols].fillna(0)
df_oot   = df_oot[feature_cols].fillna(0)

# ============================================================
# METRIC 1 — Population Stability Index
# ============================================================

def calculate_psi(expected, actual, buckets=10):
    def scale_range(series):
        quantiles = np.percentile(series, np.linspace(0, 100, buckets + 1))
        return np.clip(np.digitize(series, quantiles, right=True), 1, buckets)

    expected_counts = np.bincount(scale_range(expected), minlength=buckets)
    actual_counts   = np.bincount(scale_range(actual), minlength=buckets)
    expected_perc = expected_counts / len(expected)
    actual_perc   = actual_counts / len(actual)
    psi = np.sum((actual_perc - expected_perc) * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6)))
    return psi

psi_results = {col: calculate_psi(df_train[col], df_oot[col]) for col in feature_cols}
psi_df = pd.DataFrame(list(psi_results.items()), columns=["feature", "psi_value"])
psi_df["drift_flag"] = psi_df["psi_value"] > 0.25

# ============================================================
# METRIC 2 — Kolmogorov–Smirnov Test
# ============================================================

ks_results = []
for col in feature_cols:
    ks_stat, ks_pval = ks_2samp(df_train[col], df_oot[col])
    ks_results.append((col, ks_stat, ks_pval))
ks_df = pd.DataFrame(ks_results, columns=["feature", "ks_stat", "ks_pval"])
ks_df["drift_flag"] = ks_df["ks_pval"] < 0.05

# ============================================================
# METRIC 3 — Monotonicity Violation Rate
# ============================================================

if "predicted_rul" in df_oot.columns:
    pred = df_oot.sort_values(["Customer_ID", "snapshot_date"])
    pred["rul_diff"] = pred.groupby("Customer_ID")["predicted_rul"].diff()
    mvr = (pred["rul_diff"] > 0).mean()
else:
    mvr = np.nan

# ============================================================
# REPORT SUMMARY
# ============================================================

summary = {
    "num_features": len(feature_cols),
    "num_drifted_features": psi_df["drift_flag"].sum(),
    "avg_psi": psi_df["psi_value"].mean(),
    "avg_ks": ks_df["ks_stat"].mean(),
    "mvr": mvr
}

summary_df = pd.DataFrame([summary])
report_path = os.path.join(REPORT_DIR, f"model_monitor_report_latest.csv")
summary_df.to_csv(report_path, index=False)
psi_df.to_csv(os.path.join(REPORT_DIR, "psi_detail.csv"), index=False)
ks_df.to_csv(os.path.join(REPORT_DIR, "ks_detail.csv"), index=False)

print("✅ Model monitoring completed.")
print(summary_df)
print(f"📊 Reports saved to {REPORT_DIR}")
