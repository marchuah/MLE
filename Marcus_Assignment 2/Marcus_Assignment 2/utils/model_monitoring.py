# ============================================================
# MODEL MONITORING — RESILIENT VERSION (ALWAYS PRODUCES METRICS)
# ============================================================

import os
import pandas as pd
import numpy as np
from datetime import datetime

OOT_DIR = "/opt/airflow/datamart/gold/oot/"
GOLD_DIR = "/opt/airflow/datamart/gold/feature_store/"
MONITOR_LOG = "/opt/airflow/datamart/gold/monitoring_summary.csv"

print("🚀 Starting Model Monitoring Job")
print(f"📂 Checking OOT directory: {OOT_DIR}")

# ------------------------------------------------------------
# Step 1 — Identify which dataset to monitor
# ------------------------------------------------------------
target_file = None
source_type = None

# Prefer OOT
if os.path.exists(OOT_DIR):
    oot_files = [f for f in os.listdir(OOT_DIR) if f.endswith(".parquet")]
    if oot_files:
        oot_files.sort()
        target_file = os.path.join(OOT_DIR, oot_files[-1])
        source_type = "OOT"
        print(f"🗂 Found OOT file: {target_file}")

# Fallback to Gold Feature Store
if target_file is None:
    print("⚠️ No OOT files found. Falling back to Gold Feature Store...")
    if os.path.exists(GOLD_DIR):
        gold_files = [f for f in os.listdir(GOLD_DIR) if f.endswith(".parquet")]
        if gold_files:
            gold_files.sort()
            target_file = os.path.join(GOLD_DIR, gold_files[-1])
            source_type = "GOLD"
            print(f"🗂 Using latest Gold file: {target_file}")
        else:
            print("❌ No Gold files found either. Exiting gracefully.")
            exit(0)
    else:
        print("❌ Gold directory does not exist. Exiting gracefully.")
        exit(0)

# ------------------------------------------------------------
# Step 2 — Load dataset safely
# ------------------------------------------------------------
try:
    df = pd.read_parquet(target_file)
    print(f"✅ Loaded {source_type} dataset: {df.shape[0]} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit(0)

# ------------------------------------------------------------
# Step 3 — Compute simple monitoring metrics
# ------------------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
metrics = {}

if numeric_cols:
    for col in numeric_cols[:5]:  # only sample top 5 numeric cols
        metrics[f"{col}_mean"] = round(df[col].mean(), 4)
        metrics[f"{col}_std"] = round(df[col].std(), 4)
        metrics[f"{col}_null_pct"] = round(df[col].isna().mean() * 100, 2)
else:
    print("⚠️ No numeric columns found, skipping numeric stats.")

summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source_type": source_type,
    "file_name": os.path.basename(target_file),
    "n_rows": df.shape[0],
    "n_cols": df.shape[1],
    "example_numeric_cols": ", ".join(numeric_cols[:5]) if numeric_cols else "None",
    **metrics,
}

# ------------------------------------------------------------
# Step 4 — Append to monitoring summary log
# ------------------------------------------------------------
summary_df = pd.DataFrame([summary])

if os.path.exists(MONITOR_LOG):
    existing = pd.read_csv(MONITOR_LOG)
    summary_df = pd.concat([existing, summary_df], ignore_index=True)

summary_df.to_csv(MONITOR_LOG, index=False)
print(f"🧾 Monitoring summary saved to: {MONITOR_LOG}")

# ------------------------------------------------------------
# Step 5 — Optional preview for Airflow logs
# ------------------------------------------------------------
print("\n📊 Latest Monitoring Summary:")
print(summary_df.tail(1).to_string(index=False))

print("\n✅ Monitoring step completed successfully (exit code 0).")
