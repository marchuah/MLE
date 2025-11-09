# ============================================================
# MODEL INFERENCE SCRIPT
# ============================================================

import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_BANK_DIR = "/opt/airflow/model_bank"
GOLD_DIR = "/opt/airflow/datamart/gold/feature_store"
OUTPUT_DIR = "/opt/airflow/model_inference"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allow Airflow templating argument for snapshot date
snapshot_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
print(f"🟢 Running inference for snapshot date: {snapshot_date}")

# ------------------------------------------------------------
# INIT SPARK
# ------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("ModelInference")
    .getOrCreate()
)

# ------------------------------------------------------------
# FIND LATEST MODEL IN MODEL BANK
# ------------------------------------------------------------
model_dirs = sorted(
    [os.path.join(MODEL_BANK_DIR, d) for d in os.listdir(MODEL_BANK_DIR) if d.startswith("logistic_model_clean_")],
    reverse=True
)
if not model_dirs:
    raise FileNotFoundError("❌ No trained model found in model_bank/")
latest_model_path = model_dirs[0]
print(f"✅ Using latest model: {latest_model_path}")

# Load trained pipeline
model = PipelineModel.load(latest_model_path)

# ------------------------------------------------------------
# LOAD LATEST GOLD FEATURE STORE FOR INFERENCE
# ------------------------------------------------------------
# Find parquet file closest to the snapshot date (for that Airflow run)
pattern = os.path.join(GOLD_DIR, f"gold_feature_store_{snapshot_date.replace('-', '_')}.parquet")

if not os.path.exists(pattern):
    # fallback: use most recent available file
    all_gold = sorted(
        [f for f in os.listdir(GOLD_DIR) if f.startswith("gold_feature_store_")],
        reverse=True
    )
    if not all_gold:
        raise FileNotFoundError("❌ No gold feature store files found.")
    latest_gold_file = all_gold[0]
    gold_path = os.path.join(GOLD_DIR, latest_gold_file)
    print(f"⚠️ No gold file found for {snapshot_date}, using most recent: {latest_gold_file}")
else:
    gold_path = pattern

df = spark.read.parquet(gold_path)
print(f"✅ Loaded {df.count()} rows from {gold_path}")

# ------------------------------------------------------------
# RUN INFERENCE
# ------------------------------------------------------------
pred_df = model.transform(df)

# Select outputs of interest
output_cols = ["Customer_ID", "application_date", "probability", "prediction"]
pred_df = pred_df.select(*output_cols)

# Save output to parquet
output_path = os.path.join(OUTPUT_DIR, f"inference_output_{snapshot_date.replace('-', '_')}.parquet")
pred_df.write.mode("overwrite").parquet(output_path)

print(f"✅ Inference complete — saved predictions to {output_path}")
spark.stop()
