# ============================================================
# MODEL INFERENCE SCRIPT — Random Forest (Robust + Vector Fix)
# ============================================================

import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_BANK_DIR = "/opt/airflow/model_bank"
GOLD_DIR = "/opt/airflow/datamart/gold/feature_store"
OUTPUT_DIR = "/opt/airflow/model_inference"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# SNAPSHOT DATE
# ------------------------------------------------------------
snapshot_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
print(f"🟢 Running inference for snapshot date: {snapshot_date}")

# ------------------------------------------------------------
# INIT SPARK
# ------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("ModelInference_RF")
    .getOrCreate()
)

# ------------------------------------------------------------
# FIND LATEST RANDOM FOREST MODEL
# ------------------------------------------------------------
model_dirs = sorted(
    [os.path.join(MODEL_BANK_DIR, d) for d in os.listdir(MODEL_BANK_DIR)
     if d.startswith("randomforest_clean_")],
    reverse=True
)

if not model_dirs:
    raise FileNotFoundError("❌ No Random Forest model found in model_bank/.")
latest_model_path = model_dirs[0]
print(f"✅ Using latest Random Forest model: {latest_model_path}")

model = PipelineModel.load(latest_model_path)

# ------------------------------------------------------------
# LOAD GOLD FEATURE STORE
# ------------------------------------------------------------
pattern = os.path.join(GOLD_DIR, f"gold_feature_store_{snapshot_date.replace('-', '_')}.parquet")

if not os.path.exists(pattern):
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
# ENSURE NUMERIC TYPES BEFORE MODEL TRANSFORM
# ------------------------------------------------------------
string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType)]
numeric_candidates = [c for c in string_cols if any(x in c.lower() for x in [
    "loan", "tenure", "amount", "balance", "dpd", "score", "ratio", "feature", "income"
])]

if numeric_candidates:
    print(f"🧮 Casting {len(numeric_candidates)} numeric-like string columns to double:")
    for c in numeric_candidates:
        print(f"   - {c}")
        df = df.withColumn(c, F.col(c).cast(T.DoubleType()))
else:
    print("✅ No numeric-like string columns found, skipping type casting.")

# ------------------------------------------------------------
# RUN INFERENCE
# ------------------------------------------------------------
try:
    pred_df = model.transform(df)
except Exception as e:
    print("❌ Model transform failed, printing schema for debugging:")
    df.printSchema()
    raise e

# ------------------------------------------------------------
# SELECT AND SAVE OUTPUT (with SparseVector → Array fix)
# ------------------------------------------------------------
expected_cols = ["Customer_ID", "application_date", "probability", "prediction"]
available_cols = [c for c in expected_cols if c in pred_df.columns]

if "probability" in pred_df.columns:
    # Convert SparseVector to array, extract probability for positive class (index 1)
    pred_df = pred_df.withColumn("pd_score", vector_to_array("probability")[1])
else:
    pred_df = pred_df.withColumn("pd_score", F.lit(None).cast(T.DoubleType()))

final_cols = [c for c in ["Customer_ID", "application_date", "pd_score", "prediction"] if c in pred_df.columns]
pred_df = pred_df.select(*final_cols)

# ------------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------------
output_path = os.path.join(OUTPUT_DIR, f"inference_output_{snapshot_date.replace('-', '_')}.parquet")
pred_df.write.mode("overwrite").parquet(output_path)

print(f"✅ Inference complete — saved predictions to {output_path}")
spark.stop()
