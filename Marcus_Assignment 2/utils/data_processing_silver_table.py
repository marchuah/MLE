# ============================================================
# SILVER LAYER CREATION — CLEANED & DD/MM/YYYY FIRST (FULL FEATURES)
# ============================================================

import os
import shutil
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, regexp_replace, trim
from pyspark.sql.types import IntegerType, DoubleType
from datetime import datetime
import argparse


# ============================================================
# MAIN PROCESS
# ============================================================

def process_silver_tables(snapshot_date_str, bronze_directories, silver_directories, spark):
    print("=" * 80)
    print(f"PROCESSING SILVER TABLES for snapshot date: {snapshot_date_str}")
    print("=" * 80)

    # Parse snapshot date with DD/MM/YYYY priority
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        date_suffix = snapshot_date.strftime("%Y_%m_%d")
    except ValueError:
        snapshot_date = datetime.strptime(snapshot_date_str, "%d/%m/%Y")
        date_suffix = snapshot_date.strftime("%Y_%m_%d")

    datasets = ["lms", "financials", "attributes", "clickstream"]
    processed_dfs = {}

    for dataset_name in datasets:
        print(f"\n--- Processing {dataset_name.upper()} SILVER TABLE ---")

        try:
            os.makedirs(silver_directories[dataset_name], exist_ok=True)

            bronze_file = {
                "lms": f"bronze_loan_daily_{date_suffix}.csv",
                "financials": f"bronze_financials_{date_suffix}.csv",
                "attributes": f"bronze_attributes_{date_suffix}.csv",
                "clickstream": f"bronze_clickstream_{date_suffix}.csv",
            }[dataset_name]

            bronze_path = os.path.join(bronze_directories[dataset_name], bronze_file)

            if not os.path.exists(bronze_path):
                print(f"⚠️ Missing Bronze file → {bronze_path}")
                continue

            df = spark.read.csv(bronze_path, header=True, inferSchema=False)
            print(f"✅ Loaded Bronze {dataset_name}: {df.count()} rows")

            # Dataset-specific cleaning
            if dataset_name == "lms":
                df_clean = process_lms_silver(df)
            elif dataset_name == "financials":
                df_clean = process_financials_silver(df)
            elif dataset_name == "attributes":
                df_clean = process_attributes_silver(df)
            elif dataset_name == "clickstream":
                df_clean = process_clickstream_silver(df)
            else:
                df_clean = df

            # Add metadata
            df_clean = (
                df_clean
                .withColumn("silver_processing_timestamp", F.current_timestamp())
                .withColumn("silver_processing_date", F.lit(snapshot_date_str))
            )

            # Output Parquet file
            silver_path = os.path.join(
                silver_directories[dataset_name],
                f"silver_{dataset_name}_{date_suffix}.parquet"
            )

            if os.path.exists(silver_path):
                print(f"🧹 Removing old Silver file before overwrite: {silver_path}")
                shutil.rmtree(silver_path)

            df_clean.write.mode("overwrite").parquet(silver_path)
            print(f"✅ Saved cleaned {dataset_name} → {silver_path}")

            processed_dfs[dataset_name] = df_clean

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ SILVER LAYER COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return processed_dfs


# ============================================================
# UTILITIES
# ============================================================

def parse_date_column(df, col_name):
    """Parse snapshot_date using DD/MM/YYYY first, then fallback to others."""
    return df.withColumn(
        col_name,
        F.coalesce(
            F.to_date(col(col_name), "d/M/yyyy"),
            F.to_date(col(col_name), "yyyy-MM-dd"),
            F.to_date(col(col_name), "M/d/yyyy"),
        ),
    )


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def process_lms_silver(df):
    print("🧭 Cleaning LMS loan data...")
    if "snapshot_date" in df.columns:
        df = parse_date_column(df, "snapshot_date")

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn(
        "dpd",
        when(col("overdue_amt") > 0,
             F.datediff(col("snapshot_date"), F.add_months(col("snapshot_date"), -1))
             ).otherwise(0)
    )
    return df


def process_financials_silver(df):
    print("💰 Cleaning Financials data...")
    if "snapshot_date" in df.columns:
        df = parse_date_column(df, "snapshot_date")
    return df


def process_attributes_silver(df):
    print("👤 Cleaning Attributes data...")
    if "snapshot_date" in df.columns:
        df = parse_date_column(df, "snapshot_date")
    return df


def process_clickstream_silver(df):
    print("🖱️ Cleaning Clickstream data and deriving engagement features...")
    if "snapshot_date" in df.columns:
        df = parse_date_column(df, "snapshot_date")

    # Cast feature columns
    feature_cols = [c for c in df.columns if c.startswith("fe_")]
    for c in feature_cols:
        df = df.withColumn(c, col(c).cast(DoubleType()))

    if feature_cols:
        df = df.withColumn(
            "total_positive_features",
            sum([when(col(c) > 0, col(c)).otherwise(0) for c in feature_cols])
        )
        df = df.withColumn(
            "total_negative_features",
            sum([when(col(c) < 0, col(c)).otherwise(0) for c in feature_cols])
        )
        df = df.withColumn(
            "feature_balance_ratio",
            when(col("total_positive_features") != 0,
                 col("total_negative_features") / col("total_positive_features"))
        )

    return df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("SilverLayerProcessing").getOrCreate()

    bronze_dirs = {
        "lms": "/opt/airflow/datamart/bronze/lms/",
        "financials": "/opt/airflow/datamart/bronze/financials/",
        "attributes": "/opt/airflow/datamart/bronze/attributes/",
        "clickstream": "/opt/airflow/datamart/bronze/clickstream/",
    }

    silver_dirs = {
        "lms": "/opt/airflow/datamart/silver/lms/",
        "financials": "/opt/airflow/datamart/silver/financials/",
        "attributes": "/opt/airflow/datamart/silver/attributes/",
        "clickstream": "/opt/airflow/datamart/silver/clickstream/",
    }

    process_silver_tables(args.snapshotdate, bronze_dirs, silver_dirs, spark)
    spark.stop()
