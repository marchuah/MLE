# ============================================================
# BRONZE LAYER CREATION — DD/MM/YYYY FIXED & OVERWRITE-SAFE
# ============================================================

import os
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from datetime import datetime
from pyspark.sql.types import DateType
import argparse
import shutil


def process_bronze_tables(snapshot_date_str, bronze_directories, spark):
    """
    Process raw CSVs into Bronze layer (standardized, timestamped, overwrite-safe).
    Handles DD/MM/YYYY date format correctly.
    """

    print("=" * 80)
    print(f"PROCESSING BRONZE TABLES for snapshot date: {snapshot_date_str}")
    print("=" * 80)

    # Try to parse input date safely
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        target_date_format = snapshot_date.strftime("%Y-%m-%d")
    except ValueError:
        snapshot_date = datetime.strptime(snapshot_date_str, "%d/%m/%Y")
        target_date_format = snapshot_date.strftime("%Y-%m-%d")

    # Dataset configs
    datasets_config = {
        'lms': {
            'file_path': "data/lms_loan_daily.csv",
            'directory': bronze_directories['lms'],
            'prefix': "bronze_loan_daily_",
            'date_column': 'snapshot_date'
        },
        'financials': {
            'file_path': "data/features_financials.csv",
            'directory': bronze_directories['financials'],
            'prefix': "bronze_financials_",
            'date_column': 'snapshot_date'
        },
        'attributes': {
            'file_path': "data/features_attributes.csv",
            'directory': bronze_directories['attributes'],
            'prefix': "bronze_attributes_",
            'date_column': 'snapshot_date'
        },
        'clickstream': {
            'file_path': "data/feature_clickstream.csv",
            'directory': bronze_directories['clickstream'],
            'prefix': "bronze_clickstream_",
            'date_column': 'snapshot_date'
        },
    }

    processed_dfs = {}

    for dataset, cfg in datasets_config.items():
        print(f"\n--- Processing {dataset.upper()} ---")

        try:
            os.makedirs(cfg['directory'], exist_ok=True)

            if not os.path.exists(cfg['file_path']):
                print(f"⚠️ Source file missing: {cfg['file_path']} — skipping.")
                continue

            df = spark.read.csv(cfg['file_path'], header=True, inferSchema=False)
            total_rows = df.count()
            print(f"✅ Loaded {dataset}: {total_rows} rows")

            # Normalize DD/MM/YYYY → YYYY-MM-DD
            if cfg['date_column'] in df.columns:
                df = df.withColumn(
                    "normalized_snapshot_date",
                    F.when(
                        F.col(cfg['date_column']).rlike(r'^\d{1,2}/\d{1,2}/\d{4}$'),
                        F.date_format(F.to_date(F.col(cfg['date_column']), "d/M/yyyy"), "yyyy-MM-dd")
                    ).otherwise(F.col(cfg['date_column']))
                )
                df_filtered = df.filter(F.col("normalized_snapshot_date") == target_date_format)
                filtered_count = df_filtered.count()
                print(f"📅 Filtered {dataset} for {target_date_format}: {filtered_count} rows")

                if filtered_count == 0:
                    available_dates = df.select("normalized_snapshot_date").distinct().filter(col("normalized_snapshot_date").isNotNull())
                    if available_dates.count() > 0:
                        most_recent_date = available_dates.agg(F.max("normalized_snapshot_date")).collect()[0][0]
                        print(f"➡️ No rows for {target_date_format}, using most recent: {most_recent_date}")
                        df_filtered = df.filter(F.col("normalized_snapshot_date") == most_recent_date)
                    else:
                        print(f"❌ No valid date found in {dataset}, skipping.")
                        continue
            else:
                df_filtered = df.withColumn("normalized_snapshot_date", F.lit(target_date_format))
                print(f"⚠️ No date column in {dataset}, all rows assigned to {target_date_format}")

            # Add metadata
            df_final = (
                df_filtered
                .withColumn("bronze_ingestion_timestamp", F.current_timestamp())
                .withColumn("bronze_source_file", F.lit(cfg['file_path']))
                .withColumn("bronze_processing_date", F.lit(target_date_format))
            )

            # Output path
            filename = f"{cfg['prefix']}{target_date_format.replace('-', '_')}.csv"
            outpath = os.path.join(cfg['directory'], filename)

            if os.path.exists(outpath):
                print(f"🧹 Removing existing file before overwrite: {outpath}")
                os.remove(outpath)

            df_final.toPandas().to_csv(outpath, index=False)
            print(f"✅ Saved Bronze {dataset} → {outpath} ({df_final.count()} rows)")

            processed_dfs[dataset] = df_final

        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ BRONZE LAYER COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return processed_dfs


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("BronzeLayerProcessing").getOrCreate()

    bronze_dirs = {
        "lms": "/opt/airflow/datamart/bronze/lms/",
        "financials": "/opt/airflow/datamart/bronze/financials/",
        "attributes": "/opt/airflow/datamart/bronze/attributes/",
        "clickstream": "/opt/airflow/datamart/bronze/clickstream/"
    }

    process_bronze_tables(args.snapshotdate, bronze_dirs, spark)
    spark.stop()
