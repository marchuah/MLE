# ============================================================
# GOLD FEATURE STORE CREATION — DD/MM/YYYY SAFE VERSION
# ============================================================

import os
import glob
import pandas as pd
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, coalesce, lit, countDistinct
from pyspark.sql.types import DateType
import shutil


# ============================================================
# MAIN DRIVER
# ============================================================

def process_gold_monthly_with_forward_labels(silver_directories, gold_feature_store_directory, spark, dpd_threshold=30):
    """
    Create monthly Gold feature store files with forward-looking labels.
    Now date-safe for DD/MM/YYYY pipeline.
    """

    print("=" * 80)
    print("🚀 PROCESSING MONTHLY GOLD FEATURE STORE WITH FORWARD-LOOKING LABELS")
    print("=" * 80)

    os.makedirs(gold_feature_store_directory, exist_ok=True)

    # ------------------------------------------------------------
    # Step 1: Load all LMS data
    # ------------------------------------------------------------
    lms_pattern = os.path.join(silver_directories["lms"], "silver_lms_*.parquet")
    lms_files = glob.glob(lms_pattern)

    if not lms_files:
        raise FileNotFoundError(f"No LMS parquet files found matching pattern: {lms_pattern}")

    print(f"📁 Found {len(lms_files)} LMS files in {silver_directories['lms']}")

    df_all_lms = spark.read.option("mergeSchema", "true").parquet(*lms_files)
    df_all_lms = standardize_date_column(df_all_lms, "snapshot_date")

    print(f"✅ Loaded LMS data: {df_all_lms.count()} rows")

    horizon = df_all_lms.agg(
        F.min("snapshot_date").alias("min_date"),
        F.max("snapshot_date").alias("max_date")
    ).collect()[0]
    print(f"📅 Data horizon: {horizon['min_date']} → {horizon['max_date']}")

    # ------------------------------------------------------------
    # Step 2: Create customer-level default lookup
    # ------------------------------------------------------------
    print("\nStep 2️⃣: Creating customer-level default lookup...")
    customer_defaults = create_customer_default_lookup(df_all_lms, dpd_threshold)
    print(f"✅ Created default lookup: {customer_defaults.count()} customers")

    # ------------------------------------------------------------
    # Step 3: Extract snapshot dates from file names
    # ------------------------------------------------------------
    snapshot_dates = []
    for f in lms_files:
        try:
            name = os.path.basename(f).replace("silver_lms_", "").replace(".parquet", "")
            date_obj = datetime.strptime(name, "%Y_%m_%d")
            snapshot_dates.append(date_obj)
        except Exception as e:
            print(f"⚠️ Could not parse date from {f}: {e}")

    snapshot_dates = sorted(list(set(snapshot_dates)))
    print(f"🗓 Found {len(snapshot_dates)} snapshots: {snapshot_dates[0]} → {snapshot_dates[-1]}")

    # ------------------------------------------------------------
    # Step 4: Process each month
    # ------------------------------------------------------------
    monthly_stats = []
    for snapshot_date in snapshot_dates:
        date_str = snapshot_date.strftime("%Y-%m-%d")
        print(f"\n{'=' * 70}\nProcessing snapshot: {date_str}\n{'=' * 70}")

        try:
            monthly_df = process_single_month_gold(
                snapshot_date_str=date_str,
                silver_directories=silver_directories,
                customer_defaults=customer_defaults,
                gold_feature_store_directory=gold_feature_store_directory,
                spark=spark,
                dpd_threshold=dpd_threshold
            )

            if monthly_df is not None:
                stats = {
                    "snapshot_date": date_str,
                    "rows": monthly_df.count(),
                    "label_1": monthly_df.filter(col("label") == 1).count(),
                    "label_0": monthly_df.filter(col("label") == 0).count(),
                }
                monthly_stats.append(stats)
                print(f"✅ {date_str}: {stats['rows']} rows | {stats['label_1']} defaults")
            else:
                print(f"⚠️ Skipped {date_str} — no valid data for this snapshot")

        except Exception as e:
            print(f"❌ Error processing {date_str}: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🏁 GOLD LAYER PROCESSING COMPLETE — SUMMARY")
    print("=" * 80)
    if monthly_stats:
        df_stats = pd.DataFrame(monthly_stats)
        print(df_stats.to_string(index=False))
        total_apps = df_stats["rows"].sum()
        total_defaults = df_stats["label_1"].sum()
        print(f"\n📊 Total applications: {total_apps}")
        print(f"📉 Total defaults: {total_defaults} ({(total_defaults / total_apps) * 100:.2f}%)")
    else:
        print("⚠️ No monthly outputs produced — check Silver layer or date parsing.")
        

# ============================================================
# UTILITY — Standardize Snapshot Date Column
# ============================================================

def standardize_date_column(df, col_name="snapshot_date"):
    """Ensures all snapshot_date fields are in proper ISO yyyy-MM-dd."""
    if col_name not in df.columns:
        return df
    return df.withColumn(
        col_name,
        F.coalesce(
            F.to_date(col(col_name), "yyyy-MM-dd"),
            F.to_date(col(col_name), "d/M/yyyy"),
            F.to_date(col(col_name), "M/d/yyyy")
        )
    )


# ============================================================
# FIXED LABEL LOGIC
# ============================================================

def create_customer_default_lookup(df_all_lms, dpd_threshold):
    print(f"🧮 Building customer default lookup (DPD ≥ {dpd_threshold})")

    horizon = df_all_lms.agg(F.max("snapshot_date").alias("max_date")).collect()[0]["max_date"]
    print(f"📅 Dataset horizon ends at: {horizon}")

    first_app = (
        df_all_lms.filter(F.col("mob") == 0)
        .groupBy("Customer_ID")
        .agg(F.min("snapshot_date").alias("first_snapshot_date"))
    )

    joined = df_all_lms.join(first_app, "Customer_ID", "inner").filter(F.col("snapshot_date") >= F.col("first_snapshot_date"))

    cust_defaults = (
        joined.groupBy("Customer_ID")
        .agg(
            F.max(F.when(F.col("dpd") >= dpd_threshold, 1).otherwise(0)).alias("label"),
            F.max("dpd").alias("max_dpd_ever"),
            F.max("mob").alias("max_mob_observed"),
            F.max("snapshot_date").alias("last_seen_date")
        )
        .withColumn("has_future_data", F.when(F.col("last_seen_date") < F.lit(horizon), 1).otherwise(0))
    )

    print("\nLabel distribution:")
    cust_defaults.groupBy("label").count().orderBy("label").show()
    return cust_defaults


# ============================================================
# SINGLE-MONTH GOLD CREATION
# ============================================================

def process_single_month_gold(snapshot_date_str, silver_directories, customer_defaults,
                              gold_feature_store_directory, spark, dpd_threshold):

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_suffix = snapshot_date_str.replace("-", "_")

    silver_dfs = {}
    for ds in ["lms", "financials", "attributes", "clickstream"]:
        fname = f"silver_{ds}_{date_suffix}.parquet"
        path = os.path.join(silver_directories[ds], fname)
        if not os.path.exists(path):
            print(f"⚠️ Missing Silver dataset: {path}")
            continue
        try:
            df = spark.read.parquet(path)
            df = standardize_date_column(df, "snapshot_date")
            silver_dfs[ds] = df
            horizon = df.agg(F.min("snapshot_date").alias("min_date"), F.max("snapshot_date").alias("max_date")).collect()[0]
            print(f"📂 Loaded {ds}: {df.count()} rows | Dates: {horizon['min_date']} → {horizon['max_date']}")
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")

    if "lms" not in silver_dfs:
        print("⚠️ No LMS data, skipping this month.")
        return None

    base_df = silver_dfs["lms"].filter(col("mob") == 0)
    if base_df.count() == 0:
        print(f"⚠️ No MOB=0 applications for {snapshot_date_str}")
        return None

    customers_df = (
        base_df.groupBy("Customer_ID", "snapshot_date")
        .agg(
            countDistinct("loan_id").alias("num_applications_this_month"),
            F.max("loan_amt").alias("application_loan_amount"),
            F.max("tenure").alias("application_tenure"),
        )
    )

    feature_df = customers_df

    if "financials" in silver_dfs:
        feature_df = feature_df.join(create_financial_features(silver_dfs["financials"]), "Customer_ID", "left")
    if "attributes" in silver_dfs:
        feature_df = feature_df.join(create_attribute_features(silver_dfs["attributes"]), "Customer_ID", "left")
    if "clickstream" in silver_dfs:
        feature_df = feature_df.join(create_clickstream_features(silver_dfs["clickstream"]), "Customer_ID", "left")
    if "lms" in silver_dfs:
        feature_df = feature_df.join(create_lms_features_mob0(silver_dfs["lms"]), "Customer_ID", "left")

    feature_df = (
        feature_df.join(
            customer_defaults.select("Customer_ID", "label", "max_dpd_ever", "max_mob_observed", "has_future_data"),
            "Customer_ID", "left"
        )
        .withColumn("label", coalesce(col("label"), lit(0)))
    )

    feature_df = (
        feature_df.withColumn("gold_processing_timestamp", F.current_timestamp())
        .withColumn("application_date", col("snapshot_date"))
        .withColumn("feature_store_version", lit("2.3"))
        .withColumn("dpd_threshold", lit(dpd_threshold))
    )

    outpath = os.path.join(gold_feature_store_directory, f"gold_feature_store_{date_suffix}.parquet")
    if os.path.exists(outpath):
        print(f"🧹 Removing existing Gold output before writing: {outpath}")
        shutil.rmtree(outpath)

    feature_df.write.mode("overwrite").parquet(outpath)
    print(f"✅ Saved Gold store for {snapshot_date_str} → {outpath}")
    return feature_df


# ============================================================
# FEATURE BUILDERS
# ============================================================

def create_lms_features_mob0(lms_df):
    from pyspark.sql.functions import count, mean as spark_mean, max as spark_max, when
    return (
        lms_df.filter(col("mob") == 0)
        .groupBy("Customer_ID")
        .agg(
            count("loan_id").alias("num_active_loans_at_application"),
            spark_mean("loan_amt").alias("avg_requested_loan_amount"),
            spark_max("loan_amt").alias("max_requested_loan_amount"),
            spark_mean("tenure").alias("avg_requested_tenure"),
            spark_max("tenure").alias("max_requested_tenure"),
        )
        .withColumn("multiple_loans_flag", when(col("num_active_loans_at_application") > 1, 1).otherwise(0))
    )


def create_financial_features(df):
    from pyspark.sql.functions import when
    df = df.withColumn("debt_to_income_ratio", col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
    df = df.withColumn("high_leverage_flag", when(col("debt_to_income_ratio") > 0.7, 1).otherwise(0))
    return df.select("Customer_ID", "debt_to_income_ratio", "high_leverage_flag")


def create_attribute_features(df):
    from pyspark.sql.functions import when
    df = df.withColumn(
        "financial_maturity_score",
        when(col("Age") < 25, 2).when(col("Age") <= 55, 5).otherwise(3),
    )
    return df.select("Customer_ID", "financial_maturity_score")


def create_clickstream_features(df):
    from pyspark.sql.functions import when
    df = df.withColumn(
        "engagement_score",
        when(col("total_positive_features") >= 100, 5)
        .when(col("total_positive_features") >= 50, 3)
        .otherwise(1),
    )
    return df.select("Customer_ID", "engagement_score")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--silver_lms_dir", required=True)
    parser.add_argument("--silver_financials_dir", required=True)
    parser.add_argument("--silver_attributes_dir", required=True)
    parser.add_argument("--silver_clickstream_dir", required=True)
    parser.add_argument("--gold_features_dir", required=True)
    parser.add_argument("--dpd_threshold", type=int, default=30)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("GoldMonthlyProcessing").getOrCreate()

    silver_dirs = {
        "lms": args.silver_lms_dir,
        "financials": args.silver_financials_dir,
        "attributes": args.silver_attributes_dir,
        "clickstream": args.silver_clickstream_dir,
    }

    process_gold_monthly_with_forward_labels(
        silver_directories=silver_dirs,
        gold_feature_store_directory=args.gold_features_dir,
        spark=spark,
        dpd_threshold=args.dpd_threshold,
    )

    spark.stop()
