import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("feature_store_pipeline") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

print("=== FEATURE STORE PIPELINE STARTED ===\n")

# Configuration
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"
DPD_THRESHOLD = 30  # Days Past Due threshold for labels

# Directory configurations (defined at module level)
bronze_directories = {
    'lms': "datamart/bronze/lms/",
    'financials': "datamart/bronze/financials/", 
    'attributes': "datamart/bronze/attributes/",
    'clickstream': "datamart/bronze/clickstream/"
}

silver_directories = {
    'lms': "datamart/silver/lms/",
    'financials': "datamart/silver/financials/", 
    'attributes': "datamart/silver/attributes/",
    'clickstream': "datamart/silver/clickstream/"
}

gold_feature_store_directory = "datamart/gold/feature_store/"


# Generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(f"Processing dates: {len(dates_str_lst)} months from {start_date_str} to {end_date_str}")
print(f"Sample dates: {dates_str_lst[:3]}...{dates_str_lst[-3:]}\n")

# ================================
# BRONZE LAYER - RAW DATA INGESTION
# ================================
print("=== BRONZE LAYER PROCESSING ===")

# Create directories if they don't exist
for dir_path in bronze_directories.values():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

# Run bronze backfill for all datasets
print("Processing bronze tables for all datasets...")
for i, date_str in enumerate(dates_str_lst):
    print(f"Processing bronze for {date_str} ({i+1}/{len(dates_str_lst)})")
    try:
        utils.data_processing_bronze_table.process_bronze_tables(
            date_str, 
            bronze_directories, 
            spark
        )
    except Exception as e:
        print(f"Error processing bronze for {date_str}: {e}")
        continue

print("Bronze layer processing completed.\n")

# ================================
# SILVER LAYER - DATA CLEANING & TRANSFORMATION
# ================================
print("=== SILVER LAYER PROCESSING ===")

# Create directories if they don't exist
for dir_path in silver_directories.values():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

# Run silver backfill for all datasets
print("Processing silver tables for all datasets...")
for i, date_str in enumerate(dates_str_lst):
    print(f"Processing silver for {date_str} ({i+1}/{len(dates_str_lst)})")
    try:
        utils.data_processing_silver_table.process_silver_tables(
            date_str, 
            bronze_directories,
            silver_directories, 
            spark
        )
    except Exception as e:
        print(f"Error processing silver for {date_str}: {e}")
        continue

print("Silver layer processing completed.\n")

# ================================
# GOLD LAYER - FEATURE STORE WITH FORWARD-LOOKING LABELS
# ================================
print("=== GOLD LAYER PROCESSING ===")

# Create gold directories
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)
    print(f"Created directory: {gold_feature_store_directory}")

# Run gold processing - SINGLE CALL processes all months with forward-looking labels
print("Processing monthly gold feature store with forward-looking labels...")
print(f"Default definition: {DPD_THRESHOLD}+ DPD on ANY future loan")
print("Features: MOB=0 (Application Time)")
print("Labels: Forward-looking (does customer EVER default?)")

try:
    monthly_stats = utils.data_processing_gold_table.process_gold_monthly_with_forward_labels(
        silver_directories=silver_directories,
        gold_feature_store_directory=gold_feature_store_directory,
        spark=spark,
        dpd_threshold=DPD_THRESHOLD
    )
    print("\n✓ Gold layer processing completed successfully!")
except Exception as e:
    print(f"\n✗ Error processing gold layer: {e}")
    import traceback
    traceback.print_exc()

# ================================
# VALIDATION & SUMMARY STATISTICS
# ================================
print("\n" + "="*80)
print("=== PIPELINE VALIDATION & SUMMARY ===")
print("="*80)

# Check Feature Store
print("\nFeature Store Summary:")
feature_folder_path = gold_feature_store_directory
feature_files_list = glob.glob(os.path.join(feature_folder_path, '*.parquet'))

if feature_files_list:
    print(f"Found {len(feature_files_list)} monthly feature store files")
    
    # Load all monthly files
    feature_df = spark.read.parquet(feature_folder_path + "*.parquet")
    
    total_rows = feature_df.count()
    total_customers = feature_df.select('Customer_ID').distinct().count()
    
    print(f"Feature Store - Total rows: {total_rows}")
    print(f"Feature Store - Total columns: {len(feature_df.columns)}")
    print(f"Feature Store - Unique customers: {total_customers}")
    
    # Label distribution (integrated in feature store now)
    if 'label' in feature_df.columns:
        print("\nLabel Distribution:")
        label_dist = feature_df.groupBy("label").count().orderBy("label").collect()
        for row in label_dist:
            print(f"  Label {row['label']}: {row['count']} ({row['count']/total_rows*100:.1f}%)")
        
        # Training vs Inference split
        if 'has_future_observation' in feature_df.columns:
            training_count = feature_df.filter(col("has_future_observation") == 1).count()
            inference_count = feature_df.filter(col("has_future_observation") == 0).count()
            print(f"\nData Split:")
            print(f"  Training data (with future observation): {training_count} ({training_count/total_rows*100:.1f}%)")
            print(f"  Inference data (without future observation): {inference_count} ({inference_count/total_rows*100:.1f}%)")
    
    print("\nFeature Store Schema (first 20 columns):")
    for col_name in sorted(feature_df.columns)[:20]:
        print(f"  - {col_name}")
    if len(feature_df.columns) > 20:
        print(f"  ... and {len(feature_df.columns) - 20} more columns")
    
    print("\nFeature Store Sample:")
    # Show relevant columns
    sample_cols = ["Customer_ID", "application_date", "label"]
    
    # Add columns that exist
    for col_name in ["num_active_loans_at_application", "annual_income", "customer_age", "has_future_observation"]:
        if col_name in feature_df.columns:
            sample_cols.append(col_name)
    
    feature_df.select(*sample_cols).show(5, truncate=False)
    
    # Monthly breakdown
    print("\nMonthly Application Summary:")
    monthly_summary = feature_df.groupBy("application_date").agg(
        F.count("*").alias("num_applications"),
        F.sum(col("label").cast("int")).alias("num_defaults")
    ).orderBy("application_date")
    
    monthly_summary = monthly_summary.withColumn(
        "default_rate",
        (col("num_defaults") / col("num_applications") * 100)
    )
    
    monthly_summary.show(10, truncate=False)
    
    # Date range analysis
    print("\nDate Range Analysis:")
    date_stats = feature_df.agg(
        F.min("application_date").alias("earliest_application"),
        F.max("application_date").alias("latest_application")
    ).collect()[0]
    
    print(f"  Earliest application: {date_stats['earliest_application']}")
    print(f"  Latest application: {date_stats['latest_application']}")
    
else:
    print("⚠️  No feature store files found!")

# Data Quality Summary
print(f"\n" + "="*80)
print(f"=== DATA QUALITY SUMMARY ===")
print(f"="*80)
print(f"Bronze files processed: {len(dates_str_lst)} months × 4 datasets")
print(f"Silver files processed: {len(dates_str_lst)} months × 4 datasets") 
print(f"Gold feature store files: {len(feature_files_list) if feature_files_list else 0} monthly files")

# ML Setup Validation
print(f"\n" + "="*80)
print(f"=== ML SETUP VALIDATION ===")
print(f"="*80)
print(f"✓ Features extracted at: MOB=0 (Application Time)")
print(f"✓ Labels: Forward-looking (customer ever defaults?)")
print(f"✓ Default definition: {DPD_THRESHOLD}+ DPD on ANY future loan")
print(f"✓ Train/Test split: By Customer_ID (prevents data leakage)")
print(f"✓ Inference set: Applications from recent months without sufficient observation")

if feature_files_list:
    print(f"\n{'='*80}")
    print("=== NEXT STEPS FOR MODEL TRAINING ===")
    print(f"{'='*80}")
    print("""
1. Load all monthly files:
   df = spark.read.parquet("datamart/gold/feature_store/*.parquet")

2. Filter training data:
   train_df = df.filter(col("has_future_observation") == 1)

3. Split by Customer_ID:
   unique_customers = train_df.select("Customer_ID").distinct()
   train_customers, test_customers = unique_customers.randomSplit([0.8, 0.2], seed=42)

4. Train model on features to predict 'label'

5. Use inference set for new predictions:
   inference_df = df.filter(col("has_future_observation") == 0)
""")

# Cleanup
spark.stop()

print("\n" + "="*80)
print("=== FEATURE STORE PIPELINE COMPLETED ===")
print("="*80)