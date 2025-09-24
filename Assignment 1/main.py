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
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

print("=== FEATURE STORE PIPELINE STARTED ===\n")

# Configuration
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"
MOB_FILTER = 6  # Month on Book for feature store and labels
DPD_THRESHOLD = 30  # Days Past Due threshold for labels

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

# Create bronze directories for all datasets
bronze_directories = {
    'lms': "datamart/bronze/lms/",
    'financials': "datamart/bronze/financials/", 
    'attributes': "datamart/bronze/attributes/",
    'clickstream': "datamart/bronze/clickstream/"
}

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

# Create silver directories for all datasets
silver_directories = {
    'lms': "datamart/silver/lms/",
    'financials': "datamart/silver/financials/", 
    'attributes': "datamart/silver/attributes/",
    'clickstream': "datamart/silver/clickstream/"
}

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
# GOLD LAYER - FEATURE STORE & LABEL STORE
# ================================
print("=== GOLD LAYER PROCESSING ===")

# Create gold directories
gold_feature_store_directory = "datamart/gold/feature_store/"
gold_label_store_directory = "datamart/gold/label_store/"

for directory in [gold_feature_store_directory, gold_label_store_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Run gold backfill - Feature Store
print(f"Processing gold feature store (MOB filter: {MOB_FILTER})...")
for i, date_str in enumerate(dates_str_lst):
    print(f"Processing gold feature store for {date_str} ({i+1}/{len(dates_str_lst)})")
    try:
        utils.data_processing_gold_table.process_gold_feature_store(
            date_str, 
            silver_directories,
            gold_feature_store_directory, 
            spark,
            mob=MOB_FILTER
        )
    except Exception as e:
        print(f"Error processing gold feature store for {date_str}: {e}")
        continue

# Run gold backfill - Label Store  
print(f"Processing gold label store ({DPD_THRESHOLD}DPD at {MOB_FILTER}MOB)...")
for i, date_str in enumerate(dates_str_lst):
    print(f"Processing gold labels for {date_str} ({i+1}/{len(dates_str_lst)})")
    try:
        utils.data_processing_gold_table.process_labels_gold_table(
            date_str, 
            silver_directories['lms'],
            gold_label_store_directory, 
            spark, 
            dpd=DPD_THRESHOLD, 
            mob=MOB_FILTER
        )
    except Exception as e:
        print(f"Error processing gold labels for {date_str}: {e}")
        continue

print("Gold layer processing completed.\n")

# ================================
# VALIDATION & SUMMARY STATISTICS
# ================================
print("=== PIPELINE VALIDATION & SUMMARY ===")

# Check Feature Store
print("Feature Store Summary:")
feature_folder_path = gold_feature_store_directory
feature_files_list = [feature_folder_path + os.path.basename(f) for f in glob.glob(os.path.join(feature_folder_path, '*'))]

if feature_files_list:
    feature_df = spark.read.option("header", "true").parquet(*feature_files_list)
    print(f"Feature Store - Total rows: {feature_df.count()}")
    print(f"Feature Store - Total columns: {len(feature_df.columns)}")
    print(f"Feature Store - Unique customers: {feature_df.select('Customer_ID').distinct().count()}")
    
    print("\nFeature Store Schema:")
    for col_name in sorted(feature_df.columns)[:20]:  # Show first 20 columns
        print(f"  - {col_name}")
    if len(feature_df.columns) > 20:
        print(f"  ... and {len(feature_df.columns) - 20} more columns")
    
    print("\nFeature Store Sample:")
    feature_df.select("Customer_ID", "snapshot_date", "total_loans", "annual_income", "customer_age").show(5)
else:
    print("No feature store files found!")

# Check Label Store
print("\nLabel Store Summary:")
label_folder_path = gold_label_store_directory
label_files_list = [label_folder_path + os.path.basename(f) for f in glob.glob(os.path.join(label_folder_path, '*'))]

if label_files_list:
    label_df = spark.read.option("header", "true").parquet(*label_files_list)
    print(f"Label Store - Total rows: {label_df.count()}")
    print(f"Label Store - Unique customers: {label_df.select('Customer_ID').distinct().count()}")
    
    # Label distribution
    label_distribution = label_df.groupBy("label").count().collect()
    print("Label Distribution:")
    for row in label_distribution:
        print(f"  Label {row['label']}: {row['count']} customers")
    
    print("\nLabel Store Sample:")
    label_df.show(5)
    
    # Feature-Label compatibility check
    if feature_files_list:
        feature_customers = feature_df.select("Customer_ID").distinct()
        label_customers = label_df.select("Customer_ID").distinct()
        overlap = feature_customers.intersect(label_customers).count()
        feature_count = feature_customers.count()
        label_count = label_customers.count()
        
        print(f"\n=== ML COMPATIBILITY CHECK ===")
        print(f"Feature store customers: {feature_count}")
        print(f"Label store customers: {label_count}")
        print(f"Overlapping customers: {overlap}")
        print(f"Feature-Label compatibility: {overlap/max(feature_count, 1)*100:.1f}%")
        
        if overlap/max(feature_count, 1) > 0.8:
            print("✅ GOOD: High compatibility - Ready for ML model training!")
        else:
            print("⚠️  WARNING: Low compatibility - Check data alignment")
else:
    print("No label store files found!")

# Data Quality Summary
print(f"\n=== DATA QUALITY SUMMARY ===")
print(f"Bronze files processed: {len(dates_str_lst)} months × 4 datasets")
print(f"Silver files processed: {len(dates_str_lst)} months × 4 datasets") 
print(f"Gold feature store files: {len(feature_files_list) if feature_files_list else 0}")
print(f"Gold label store files: {len(label_files_list) if label_files_list else 0}")

# Cleanup
spark.stop()

print("\n=== FEATURE STORE PIPELINE COMPLETED ===")
print("Ready for Assignment 2: Machine Learning Model Training!")