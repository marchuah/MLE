import argparse
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

# to call this script: python gold_feature_store.py --snapshotdate "2023-01-01" --mob 6

def main(snapshotdate, mob=None, create_labels=False, dpd=30):
    print('\n\n---starting gold feature store job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_feature_store") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # Define directory structure
    silver_directories = {
        'lms': "datamart/silver/lms/",
        'financials': "datamart/silver/financials/", 
        'attributes': "datamart/silver/attributes/",
        'clickstream': "datamart/silver/clickstream/"
    }
    
    gold_feature_store_directory = "datamart/gold/feature_store/"
    gold_label_store_directory = "datamart/gold/label_store/"
    
    # Create directories if they don't exist
    for directory in [gold_feature_store_directory, gold_label_store_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Process gold feature store
    print("=== Creating Gold Feature Store ===")
    feature_store_df = utils.data_processing_gold_table.process_gold_feature_store(
        date_str, 
        silver_directories,
        gold_feature_store_directory, 
        spark,
        mob=mob
    )
    
    # Optionally create labels (for model training)
    if create_labels:
        print(f"\n=== Creating Gold Label Store ({dpd}DPD at {mob}MOB) ===")
        label_store_df = utils.data_processing_gold_table.process_labels_gold_table(
            date_str,
            silver_directories['lms'],
            gold_label_store_directory,
            spark,
            dpd=dpd,
            mob=mob
        )
        
        # Show feature-label compatibility check
        feature_customers = feature_store_df.select("Customer_ID").distinct().count()
        label_customers = label_store_df.select("Customer_ID").distinct().count()
        
        print(f"\n=== Compatibility Check ===")
        print(f"Feature store customers: {feature_customers}")
        print(f"Label store customers: {label_customers}")
        
        # Check overlap
        overlap = feature_store_df.select("Customer_ID").intersect(
            label_store_df.select("Customer_ID")
        ).count()
        print(f"Overlapping customers: {overlap}")
        print(f"Feature-Label compatibility: {overlap/max(feature_customers, 1)*100:.1f}%")
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed gold feature store job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run gold feature store job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD or M/D/YYYY")
    parser.add_argument("--mob", type=int, default=None, help="Month on Book filter (optional)")
    parser.add_argument("--create_labels", action="store_true", help="Also create label store")
    parser.add_argument("--dpd", type=int, default=30, help="Days Past Due threshold for labels (default: 30)")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.mob, args.create_labels, args.dpd)