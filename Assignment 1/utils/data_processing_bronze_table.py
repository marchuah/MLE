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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_tables(snapshot_date_str, bronze_directories, spark):
    """
    Process all feature datasets into bronze tables
    """
    print(f"Processing bronze tables for snapshot date: {snapshot_date_str}")
    
    # prepare arguments - handle both date formats
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        target_date_format = snapshot_date_str  # Keep YYYY-MM-DD format for filtering
    except ValueError:
        # If input is in M/D/YYYY format, convert it
        snapshot_date = datetime.strptime(snapshot_date_str, "%m/%d/%Y")
        target_date_format = snapshot_date.strftime("%Y-%m-%d")
    
    print(f"Target snapshot date: {target_date_format}")
    
    # Dataset configurations
    datasets_config = {
        'lms': {
            'file_path': "data/lms_loan_daily.csv",
            'directory': bronze_directories['lms'],
            'partition_prefix': "bronze_loan_daily_",
            'date_column': 'snapshot_date'
        },
        'financials': {
            'file_path': "data/features_financials.csv", 
            'directory': bronze_directories['financials'],
            'partition_prefix': "bronze_financials_",
            'date_column': 'snapshot_date'
        },
        'attributes': {
            'file_path': "data/features_attributes.csv",
            'directory': bronze_directories['attributes'], 
            'partition_prefix': "bronze_attributes_",
            'date_column': 'snapshot_date'
        },
        'clickstream': {
            'file_path': "data/features_clickstream.csv",
            'directory': bronze_directories['clickstream'],
            'partition_prefix': "bronze_clickstream_",
            'date_column': 'snapshot_date'
        }
    }
    
    processed_dfs = {}
    
    # Process each dataset
    for dataset_name, config in datasets_config.items():
        print(f"\n--- Processing {dataset_name} dataset ---")
        
        try:
            # Check if file exists
            if not os.path.exists(config['file_path']):
                print(f"Warning: File {config['file_path']} not found. Skipping {dataset_name}")
                continue
                
            # Load data from CSV with explicit string type for date column to handle various formats
            df = spark.read.csv(config['file_path'], header=True, inferSchema=False)
            
            print(f"Loaded {dataset_name} with {df.count()} total rows")
            print(f"Schema: {df.columns}")
            
            # Show sample of date column to understand format
            if config['date_column'] in df.columns:
                sample_dates = df.select(config['date_column']).distinct().limit(5).collect()
                print(f"Sample {config['date_column']} values: {[row[0] for row in sample_dates]}")
                
                # Normalize date column to YYYY-MM-DD format for consistent filtering
                df_with_normalized_date = df.withColumn(
                    'normalized_snapshot_date',
                    F.when(
                        # Check if date is in M/D/YYYY format
                        F.col(config['date_column']).rlike(r'^\d{1,2}/\d{1,2}/\d{4}'

),
                        F.date_format(
                            F.to_date(F.col(config['date_column']), 'M/d/yyyy'),
                            'yyyy-MM-dd'
                        )
                    ).when(
                        # Check if date is already in YYYY-MM-DD format
                        F.col(config['date_column']).rlike(r'^\d{4}-\d{2}-\d{2}'


),
                        F.col(config['date_column'])
                    ).otherwise(
                        # Try to parse other formats or set to null
                        F.lit(None)
                    )
                )
                
                # Filter by snapshot date
                df_filtered = df_with_normalized_date.filter(
                    F.col('normalized_snapshot_date') == target_date_format
                )
                
                filtered_count = df_filtered.count()
                print(f"{dataset_name} - {target_date_format} row count: {filtered_count}")
                
                if filtered_count == 0:
                    print(f"Warning: No records found for {target_date_format} in {dataset_name}")
                    # Optionally, you might want to continue processing with all data or skip this dataset
                    continue
                    
            else:
                # If no snapshot_date column, take all data but add metadata
                df_filtered = df.withColumn('processed_date', F.lit(target_date_format))
                print(f"{dataset_name} - total row count (no date filtering): {df_filtered.count()}")
            
            # Add bronze metadata columns for data lineage and quality tracking
            df_with_metadata = df_filtered.withColumn('bronze_ingestion_timestamp', F.current_timestamp()) \
                                         .withColumn('bronze_source_file', F.lit(config['file_path'])) \
                                         .withColumn('bronze_processing_date', F.lit(target_date_format))
            
            # Save bronze table to datamart
            partition_name = config['partition_prefix'] + target_date_format.replace('-','_') + '.csv'
            filepath = config['directory'] + partition_name
            
            # Convert to pandas for CSV output (handle potential data quality issues)
            pandas_df = df_with_metadata.toPandas()
            
            # Basic data quality logging
            print(f"Final {dataset_name} dataset shape: {pandas_df.shape}")
            print(f"Columns: {list(pandas_df.columns)}")
            
            pandas_df.to_csv(filepath, index=False)
            print(f"Saved {dataset_name} bronze table to: {filepath}")
            
            processed_dfs[dataset_name] = df_with_metadata
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return processed_dfs