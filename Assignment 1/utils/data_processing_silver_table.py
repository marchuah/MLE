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

from pyspark.sql.functions import col, when, regexp_replace, trim, upper
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType


def process_silver_tables(snapshot_date_str, bronze_directories, silver_directories, spark):
    """
    Process all bronze tables into silver tables with data cleaning and transformations
    """
    print(f"Processing silver tables for snapshot date: {snapshot_date_str}")
    
    # Handle date format
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        date_suffix = snapshot_date_str.replace('-','_')
    except ValueError:
        snapshot_date = datetime.strptime(snapshot_date_str, "%m/%d/%Y")
        date_suffix = snapshot_date.strftime("%Y_%m_%d")
    
    processed_dfs = {}
    
    # Process each dataset
    datasets = ['lms', 'financials', 'attributes', 'clickstream']
    
    for dataset_name in datasets:
        print(f"\n--- Processing {dataset_name} silver table ---")
        
        try:
            # Load bronze table
            partition_name = f"bronze_{dataset_name.replace('lms', 'loan_daily')}_{date_suffix}.csv"
            if dataset_name == 'lms':
                partition_name = f"bronze_loan_daily_{date_suffix}.csv"
            elif dataset_name == 'financials':
                partition_name = f"bronze_financials_{date_suffix}.csv"
            elif dataset_name == 'attributes':
                partition_name = f"bronze_attributes_{date_suffix}.csv"
            elif dataset_name == 'clickstream':
                partition_name = f"bronze_clickstream_{date_suffix}.csv"
            
            bronze_filepath = bronze_directories[dataset_name] + partition_name
            
            if not os.path.exists(bronze_filepath):
                print(f"Warning: Bronze file {bronze_filepath} not found. Skipping {dataset_name}")
                continue
            
            df = spark.read.csv(bronze_filepath, header=True, inferSchema=False)  # Load as strings first
            print(f'Loaded from: {bronze_filepath}, row count: {df.count()}')
            
            # Apply dataset-specific transformations
            if dataset_name == 'lms':
                df_clean = process_lms_silver(df)
            elif dataset_name == 'financials':
                df_clean = process_financials_silver(df)
            elif dataset_name == 'attributes':
                df_clean = process_attributes_silver(df)
            elif dataset_name == 'clickstream':
                df_clean = process_clickstream_silver(df)
            
            # Add silver metadata
            df_clean = df_clean.withColumn('silver_processing_timestamp', F.current_timestamp()) \
                              .withColumn('silver_processing_date', F.lit(snapshot_date_str))
            
            # Save silver table
            silver_partition_name = f"silver_{dataset_name.replace('lms', 'loan_daily')}_{date_suffix}.parquet"
            silver_filepath = silver_directories[dataset_name] + silver_partition_name
            
            df_clean.write.mode("overwrite").parquet(silver_filepath)
            print(f'Saved to: {silver_filepath}')
            
            processed_dfs[dataset_name] = df_clean
            
        except Exception as e:
            print(f"Error processing {dataset_name} silver table: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return processed_dfs


def process_lms_silver(df):
    """
    Clean and transform LMS loan data
    """
    print("Processing LMS data transformations...")
    
    # Define schema and clean data types
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # Apply type conversions with error handling
    for column, new_type in column_type_map.items():
        if column in df.columns:
            if new_type == DateType():
                # Handle multiple date formats
                df = df.withColumn(column, 
                    F.coalesce(
                        F.to_date(col(column), 'yyyy-MM-dd'),
                        F.to_date(col(column), 'M/d/yyyy'),
                        F.to_date(col(column), 'MM/dd/yyyy')
                    )
                )
            else:
                df = df.withColumn(column, col(column).cast(new_type))

    # Add business logic transformations
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    
    # Calculate installments missed and DPD
    df = df.withColumn("installments_missed", 
                      F.when(col("due_amt") > 0, 
                            F.ceil(col("overdue_amt") / col("due_amt")))
                       .otherwise(0).cast(IntegerType()))
    
    df = df.withColumn("first_missed_date", 
                      F.when(col("installments_missed") > 0, 
                            F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
                       .cast(DateType()))
    
    df = df.withColumn("dpd", 
                      F.when(col("overdue_amt") > 0.0, 
                            F.datediff(col("snapshot_date"), col("first_missed_date")))
                       .otherwise(0).cast(IntegerType()))
    
    return df


def process_financials_silver(df):
    """
    Clean and transform financial features data
    """
    print("Processing Financials data transformations...")
    
    # Clean string columns - remove special characters and standardize
    string_columns = ['Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    
    for col_name in string_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, 
                              regexp_replace(trim(col(col_name)), r'[*#$%&@]', ''))
            df = df.withColumn(col_name,
                              when(col(col_name) == '', None).otherwise(col(col_name)))
    
    # Clean numeric columns - handle dirty data
    numeric_columns = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]
    
    for col_name in numeric_columns:
        if col_name in df.columns:
            # Remove trailing underscores and non-numeric characters except decimals and minus signs
            df = df.withColumn(col_name,
                              regexp_replace(col(col_name), r'[^0-9.-]', ''))
            # Convert to double, handling empty strings as null
            df = df.withColumn(col_name,
                              when(col(col_name) == '', None)
                              .otherwise(col(col_name).cast(DoubleType())))
    
    # Handle date columns
    if 'snapshot_date' in df.columns:
        df = df.withColumn('snapshot_date',
                          F.coalesce(
                              F.to_date(col('snapshot_date'), 'yyyy-MM-dd'),
                              F.to_date(col('snapshot_date'), 'M/d/yyyy'),
                              F.to_date(col('snapshot_date'), 'MM/dd/yyyy')
                          ))
    
    # Add derived features
    df = df.withColumn('debt_to_income_ratio',
                      when(col('Annual_Income') > 0,
                           col('Outstanding_Debt') / col('Annual_Income'))
                      .otherwise(None))
    
    df = df.withColumn('monthly_debt_service_ratio',
                      when(col('Monthly_Inhand_Salary') > 0,
                           col('Total_EMI_per_month') / col('Monthly_Inhand_Salary'))
                      .otherwise(None))
    
    return df


def process_attributes_silver(df):
    """
    Clean and transform customer attributes data
    """
    print("Processing Attributes data transformations...")
    
    # Clean customer name
    if 'Name' in df.columns:
        df = df.withColumn('Name', trim(col('Name')))
        df = df.withColumn('Name_cleaned', 
                          regexp_replace(col('Name'), r'[^a-zA-Z\s]', ''))
    
    # Handle age - convert to integer and validate
    if 'Age' in df.columns:
        df = df.withColumn('Age',
                          when(col('Age').cast(IntegerType()).between(18, 100),
                               col('Age').cast(IntegerType()))
                          .otherwise(None))
    
    # Clean SSN - mask corrupted SSNs but keep valid ones
    if 'SSN' in df.columns:
        df = df.withColumn('SSN_is_valid',
                          col('SSN').rlike(r'^\d{3}-\d{2}-\d{4}$'))
        df = df.withColumn('SSN_cleaned',
                          when(col('SSN_is_valid'), col('SSN'))
                          .otherwise('CORRUPTED'))
    
    # Clean occupation
    if 'Occupation' in df.columns:
        df = df.withColumn('Occupation',
                          regexp_replace(trim(col('Occupation')), '_', ' '))
        df = df.withColumn('Occupation', 
                          regexp_replace(col('Occupation'), r'[^a-zA-Z\s]', ''))
    
    # Handle date columns
    if 'snapshot_date' in df.columns:
        df = df.withColumn('snapshot_date',
                          F.coalesce(
                              F.to_date(col('snapshot_date'), 'yyyy-MM-dd'),
                              F.to_date(col('snapshot_date'), 'M/d/yyyy'),
                              F.to_date(col('snapshot_date'), 'MM/dd/yyyy')
                          ))
    
    return df


def process_clickstream_silver(df):
    """
    Clean and transform clickstream features data
    """
    print("Processing Clickstream data transformations...")
    
    # Get feature columns (fe_1 through fe_20)
    feature_cols = [col_name for col_name in df.columns if col_name.startswith('fe_')]
    
    # Convert all feature columns to double type
    for col_name in feature_cols:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
    
    # Handle date columns  
    if 'snapshot_date' in df.columns:
        df = df.withColumn('snapshot_date',
                          F.coalesce(
                              F.to_date(col('snapshot_date'), 'yyyy-MM-dd'),
                              F.to_date(col('snapshot_date'), 'M/d/yyyy'),
                              F.to_date(col('snapshot_date'), 'MM/dd/yyyy')
                          ))
    
    # Add some derived features for clickstream analysis
    # Sum of all positive and negative features
    positive_features = [F.when(col(col_name) > 0, col(col_name)).otherwise(0) 
                        for col_name in feature_cols]
    negative_features = [F.when(col(col_name) < 0, col(col_name)).otherwise(0) 
                        for col_name in feature_cols]
    
    if positive_features:
        df = df.withColumn('total_positive_features', sum(positive_features))
        df = df.withColumn('total_negative_features', sum(negative_features))
        df = df.withColumn('feature_balance_ratio',
                          when(col('total_positive_features') != 0,
                               col('total_negative_features') / col('total_positive_features'))
                          .otherwise(None))
    
    return df