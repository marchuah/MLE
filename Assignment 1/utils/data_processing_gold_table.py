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

from pyspark.sql.functions import col, when, coalesce, lit, max as spark_max, min as spark_min, mean as spark_mean, stddev, count
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType


def process_gold_feature_store(snapshot_date_str, silver_directories, gold_feature_store_directory, spark, mob=None):
    """
    Create gold feature store by combining and aggregating all silver tables
    """
    print(f"Processing gold feature store for snapshot date: {snapshot_date_str}")
    
    # Handle date format
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        date_suffix = snapshot_date_str.replace('-','_')
    except ValueError:
        snapshot_date = datetime.strptime(snapshot_date_str, "%m/%d/%Y")
        date_suffix = snapshot_date.strftime("%Y_%m_%d")
    
    # Load all silver tables
    silver_dfs = {}
    
    for dataset_name in ['lms', 'financials', 'attributes', 'clickstream']:
        try:
            if dataset_name == 'lms':
                partition_name = f"silver_loan_daily_{date_suffix}.parquet"
            else:
                partition_name = f"silver_{dataset_name}_{date_suffix}.parquet"
            
            filepath = silver_directories[dataset_name] + partition_name
            
            if not os.path.exists(filepath):
                print(f"Warning: Silver file {filepath} not found. Skipping {dataset_name}")
                continue
                
            df = spark.read.parquet(filepath)
            print(f'Loaded {dataset_name} from: {filepath}, row count: {df.count()}')
            silver_dfs[dataset_name] = df
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")
            continue
    
    # Start with LMS data as the base (if available) or use any dataset with Customer_ID
    base_df = None
    
    if 'lms' in silver_dfs:
        base_df = silver_dfs['lms']
        
        # Filter by MOB if specified (for label creation compatibility)
        if mob is not None:
            base_df = base_df.filter(col("mob") == mob)
            print(f"Filtered LMS data to MOB {mob}, row count: {base_df.count()}")
    
    # If no LMS data, start with attributes or financials
    elif 'attributes' in silver_dfs:
        base_df = silver_dfs['attributes'].select("Customer_ID", "snapshot_date").distinct()
    elif 'financials' in silver_dfs:
        base_df = silver_dfs['financials'].select("Customer_ID", "snapshot_date").distinct()
    elif 'clickstream' in silver_dfs:
        base_df = silver_dfs['clickstream'].select("Customer_ID", "snapshot_date").distinct()
    
    if base_df is None:
        raise Exception("No valid silver tables found to create gold feature store")
    
    # Get unique customers for feature engineering
    customers_df = base_df.select("Customer_ID", "snapshot_date").distinct()
    print(f"Found {customers_df.count()} unique customers for feature engineering")
    
    # Build comprehensive feature set
    feature_df = customers_df
    
    # Add LMS features
    if 'lms' in silver_dfs:
        lms_features = create_lms_features(silver_dfs['lms'], mob)
        feature_df = feature_df.join(lms_features, "Customer_ID", "left")
        print("Added LMS features")
    
    # Add financial features
    if 'financials' in silver_dfs:
        financial_features = create_financial_features(silver_dfs['financials'])
        feature_df = feature_df.join(financial_features, "Customer_ID", "left")
        print("Added Financial features")
    
    # Add customer attribute features
    if 'attributes' in silver_dfs:
        attribute_features = create_attribute_features(silver_dfs['attributes'])
        feature_df = feature_df.join(attribute_features, "Customer_ID", "left")
        print("Added Attribute features")
    
    # Add clickstream features  
    if 'clickstream' in silver_dfs:
        clickstream_features = create_clickstream_features(silver_dfs['clickstream'])
        feature_df = feature_df.join(clickstream_features, "Customer_ID", "left")
        print("Added Clickstream features")
    
    # Add gold metadata
    feature_df = feature_df.withColumn('gold_processing_timestamp', F.current_timestamp()) \
                          .withColumn('gold_processing_date', lit(snapshot_date_str)) \
                          .withColumn('feature_store_version', lit('1.0'))
    
    if mob is not None:
        feature_df = feature_df.withColumn('mob_filter', lit(mob))
    
    print(f"Final feature store shape: {feature_df.count()} rows, {len(feature_df.columns)} columns")
    print(f"Feature columns: {feature_df.columns}")
    
    # Save gold feature store
    partition_name = f"gold_feature_store_{date_suffix}.parquet"
    if mob is not None:
        partition_name = f"gold_feature_store_{mob}mob_{date_suffix}.parquet"
        
    filepath = gold_feature_store_directory + partition_name
    feature_df.write.mode("overwrite").parquet(filepath)
    print(f'Saved gold feature store to: {filepath}')
    
    return feature_df


def create_lms_features(lms_df, mob=None):
    """
    Create aggregated features from LMS loan data
    """
    print("Creating LMS features...")
    
    # Customer-level loan aggregations
    lms_features = lms_df.groupBy("Customer_ID").agg(
        count("loan_id").alias("total_loans"),
        spark_max("loan_amt").alias("max_loan_amount"),
        spark_min("loan_amt").alias("min_loan_amount"),
        spark_mean("loan_amt").alias("avg_loan_amount"),
        spark_max("tenure").alias("max_tenure"),
        spark_mean("tenure").alias("avg_tenure"),
        spark_max("balance").alias("current_total_balance"),
        spark_mean("balance").alias("avg_balance"),
        spark_max("overdue_amt").alias("max_overdue_amount"),
        spark_mean("overdue_amt").alias("avg_overdue_amount"),
        spark_max("dpd").alias("max_dpd"),
        spark_mean("dpd").alias("avg_dpd"),
        spark_max("mob").alias("max_mob"),
        spark_mean("paid_amt").alias("avg_paid_amount")
    )
    
    # Add derived features
    lms_features = lms_features.withColumn("total_loan_exposure", 
                                          col("total_loans") * col("avg_loan_amount"))
    
    lms_features = lms_features.withColumn("utilization_ratio",
                                          when(col("avg_loan_amount") > 0,
                                               col("avg_balance") / col("avg_loan_amount"))
                                          .otherwise(0))
    
    lms_features = lms_features.withColumn("delinquency_flag",
                                          when(col("max_dpd") > 0, 1).otherwise(0))
    
    lms_features = lms_features.withColumn("high_risk_flag",
                                          when(col("max_dpd") >= 30, 1).otherwise(0))
    
    return lms_features


def create_financial_features(financial_df):
    """
    Create features from financial data
    """
    print("Creating Financial features...")
    
    # Select and rename key financial metrics
    financial_features = financial_df.select(
        "Customer_ID",
        col("Annual_Income").alias("annual_income"),
        col("Monthly_Inhand_Salary").alias("monthly_salary"),
        col("Num_Bank_Accounts").alias("num_bank_accounts"),
        col("Num_Credit_Card").alias("num_credit_cards"),
        col("Interest_Rate").alias("avg_interest_rate"),
        col("Num_of_Loan").alias("external_loans_count"),
        col("Outstanding_Debt").alias("total_outstanding_debt"),
        col("Credit_Utilization_Ratio").alias("credit_utilization"),
        col("Total_EMI_per_month").alias("monthly_emi"),
        col("Amount_invested_monthly").alias("monthly_investment"),
        col("Monthly_Balance").alias("monthly_balance"),
        col("debt_to_income_ratio"),
        col("monthly_debt_service_ratio")
    )
    
    # Add derived financial health indicators
    financial_features = financial_features.withColumn("financial_health_score",
        when(col("debt_to_income_ratio") <= 0.3, 5)
        .when(col("debt_to_income_ratio") <= 0.5, 4)
        .when(col("debt_to_income_ratio") <= 0.7, 3)
        .when(col("debt_to_income_ratio") <= 1.0, 2)
        .otherwise(1)
    )
    
    financial_features = financial_features.withColumn("liquidity_ratio",
        when(col("monthly_salary") > 0,
             col("monthly_balance") / col("monthly_salary"))
        .otherwise(None)
    )
    
    financial_features = financial_features.withColumn("investment_ratio",
        when(col("monthly_salary") > 0,
             col("monthly_investment") / col("monthly_salary"))
        .otherwise(0)
    )
    
    return financial_features


def create_attribute_features(attributes_df):
    """
    Create features from customer attributes
    """
    print("Creating Attribute features...")
    
    attribute_features = attributes_df.select(
        "Customer_ID",
        col("Age").alias("customer_age"),
        col("SSN_is_valid").alias("ssn_valid_flag"),
        col("Occupation").alias("occupation")
    )
    
    # Add age-based risk categories
    attribute_features = attribute_features.withColumn("age_risk_category",
        when(col("customer_age") <= 25, "young")
        .when(col("customer_age") <= 35, "young_adult") 
        .when(col("customer_age") <= 50, "middle_age")
        .when(col("customer_age") <= 65, "mature")
        .otherwise("senior")
    )
    
    # Create occupation-based risk flags (example categorization)
    high_stability_occupations = ['Lawyer', 'Engineer', 'Doctor', 'Teacher']
    
    attribute_features = attribute_features.withColumn("occupation_stability_score",
        when(col("occupation").isin(high_stability_occupations), 5)
        .when(col("occupation").isNotNull(), 3)
        .otherwise(1)
    )
    
    return attribute_features


def create_clickstream_features(clickstream_df):
    """
    Create features from clickstream data
    """
    print("Creating Clickstream features...")
    
    # Get all feature columns
    feature_cols = [col_name for col_name in clickstream_df.columns if col_name.startswith('fe_')]
    
    clickstream_features = clickstream_df.select(
        "Customer_ID",
        col("total_positive_features").alias("clickstream_positive_activity"),
        col("total_negative_features").alias("clickstream_negative_activity"), 
        col("feature_balance_ratio").alias("clickstream_balance_ratio"),
        *[col(f_col).alias(f"clickstream_{f_col}") for f_col in feature_cols]
    )
    
    # Add engagement level categorization
    clickstream_features = clickstream_features.withColumn("engagement_level",
        when(col("clickstream_positive_activity") >= 100, "high")
        .when(col("clickstream_positive_activity") >= 50, "medium")
        .otherwise("low")
    )
    
    return clickstream_features


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    """
    Legacy function to create labels (separate from feature store)
    """
    print(f"Processing gold label store for {dpd}DPD at {mob}MOB")
    
    # Handle date format
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        date_suffix = snapshot_date_str.replace('-','_')
    except ValueError:
        snapshot_date = datetime.strptime(snapshot_date_str, "%m/%d/%Y")
        date_suffix = snapshot_date.strftime("%Y_%m_%d")
    
    # Load silver LMS data
    partition_name = f"silver_loan_daily_{date_suffix}.parquet"
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # Filter to specific MOB
    df = df.filter(col("mob") == mob)

    # Create label
    df = df.withColumn("label", when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    # Select columns for label store
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # Save gold label store
    partition_name = f"gold_label_store_{dpd}dpd_{mob}mob_{date_suffix}.parquet"
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')

    return df