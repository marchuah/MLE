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


def process_gold_feature_store(snapshot_date_str, silver_directories, gold_feature_store_directory, spark, mob=0):
    """
    Create gold feature store for MOB=0 (application time) prediction
    
    Business Case: Predict if customer will default using only 
    information available at MOB=0 (loan application time)
    
    Args:
        mob: Filter to specific MOB for features. Use 0 for application-time features.
    """
    print(f"Processing gold feature store for snapshot date: {snapshot_date_str}")
    print(f"Feature MOB filter: {mob} (Application Time)")
    
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
    
    # Start with LMS data filtered to MOB=0 (application time)
    base_df = None
    
    if 'lms' in silver_dfs:
        # CRITICAL: Filter to MOB=0 only for application-time features
        base_df = silver_dfs['lms'].filter(col("mob") == mob)
        print(f"Filtered LMS data to MOB={mob}, row count: {base_df.count()}")
    
    # If no LMS data at MOB=0, start with other sources
    if base_df is None or base_df.count() == 0:
        if 'attributes' in silver_dfs:
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
    
    # Build comprehensive feature set (all at MOB=0)
    feature_df = customers_df
    
    # Add LMS features (historical loan behavior at application time)
    if 'lms' in silver_dfs:
        lms_features = create_lms_features_mob0(silver_dfs['lms'], mob)
        feature_df = feature_df.join(lms_features, "Customer_ID", "left")
        print("Added LMS features (MOB=0 historical data)")
    
    # Add financial features (snapshot at application time)
    if 'financials' in silver_dfs:
        financial_features = create_financial_features(silver_dfs['financials'])
        feature_df = feature_df.join(financial_features, "Customer_ID", "left")
        print("Added Financial features")
    
    # Add customer attribute features
    if 'attributes' in silver_dfs:
        attribute_features = create_attribute_features(silver_dfs['attributes'])
        feature_df = feature_df.join(attribute_features, "Customer_ID", "left")
        print("Added Attribute features")
    
    # Add clickstream features (pre-application behavior)
    if 'clickstream' in silver_dfs:
        clickstream_features = create_clickstream_features(silver_dfs['clickstream'])
        feature_df = feature_df.join(clickstream_features, "Customer_ID", "left")
        print("Added Clickstream features")
    
    # Add gold metadata
    feature_df = feature_df.withColumn('gold_processing_timestamp', F.current_timestamp()) \
                          .withColumn('gold_processing_date', lit(snapshot_date_str)) \
                          .withColumn('feature_store_version', lit('1.0')) \
                          .withColumn('feature_mob', lit(mob))
    
    print(f"Final feature store shape: {feature_df.count()} rows, {len(feature_df.columns)} columns")
    print(f"Feature columns: {feature_df.columns}")
    
    # Save gold feature store
    partition_name = f"gold_feature_store_{date_suffix}.parquet"
    filepath = gold_feature_store_directory + partition_name
    feature_df.write.mode("overwrite").parquet(filepath)
    print(f'Saved gold feature store to: {filepath}')
    
    return feature_df


def create_lms_features_mob0(lms_df, mob=0):
    """
    Create features from LMS data at MOB=0 (application time)
    
    Key Point: At MOB=0, we capture:
    - Current loan application details (amount, tenure requested)
    - Historical performance on PAST loans (if repeat customer)
    - NO future performance data from this loan
    """
    print("Creating LMS features at MOB=0...")
    
    # Filter to MOB=0 only
    lms_mob0 = lms_df.filter(col("mob") == mob)
    
    # Application-time loan characteristics (what they're applying for NOW)
    lms_features = lms_mob0.groupBy("Customer_ID").agg(
        count("loan_id").alias("num_active_loans_at_application"),
        spark_max("loan_amt").alias("max_requested_loan_amount"),
        spark_mean("loan_amt").alias("avg_requested_loan_amount"),
        spark_max("tenure").alias("max_requested_tenure"),
        spark_mean("tenure").alias("avg_requested_tenure"),
        # At MOB=0, balance should equal loan amount (just disbursed)
        spark_max("balance").alias("initial_total_balance"),
        spark_mean("balance").alias("avg_initial_balance")
    )
    
    # Add derived application-time features
    lms_features = lms_features.withColumn("total_loan_exposure_at_application", 
                                          col("num_active_loans_at_application") * col("avg_requested_loan_amount"))
    
    # Flag for multiple simultaneous loan applications
    lms_features = lms_features.withColumn("multiple_loans_flag",
                                          when(col("num_active_loans_at_application") > 1, 1).otherwise(0))
    
    # High value application flag
    lms_features = lms_features.withColumn("high_value_application",
                                          when(col("max_requested_loan_amount") >= 50000, 1).otherwise(0))
    
    return lms_features


def create_financial_features(financial_df):
    """
    Create features from financial data (snapshot at application time)
    """
    print("Creating Financial features...")
    
    # Select and rename key financial metrics
    financial_features = financial_df.select(
        "Customer_ID",
        col("Annual_Income").alias("annual_income"),
        col("Monthly_Inhand_Salary").alias("monthly_salary"),
        col("Num_Bank_Accounts").alias("num_bank_accounts"),
        col("Num_Credit_Card").alias("num_credit_cards"),
        col("Interest_Rate").alias("existing_avg_interest_rate"),
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
    
    # Add savings capacity indicator
    financial_features = financial_features.withColumn("discretionary_income",
        col("monthly_salary") - col("monthly_emi")
    )
    
    financial_features = financial_features.withColumn("savings_capacity_ratio",
        when(col("monthly_salary") > 0,
             col("discretionary_income") / col("monthly_salary"))
        .otherwise(0)
    )
    
    # Credit portfolio diversity
    financial_features = financial_features.withColumn("credit_product_diversity",
        col("num_bank_accounts") + col("num_credit_cards") + col("external_loans_count")
    )
    
    # High leverage flag
    financial_features = financial_features.withColumn("high_leverage_flag",
        when(col("debt_to_income_ratio") > 0.7, 1).otherwise(0)
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
    
    # Create occupation-based stability score
    high_stability_occupations = ['Lawyer', 'Engineer', 'Doctor', 'Teacher', 'Accountant', 'Scientist']
    medium_stability_occupations = ['Manager', 'Architect', 'Developer', 'Analyst']
    
    attribute_features = attribute_features.withColumn("occupation_stability_score",
        when(col("occupation").isin(high_stability_occupations), 5)
        .when(col("occupation").isin(medium_stability_occupations), 3)
        .when(col("occupation").isNotNull(), 2)
        .otherwise(1)
    )
    
    # Age-based maturity score
    attribute_features = attribute_features.withColumn("financial_maturity_score",
        when((col("customer_age") >= 30) & (col("customer_age") <= 55), 5)
        .when((col("customer_age") >= 25) & (col("customer_age") < 30), 4)
        .when((col("customer_age") >= 56) & (col("customer_age") <= 65), 4)
        .when(col("customer_age") < 25, 2)
        .otherwise(3)
    )
    
    return attribute_features


def create_clickstream_features(clickstream_df):
    """
    Create features from clickstream data (pre-application browsing behavior)
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
    
    # Add engagement quality score
    clickstream_features = clickstream_features.withColumn("engagement_quality_score",
        when(col("clickstream_balance_ratio") >= 0.8, 5)
        .when(col("clickstream_balance_ratio") >= 0.6, 4)
        .when(col("clickstream_balance_ratio") >= 0.4, 3)
        .when(col("clickstream_balance_ratio") >= 0.2, 2)
        .otherwise(1)
    )
    
    # Pre-application research indicator
    clickstream_features = clickstream_features.withColumn("thorough_researcher",
        when((col("clickstream_positive_activity") >= 50) & 
             (col("clickstream_balance_ratio") >= 0.6), 1).otherwise(0)
    )
    
    return clickstream_features


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob, label_type="within_window"):
    """
    Create labels for model training
    Labels come from FUTURE loan performance observation
    
    Args:
        dpd: Days past due threshold (e.g., 30 for default definition)
        mob: Observation window in months (e.g., 6 means observe up to MOB=6)
        label_type: "within_window" (default anytime in MOB 1-6) or "at_mob" (only at specific MOB)
    
    Business Cases:
    - "within_window": Label=1 if customer defaults ANYTIME within observation window
    - "at_mob": Label=1 if customer is in default specifically at the target MOB
    """
    print(f"Processing gold label store for {dpd}DPD within {mob}MOB window")
    print(f"Label type: {label_type}")
    
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

    if label_type == "within_window":
        # Label = 1 if customer EVER defaults within the observation window (MOB 1 to mob)
        print(f"Creating labels: Default anytime between MOB=1 and MOB={mob}")
        df_window = df.filter((col("mob") >= 1) & (col("mob") <= mob))
        
        # For each customer-loan, check if they ever hit the DPD threshold
        df_labels = df_window.groupBy("loan_id", "Customer_ID", "snapshot_date").agg(
            spark_max(when(col("dpd") >= dpd, 1).otherwise(0)).alias("label")
        )
        
        df_labels = df_labels.withColumn("label_def", 
                                        lit(f"{dpd}dpd_within_{mob}mob").cast(StringType()))
        
    else:  # "at_mob"
        # Label = 1 if customer is in default at the specific MOB
        print(f"Creating labels: Default status at MOB={mob}")
        df = df.filter(col("mob") == mob)
        df_labels = df.withColumn("label", 
                                  when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
        df_labels = df_labels.withColumn("label_def", 
                                        lit(f"{dpd}dpd_at_{mob}mob").cast(StringType()))
        df_labels = df_labels.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # Print label distribution
    label_dist = df_labels.groupBy("label").count()
    print("Label distribution:")
    label_dist.show()

    # Save gold label store

    partition_name = f"gold_label_store_{dpd}dpd_{mob}mob_{date_suffix}.parquet"
    filepath = gold_label_store_directory + partition_name
    df_labels.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')

    return df_labels