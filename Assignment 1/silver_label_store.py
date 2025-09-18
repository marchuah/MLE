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

# to call this script: python silver_feature_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting silver feature store job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_feature_store") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create bronze and silver datalake directories for all feature datasets
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
    
    # Create directories if they don't exist
    for dir_path in silver_directories.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # run data processing for silver tables
    utils.data_processing_silver_table.process_silver_tables(
        date_str, 
        bronze_directories,
        silver_directories, 
        spark
    )
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed silver feature store job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run silver feature store job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)