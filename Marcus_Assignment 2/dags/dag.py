# ============================================================
# AIRFLOW DAG — End-to-End MLE Pipeline (Datamart → Bronze → Silver → Gold → Model → Monitor)
# ============================================================

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
import os

# -------------------------------
# Default DAG Arguments
# -------------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# -------------------------------
# Helper Function: Check Silver Outputs
# -------------------------------
def check_silver_outputs():
    """Skip Gold layer if Silver output folders are missing or empty."""
    required_dirs = [
        "/opt/airflow/datamart/silver/lms/",
        "/opt/airflow/datamart/silver/financials/",
        "/opt/airflow/datamart/silver/attributes/",
        "/opt/airflow/datamart/silver/clickstream/",
    ]
    all_exist = True
    for d in required_dirs:
        if not os.path.exists(d) or not os.listdir(d):
            print(f"⚠️ Missing or empty Silver output folder: {d}")
            all_exist = False
    if not all_exist:
        print("⚠️ Silver data not ready. Skipping Gold task for this run.")
    else:
        print("✅ All Silver outputs detected — proceeding to Gold.")
    return all_exist

# -------------------------------
# DAG Definition
# -------------------------------
with DAG(
    'dag',
    default_args=default_args,
    description='End-to-end monthly data pipeline for datamart creation, feature store, model inference, and monitoring',
    schedule_interval='0 0 1 * *',  # Run monthly at midnight
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,     # Ensures serialized monthly runs
    concurrency=1,         # Prevents overlapping runs
    tags=['MLE', 'pipeline', 'datamart', 'monitoring']
) as dag:

    # ============================================================
    # 1️⃣  Label & Feature Store Pipeline (monthly incremental)
    # ============================================================
    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    # --- Bronze Layer ---
    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/utils && '
            'python3 data_processing_bronze_table.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- Silver Layer ---
    silver_label_store = BashOperator(
        task_id='silver_label_store',
        bash_command=(
            'cd /opt/airflow/utils && '
            'python3 data_processing_silver_table.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- Silver Output Check ---
    check_silver_ready = ShortCircuitOperator(
        task_id='check_silver_ready',
        python_callable=check_silver_outputs,
    )

    # --- Gold Layer ---
    gold_label_feature_store = BashOperator(
        task_id='gold_label_feature_store',
        bash_command=(
            'cd /opt/airflow/utils && '
            'python3 data_processing_gold_table.py '
            '--silver_lms_dir "/opt/airflow/datamart/silver/lms/" '
            '--silver_financials_dir "/opt/airflow/datamart/silver/financials/" '
            '--silver_attributes_dir "/opt/airflow/datamart/silver/attributes/" '
            '--silver_clickstream_dir "/opt/airflow/datamart/silver/clickstream/" '
            '--gold_features_dir "/opt/airflow/datamart/gold/feature_store/" '
            '--dpd_threshold 30'
        ),
    )

    label_feature_store_completed = DummyOperator(task_id="label_feature_store_completed")

    # Define execution order for Data Layers
    dep_check_source_label_data >> bronze_label_store >> silver_label_store
    silver_label_store >> check_silver_ready >> gold_label_feature_store >> label_feature_store_completed

    # ============================================================
    # 2️⃣  Model Inference Pipeline
    # ============================================================
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_1_inference = BashOperator(
        task_id="model_1_inference",
        bash_command=(
            'cd /opt/airflow/utils && '
            'python3 model_inference.py "{{ ds }}"'
        ),
    )

    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    label_feature_store_completed >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed

    # ============================================================
    # 3️⃣  Model Monitoring Pipeline
    # ============================================================
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_1_monitor = BashOperator(
        task_id="model_1_monitor",
        bash_command=(
            'cd /opt/airflow/utils && '
            'python3 model_monitoring.py'
        ),
    )

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")

    # Define execution order
    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed

# ============================================================
# END OF DAG
# ============================================================
