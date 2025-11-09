🧱 Credit Default Prediction Pipeline (Bronze → Silver → Gold)
📖 Overview

This project implements an end-to-end data engineering and machine learning pipeline for credit default prediction, built entirely on the Medallion Architecture using PySpark and Apache Airflow.

The pipeline automates data ingestion, cleaning, feature engineering, and model-ready data preparation.
It demonstrates the principles of MLOps and data reliability, producing repeatable, auditable, and production-ready datasets.

MLE Pipeline

🪣 Bronze Layer – Ingests raw monthly CSV extracts from multiple data sources (loan, financials, attributes, clickstream)

🧼 Silver Layer – Cleans, normalizes, and type-casts data; handles missing values and outliers

🪙 Gold Layer – Builds label and feature stores used for model training and inference

⚙️ Airflow DAG orchestrates all layers in sequence (Bronze → Silver → Gold)

Prerequisites

🐳 Docker & Docker Compose

🐍 Python 3.11+ (if testing locally)

Apache Airflow (auto-deployed via Docker)


# Start containers
docker-compose up --build 
(if the airflow got stuck and did not start, you can press ctrl+c and then docker compose up again)

This will start:

airflow-webserver at http://localhost:8080
Username: admin
Password: admin

Trigger the DAG main_etl_pipeline
Monitor progress (Bronze → Silver → Gold > Inference > Monitoring)

Output will be Datamart, 

jupyter_lab at http://localhost:8888

