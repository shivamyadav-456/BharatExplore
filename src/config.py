"""Configuration settings and constants for BharatExplore."""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANALYTICS_DATA_DIR = DATA_DIR / "analytics"
MODELS_DATA_DIR = DATA_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

# Ensure all critical directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ANALYTICS_DATA_DIR, MODELS_DATA_DIR, REPORTS_DIR, DOCS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset Paths
DEFAULT_RAW_ZIP = RAW_DATA_DIR / "archive.zip"
DEFAULT_RAW_CSV = RAW_DATA_DIR / "Review_db.csv"
FALLBACK_RAW_ZIP = Path.home() / "Downloads" / "archive.zip"

# Processed and Analytics File Paths
CLEANED_REVIEWS_PARQUET = PROCESSED_DATA_DIR / "cleaned_reviews.parquet"
DESTINATION_PROFILES_PARQUET = ANALYTICS_DATA_DIR / "destination_profiles.parquet"
CITY_STATS_PARQUET = ANALYTICS_DATA_DIR / "city_stats.parquet"
SENTIMENT_SAMPLE_PARQUET = ANALYTICS_DATA_DIR / "sentiment_sample.parquet"

# Reports
DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality_report.json"
ANALYSIS_SUMMARY_REPORT = REPORTS_DIR / "analysis_summary.json"
MODEL_EVALUATION_REPORT = REPORTS_DIR / "model_evaluation.json"

# Spark Configuration Settings
SPARK_APP_NAME = "BharatExplore_BigData_Pipeline"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "4g"

# Java 17+ JVM add-opens options needed for modern JVMs (Java 17, 21, 23)
JVM_ADD_OPENS = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
)

# Pipeline & Evaluation Thresholds
MIN_REVIEWS_DEFAULT = 30
DEV_MODE_SAMPLE_FRACTION = 0.05  # For lightweight testing when memory constrained
TOP_DESTINATIONS_LIMIT = 50

# Recommendation Engine Default Weights
REC_WEIGHT_SIMILARITY = 0.40
REC_WEIGHT_BAYESIAN_RATING = 0.35
REC_WEIGHT_SENTIMENT = 0.15
REC_WEIGHT_POPULARITY = 0.10

# Expected Schema Columns
EXPECTED_COLUMNS = ["City", "Place", "Review", "Rating", "Name", "Date", "Raw_Review"]
RATING_SCALE = (1.0, 5.0)
