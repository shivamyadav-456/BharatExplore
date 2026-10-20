"""Data ingestion module for BharatExplore.

Handles ZIP extraction, CSV location detection, explicit schema definition,
and safe PySpark ingestion with metadata logging.
"""

import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config import (
    DEFAULT_RAW_CSV,
    DEFAULT_RAW_ZIP,
    EXPECTED_COLUMNS,
    FALLBACK_RAW_ZIP,
    RAW_DATA_DIR,
)
from src.utils import get_logger

logger = get_logger("Ingest")


def get_dataset_source_path() -> Path:
    """
    Locates the raw dataset CSV or ZIP. If only ZIP exists, extracts Review_db.csv.
    Raises FileNotFoundError with clear Kaggle instructions if missing.
    """
    # 1. Direct CSV check
    if DEFAULT_RAW_CSV.exists() and DEFAULT_RAW_CSV.stat().st_size > 0:
        logger.info(f"Located direct CSV at: {DEFAULT_RAW_CSV}")
        return DEFAULT_RAW_CSV

    # 2. Check local data/raw/archive.zip
    target_zip = None
    if DEFAULT_RAW_ZIP.exists() and DEFAULT_RAW_ZIP.stat().st_size > 0:
        target_zip = DEFAULT_RAW_ZIP
    elif FALLBACK_RAW_ZIP.exists() and FALLBACK_RAW_ZIP.stat().st_size > 0:
        target_zip = FALLBACK_RAW_ZIP

    if target_zip:
        logger.info(f"Extracting Review_db.csv from archive: {target_zip}...")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target_zip, "r") as zf:
            file_names = zf.namelist()
            csv_candidates = [f for f in file_names if f.lower().endswith(".csv")]
            if not csv_candidates:
                raise ValueError(f"No CSV file found inside archive: {target_zip}")
            
            # Extract to data/raw/
            extracted_filename = csv_candidates[0]
            zf.extract(extracted_filename, path=RAW_DATA_DIR)
            extracted_path = RAW_DATA_DIR / extracted_filename
            
            # If named differently, rename to Review_db.csv
            if extracted_path != DEFAULT_RAW_CSV:
                if DEFAULT_RAW_CSV.exists():
                    DEFAULT_RAW_CSV.unlink()
                extracted_path.rename(DEFAULT_RAW_CSV)
            
            logger.info(f"Successfully extracted dataset to: {DEFAULT_RAW_CSV} ({DEFAULT_RAW_CSV.stat().st_size / (1024*1024):.2f} MB)")
            return DEFAULT_RAW_CSV

    # 3. Missing dataset error
    msg = (
        "Dataset not found!\n"
        "Please download the dataset from Kaggle:\n"
        "https://www.kaggle.com/datasets/ritvik1909/indian-places-to-visit-reviews-data\n"
        f"and place 'archive.zip' or 'Review_db.csv' in '{RAW_DATA_DIR}'."
    )
    logger.error(msg)
    raise FileNotFoundError(msg)


def get_pyspark_schema():
    """Returns the explicit PySpark StructType schema for Review_db.csv."""
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    return StructType([
        StructField("City", StringType(), True),
        StructField("Place", StringType(), True),
        StructField("Review", StringType(), True),
        StructField("Rating", StringType(), True),  # Read as String first to safely handle malformed ratings
        StructField("Name", StringType(), True),
        StructField("Date", StringType(), True),
        StructField("Raw_Review", StringType(), True),
    ])


def ingest_data_spark(spark: "SparkSession", csv_path: Optional[Path] = None) -> Tuple["DataFrame", Dict[str, Any]]:
    """
    Reads the raw CSV dataset into a PySpark DataFrame using explicit schema,
    multi-line parsing, and logs timing and record counts.
    """
    if csv_path is None:
        csv_path = get_dataset_source_path()

    start_time = time.time()
    file_size_mb = round(csv_path.stat().st_size / (1024 * 1024), 2)
    logger.info(f"Starting PySpark ingestion of {csv_path} (Size: {file_size_mb} MB)...")

    schema = get_pyspark_schema()

    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("multiLine", "true")
        .option("escape", "\"")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .load(str(csv_path))
    )

    ingest_duration = round(time.time() - start_time, 2)
    logger.info(f"PySpark DataFrame initialized in {ingest_duration}s.")

    metadata = {
        "source_file": str(csv_path.resolve()),
        "file_size_mb": file_size_mb,
        "ingest_duration_seconds": ingest_duration,
        "columns": EXPECTED_COLUMNS,
    }
    return df, metadata
