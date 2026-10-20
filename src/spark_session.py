"""Spark session builder and manager with JVM 17/21/23 compatibility for Windows."""

import os
import sys
from typing import Optional
from src.config import (
    BASE_DIR,
    JVM_ADD_OPENS,
    SPARK_APP_NAME,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_MASTER,
)
from src.utils import get_logger

logger = get_logger("SparkSessionManager")


def get_spark_session(app_name: str = SPARK_APP_NAME) -> "SparkSession":
    """
    Creates or returns an active PySpark SparkSession configured for local execution.
    Applies modern JVM options (--add-opens) necessary for Java 17/21/23 on Windows.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as e:
        logger.error("pyspark is not installed in the current environment.")
        raise ImportError(
            "PySpark is missing. Please run: pip install -r requirements.txt"
        ) from e

    # Ensure warehouse directory exists in local workspace
    warehouse_dir = str(BASE_DIR / "data" / "spark_warehouse")
    os.makedirs(warehouse_dir, exist_ok=True)

    # Set JAVA_TOOL_OPTIONS if not already configured with add-opens
    existing_opts = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "--add-opens" not in existing_opts:
        os.environ["JAVA_TOOL_OPTIONS"] = f"{existing_opts} {JVM_ADD_OPENS}".strip()

    logger.info("Initializing SparkSession...")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.driver.extraJavaOptions", JVM_ADD_OPENS)
        .config("spark.executor.extraJavaOptions", JVM_ADD_OPENS)
    )

    try:
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f"SparkSession created successfully (Version: {spark.version}).")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize SparkSession: {e}")
        raise RuntimeError(
            f"Spark failed to start. Ensure Java 11/17/21/23 is installed and JAVA_HOME is set. Error: {e}"
        ) from e


def stop_spark_session(spark: Optional["SparkSession"] = None) -> None:
    """Stops the active SparkSession."""
    if spark is not None:
        try:
            spark.stop()
            logger.info("SparkSession stopped cleanly.")
        except Exception as e:
            logger.warning(f"Error stopping SparkSession: {e}")
