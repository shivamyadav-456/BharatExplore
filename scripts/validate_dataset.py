"""Dataset validation and profiling script for BharatExplore.

Inspects the raw dataset (archive.zip or Review_db.csv), validates schemas,
row counts, unique entities, rating distribution, null frequencies, and saves
a comprehensive machine-readable report to reports/data_quality_report.json.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_QUALITY_REPORT, EXPECTED_COLUMNS, RATING_SCALE
from src.ingest import get_dataset_source_path, ingest_data_spark
from src.spark_session import get_spark_session, stop_spark_session
from src.utils import get_logger, save_json_report

logger = get_logger("DatasetValidator")


def validate_dataset(spark=None, sample_fraction: float = 1.0) -> dict:
    """
    Performs data quality checks and schema profiling on the raw dataset.
    """
    own_spark = False
    if spark is None:
        spark = get_spark_session("BharatExplore_Validator")
        own_spark = True

    try:
        csv_path = get_dataset_source_path()
        df, ingest_meta = ingest_data_spark(spark, csv_path)

        if sample_fraction < 1.0:
            logger.info(f"Running validation on sample fraction: {sample_fraction}")
            df = df.sample(withReplacement=False, fraction=sample_fraction, seed=42)

        start_time = time.time()
        logger.info("Computing dataset statistics with PySpark...")

        from pyspark.sql import functions as F

        # Cache df temporarily for profiling
        df.cache()

        total_rows = df.count()
        logger.info(f"Total raw rows: {total_rows:,}")

        # Column existence check
        actual_cols = df.columns
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in actual_cols]

        # Null counts per column
        null_counts = {}
        for col in actual_cols:
            null_cnt = df.filter(F.col(col).isNull() | (F.trim(F.col(col)) == "")).count()
            null_counts[col] = {
                "null_or_empty_count": null_cnt,
                "null_percentage": round((null_cnt / total_rows) * 100, 2) if total_rows > 0 else 0.0,
            }

        # Distinct counts
        unique_cities = df.select("City").filter(F.col("City").isNotNull() & (F.trim(F.col("City")) != "")).distinct().count()
        unique_destinations = df.select("Place").filter(F.col("Place").isNotNull() & (F.trim(F.col("Place")) != "")).distinct().count()

        # Rating validation
        # Cast rating to double and check valid scale
        numeric_rating_df = df.withColumn(
            "Rating_Num",
            F.when(F.col("Rating").rlike(r"^[0-9]+(\.[0-9]+)?$"), F.col("Rating").cast("double")).otherwise(None)
        )
        invalid_ratings_cnt = numeric_rating_df.filter(
            F.col("Rating_Num").isNull() | (F.col("Rating_Num") < RATING_SCALE[0]) | (F.col("Rating_Num") > RATING_SCALE[1])
        ).count()

        # Rating frequency distribution
        rating_dist = (
            numeric_rating_df.filter(F.col("Rating_Num").isNotNull())
            .groupBy("Rating_Num")
            .count()
            .orderBy("Rating_Num")
            .toPandas()
        )
        rating_distribution = {
            str(row["Rating_Num"]): int(row["count"])
            for _, row in rating_dist.iterrows()
        }

        # Duplicate review patterns (identical city, place, and review text)
        duplicate_reviews_cnt = (
            df.groupBy("City", "Place", "Review")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        # Review count distribution per destination
        place_counts = (
            df.groupBy("City", "Place")
            .count()
            .withColumnRenamed("count", "review_count")
            .cache()
        )

        destinations_with_min_20 = place_counts.filter(F.col("review_count") >= 20).count()
        destinations_with_min_50 = place_counts.filter(F.col("review_count") >= 50).count()
        max_reviews_per_dest = place_counts.agg(F.max("review_count")).collect()[0][0] or 0
        median_reviews_per_dest = float(place_counts.approxQuantile("review_count", [0.5], 0.01)[0])

        # Release caches
        df.unpersist()
        place_counts.unpersist()

        duration = round(time.time() - start_time, 2)

        report = {
            "validation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profiling_duration_seconds": duration,
            "sample_fraction": sample_fraction,
            "is_sample": sample_fraction < 1.0,
            "dataset_metadata": ingest_meta,
            "metrics": {
                "total_rows": total_rows,
                "total_columns": len(actual_cols),
                "columns_present": actual_cols,
                "missing_expected_columns": missing_cols,
                "unique_cities": unique_cities,
                "unique_destinations": unique_destinations,
                "invalid_ratings_count": invalid_ratings_cnt,
                "duplicate_review_groups": duplicate_reviews_cnt,
                "destinations_with_gte_20_reviews": destinations_with_min_20,
                "destinations_with_gte_50_reviews": destinations_with_min_50,
                "max_reviews_single_destination": int(max_reviews_per_dest),
                "median_reviews_per_destination": median_reviews_per_dest,
            },
            "null_value_analysis": null_counts,
            "rating_distribution": rating_distribution,
            "data_quality_assessment": {
                "date_column_usable": null_counts.get("Date", {}).get("null_percentage", 0) < 50.0,
                "temporal_analysis_advisable": False,  # Date is known to be empty
                "pii_risk_field": "Name",
                "recommended_min_review_threshold": 20,
            }
        }

        save_json_report(report, DATA_QUALITY_REPORT)
        logger.info(f"Data quality report saved to {DATA_QUALITY_REPORT}")
        return report

    finally:
        if own_spark:
            stop_spark_session(spark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and profile BharatExplore dataset.")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of dataset to sample (0.0 to 1.0)")
    args = parser.parse_args()

    validate_dataset(sample_fraction=args.sample)
