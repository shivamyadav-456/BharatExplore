"""Data preprocessing and cleaning module for BharatExplore using PySpark."""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config import CLEANED_REVIEWS_PARQUET, RATING_SCALE
from src.utils import get_logger

logger = get_logger("Preprocessing")


def preprocess_reviews_spark(
    df: "DataFrame",
    output_parquet: Optional[Path] = None,
    save_parquet: bool = True
) -> Tuple["DataFrame", Dict[str, Any]]:
    """
    Cleans raw review DataFrame using PySpark:
    - Normalizes City and Place names (trims, standardizes case).
    - Removes records with missing City, Place, or Rating.
    - Validates Rating within [1.0, 5.0].
    - Anonymizes / excludes Name to eliminate PII leakage.
    - Standardizes empty dates to NULL without fabrication.
    - Removes duplicate reviews based on (City, Place, Review, Rating).
    - Computes review length metrics (character count, word count).
    - Saves result as partitioned or compressed Parquet.
    """
    from pyspark.sql import functions as F

    start_time = time.time()
    logger.info("Beginning PySpark preprocessing pipeline...")

    raw_count = df.count()
    logger.info(f"Initial raw record count: {raw_count:,}")

    # 1. Normalize City and Place
    df_clean = (
        df.withColumn("City", F.trim(F.initcap(F.col("City"))))
        .withColumn("Place", F.trim(F.initcap(F.col("Place"))))
        .filter((F.col("City").isNotNull()) & (F.col("City") != ""))
        .filter((F.col("Place").isNotNull()) & (F.col("Place") != ""))
    )

    # 2. Rating cleaning and validation
    df_clean = df_clean.withColumn(
        "Rating_Numeric",
        F.when(
            F.col("Rating").rlike(r"^[0-9]+(\.[0-9]+)?$"),
            F.col("Rating").cast("double")
        ).otherwise(None)
    ).filter(
        (F.col("Rating_Numeric").isNotNull()) &
        (F.col("Rating_Numeric") >= RATING_SCALE[0]) &
        (F.col("Rating_Numeric") <= RATING_SCALE[1])
    )

    # 3. Clean and normalize Review text
    # Ensure Review and Raw_Review are trimmed
    df_clean = (
        df_clean.withColumn("Review_Clean", F.trim(F.col("Review")))
        .withColumn("Raw_Review_Clean", F.trim(F.col("Raw_Review")))
        .filter(
            ((F.col("Review_Clean").isNotNull()) & (F.col("Review_Clean") != "")) |
            ((F.col("Raw_Review_Clean").isNotNull()) & (F.col("Raw_Review_Clean") != ""))
        )
    )

    # 4. Review length metrics
    # Prefer Raw_Review for length if available, else Review
    df_clean = df_clean.withColumn(
        "effective_text",
        F.coalesce(F.col("Raw_Review_Clean"), F.col("Review_Clean"))
    ).withColumn(
        "review_char_length",
        F.length(F.col("effective_text"))
    ).withColumn(
        "review_word_count",
        F.size(F.split(F.col("effective_text"), r"\s+"))
    )

    # 5. PII sanitization: replace Name with 'Anonymous' and leave Date as null
    df_clean = df_clean.select(
        F.col("City"),
        F.col("Place"),
        F.col("Rating_Numeric").alias("Rating"),
        F.col("Review_Clean").alias("Review"),
        F.col("Raw_Review_Clean").alias("Raw_Review"),
        F.col("review_char_length"),
        F.col("review_word_count"),
        F.lit("Verified Visitor").alias("Reviewer_Label")  # Anonymized label
    )

    # 6. Deduplication rule: drop duplicates based on City, Place, Review, Rating
    count_before_dedup = df_clean.count()
    df_clean = df_clean.dropDuplicates(["City", "Place", "Review", "Rating"])
    count_after_dedup = df_clean.count()
    duplicates_removed = count_before_dedup - count_after_dedup
    logger.info(f"Deduplication completed. Dropped {duplicates_removed:,} duplicate records.")

    # 7. Parquet export
    if save_parquet:
        import shutil
        import sys
        out_path = output_parquet or CLEANED_REVIEWS_PARQUET
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing cleaned reviews Parquet to {out_path}...")
        
        # On Windows without winutils.exe, write directly via PyArrow to avoid Hadoop permission error
        if sys.platform == "win32":
            if out_path.exists():
                if out_path.is_dir():
                    shutil.rmtree(out_path)
                else:
                    out_path.unlink()
            pdf = df_clean.toPandas()
            pdf.to_parquet(out_path, index=False, engine="pyarrow")
        else:
            try:
                df_clean.write.mode("overwrite").parquet(str(out_path))
            except Exception:
                if out_path.exists() and out_path.is_dir():
                    shutil.rmtree(out_path)
                pdf = df_clean.toPandas()
                pdf.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Cleaned reviews Parquet written successfully.")

    duration = round(time.time() - start_time, 2)
    meta = {
        "raw_record_count": raw_count,
        "cleaned_record_count": count_after_dedup,
        "records_dropped_total": raw_count - count_after_dedup,
        "duplicates_removed": duplicates_removed,
        "preprocessing_duration_seconds": duration,
    }
    logger.info(f"Preprocessing completed in {duration}s. Final rows: {count_after_dedup:,}")
    return df_clean, meta
