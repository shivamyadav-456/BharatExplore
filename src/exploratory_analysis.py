"""Exploratory Data Analysis and statistical aggregation using PySpark."""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config import (
    ANALYSIS_SUMMARY_REPORT,
    CITY_STATS_PARQUET,
    DESTINATION_PROFILES_PARQUET,
    MIN_REVIEWS_DEFAULT,
)
from src.utils import calculate_bayesian_rating, get_logger, save_json_report

logger = get_logger("ExploratoryAnalysis")


def run_exploratory_analysis_spark(
    df_clean: "DataFrame",
    min_reviews_threshold: int = MIN_REVIEWS_DEFAULT,
    save_parquet: bool = True
) -> Tuple["DataFrame", "DataFrame", Dict[str, Any]]:
    """
    Computes distributed aggregations on cleaned reviews using PySpark:
    1. Overall rating distribution, mean, and variance.
    2. City-wise review volume, destination count, and mean ratings.
    3. Destination-wise review count, average rating, variance, and Bayesian score.
    4. Aggregated review text snippet profiles for TF-IDF content extraction.
    5. Review length correlations by star rating.
    """
    from pyspark.sql import functions as F

    start_time = time.time()
    logger.info("Starting PySpark Exploratory Data Analysis & Aggregations...")

    # Cache df_clean for repeated distributed operations
    df_clean.cache()

    total_reviews = df_clean.count()

    # 1. Overall Rating Metrics
    rating_stats = df_clean.agg(
        F.mean("Rating").alias("mean_rating"),
        F.variance("Rating").alias("var_rating"),
        F.stddev("Rating").alias("std_rating"),
        F.min("Rating").alias("min_rating"),
        F.max("Rating").alias("max_rating")
    ).collect()[0]

    global_mean_rating = float(rating_stats["mean_rating"] or 0.0)
    global_var_rating = float(rating_stats["var_rating"] or 0.0)
    global_std_rating = float(rating_stats["std_rating"] or 0.0)

    # Rating Frequency Distribution
    rating_dist_rows = (
        df_clean.groupBy("Rating")
        .count()
        .orderBy("Rating")
        .collect()
    )
    rating_distribution = {
        str(row["Rating"]): {
            "count": int(row["count"]),
            "percentage": round((row["count"] / total_reviews) * 100, 2)
        }
        for row in rating_dist_rows
    }

    # Review Length Metrics
    len_stats = df_clean.agg(
        F.mean("review_word_count").alias("avg_word_count"),
        F.mean("review_char_length").alias("avg_char_length")
    ).collect()[0]
    avg_word_count = round(float(len_stats["avg_word_count"] or 0.0), 2)
    avg_char_length = round(float(len_stats["avg_char_length"] or 0.0), 2)

    # Word count by star rating
    rating_len_rows = (
        df_clean.groupBy("Rating")
        .agg(F.mean("review_word_count").alias("avg_words"))
        .orderBy("Rating")
        .collect()
    )
    word_count_by_rating = {
        str(row["Rating"]): round(float(row["avg_words"]), 1)
        for row in rating_len_rows
    }

    # 2. City-Level Aggregates
    logger.info("Computing city-level statistics...")
    city_df = (
        df_clean.groupBy("City")
        .agg(
            F.count("Rating").alias("total_reviews"),
            F.countDistinct("Place").alias("destination_count"),
            F.round(F.mean("Rating"), 2).alias("avg_rating"),
            F.round(F.variance("Rating"), 3).alias("rating_variance"),
            F.round(F.stddev("Rating"), 3).alias("rating_stddev")
        )
        .orderBy(F.desc("total_reviews"))
    )

    if save_parquet:
        import shutil
        import sys
        CITY_STATS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            if CITY_STATS_PARQUET.exists():
                if CITY_STATS_PARQUET.is_dir():
                    shutil.rmtree(CITY_STATS_PARQUET)
                else:
                    CITY_STATS_PARQUET.unlink()
            city_df.toPandas().to_parquet(CITY_STATS_PARQUET, index=False, engine="pyarrow")
        else:
            try:
                city_df.write.mode("overwrite").parquet(str(CITY_STATS_PARQUET))
            except Exception:
                if CITY_STATS_PARQUET.exists() and CITY_STATS_PARQUET.is_dir():
                    shutil.rmtree(CITY_STATS_PARQUET)
                city_df.toPandas().to_parquet(CITY_STATS_PARQUET, index=False, engine="pyarrow")
        logger.info(f"City statistics written to {CITY_STATS_PARQUET}")

    # 3. Destination-Level Aggregates & Profile Building
    logger.info("Computing destination profiles and Bayesian confidence ratings...")

    # Group by City and Place
    # Collect sample keywords / text snippets (up to 15 distinct review snippets) for TF-IDF
    dest_df = (
        df_clean.groupBy("City", "Place")
        .agg(
            F.count("Rating").alias("review_count"),
            F.round(F.mean("Rating"), 3).alias("avg_rating"),
            F.round(F.stddev("Rating"), 3).alias("rating_stddev"),
            # Concatenate review texts for content profile (take first 20 reviews concatenated)
            F.slice(F.collect_list("Review"), 1, 20).alias("review_samples")
        )
    )

    # Native Spark SQL Bayesian rating calculation
    v_col = F.col("review_count").cast("double")
    m_val = float(min_reviews_threshold)
    r_col = F.col("avg_rating").cast("double")
    c_val = float(global_mean_rating)

    bayesian_calc = F.round((v_col / (v_col + F.lit(m_val))) * r_col + (F.lit(m_val) / (v_col + F.lit(m_val))) * F.lit(c_val), 3)

    dest_df = (
        dest_df.withColumn("bayesian_rating", bayesian_calc)
        .withColumn("profile_text", F.concat_ws(" ", F.col("review_samples")))
        .drop("review_samples")
    )

    if save_parquet:
        import shutil
        import sys
        DESTINATION_PROFILES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            if DESTINATION_PROFILES_PARQUET.exists():
                if DESTINATION_PROFILES_PARQUET.is_dir():
                    shutil.rmtree(DESTINATION_PROFILES_PARQUET)
                else:
                    DESTINATION_PROFILES_PARQUET.unlink()
            dest_df.toPandas().to_parquet(DESTINATION_PROFILES_PARQUET, index=False, engine="pyarrow")
        else:
            try:
                dest_df.write.mode("overwrite").parquet(str(DESTINATION_PROFILES_PARQUET))
            except Exception:
                if DESTINATION_PROFILES_PARQUET.exists() and DESTINATION_PROFILES_PARQUET.is_dir():
                    shutil.rmtree(DESTINATION_PROFILES_PARQUET)
                dest_df.toPandas().to_parquet(DESTINATION_PROFILES_PARQUET, index=False, engine="pyarrow")
        logger.info(f"Destination profiles written to {DESTINATION_PROFILES_PARQUET}")

    # Top and Bottom Destinations for summary
    top_destinations = (
        dest_df.filter(F.col("review_count") >= min_reviews_threshold)
        .orderBy(F.desc("bayesian_rating"))
        .limit(10)
        .toPandas()
        .to_dict(orient="records")
    )

    top_cities = (
        city_df.limit(10)
        .toPandas()
        .to_dict(orient="records")
    )

    # Release cache
    df_clean.unpersist()

    duration = round(time.time() - start_time, 2)

    summary = {
        "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": duration,
        "global_metrics": {
            "total_reviews": total_reviews,
            "global_mean_rating": round(global_mean_rating, 3),
            "global_variance_rating": round(global_var_rating, 3),
            "global_stddev_rating": round(global_std_rating, 3),
            "avg_words_per_review": avg_word_count,
            "avg_chars_per_review": avg_char_length,
            "min_reviews_threshold": min_reviews_threshold,
        },
        "rating_distribution": rating_distribution,
        "word_count_by_rating": word_count_by_rating,
        "top_cities_summary": top_cities,
        "top_destinations_summary": top_destinations,
    }

    save_json_report(summary, ANALYSIS_SUMMARY_REPORT)
    logger.info(f"Analysis summary report saved to {ANALYSIS_SUMMARY_REPORT}")

    return city_df, dest_df, summary
