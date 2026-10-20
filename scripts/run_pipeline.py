"""Master end-to-end Big Data pipeline for BharatExplore.

Executes:
1. Data Ingestion & Data Quality Profiling (PySpark)
2. Data Preprocessing, Cleaning & Anonymization (PySpark)
3. Exploratory Data Analysis & Distributed Aggregations (PySpark)
4. Sentiment Analysis & Destination Valence Aggregation
5. Recommendation System Indexing & Offline Evaluation
6. Statistical Hypothesis Testing
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLEANED_REVIEWS_PARQUET,
    DESTINATION_PROFILES_PARQUET,
    DEV_MODE_SAMPLE_FRACTION,
    MIN_REVIEWS_DEFAULT,
    REPORTS_DIR,
    SENTIMENT_SAMPLE_PARQUET,
)
from src.evaluation import evaluate_recommendation_system
from src.exploratory_analysis import run_exploratory_analysis_spark
from src.hypothesis_testing import (
    run_city_rating_comparison_test,
    run_review_length_correlation_test,
    run_sentiment_rating_association_test,
)
from src.ingest import get_dataset_source_path, ingest_data_spark
from src.preprocessing import preprocess_reviews_spark
from src.recommendation_engine import TravelRecommender
from src.sentiment_analysis import (
    aggregate_destination_sentiment,
    score_text_sentiment,
)
from src.spark_session import get_spark_session, stop_spark_session
from src.utils import get_logger, save_json_report

logger = get_logger("MasterPipeline")


def run_pipeline(
    sample_fraction: float = 1.0,
    dev_mode: bool = False,
    min_reviews: int = MIN_REVIEWS_DEFAULT
) -> None:
    """Executes the complete BharatExplore Big Data processing pipeline."""
    start_total_time = time.time()
    logger.info("=" * 65)
    logger.info("  BHARATEXPLORE: BIG DATA TRAVEL RECOMMENDATION PIPELINE  ")
    logger.info("=" * 65)

    if dev_mode:
        sample_fraction = DEV_MODE_SAMPLE_FRACTION
        logger.warning(f"DEV MODE ACTIVATED: Processing {sample_fraction * 100:.1f}% sample of dataset.")

    spark = get_spark_session("BharatExplore_Master_Pipeline")

    try:
        # 1. Ingestion
        logger.info("\n>>> STAGE 1: INGESTION & DATA VALIDATION")
        csv_path = get_dataset_source_path()
        df_raw, ingest_meta = ingest_data_spark(spark, csv_path)

        if sample_fraction < 1.0:
            df_raw = df_raw.sample(withReplacement=False, fraction=sample_fraction, seed=42)

        # 2. Preprocessing & Deduplication
        logger.info("\n>>> STAGE 2: PREPROCESSING & DEDUPLICATION (PYSPARK)")
        df_clean, prep_meta = preprocess_reviews_spark(df_raw, save_parquet=True)

        # 3. Exploratory Analysis & Profiling
        logger.info("\n>>> STAGE 3: DISTRIBUTED AGGREGATIONS & PROFILING (PYSPARK)")
        city_df, dest_df, eda_summary = run_exploratory_analysis_spark(
            df_clean, min_reviews_threshold=min_reviews, save_parquet=True
        )

        # 4. Sentiment Analysis
        logger.info("\n>>> STAGE 4: SENTIMENT ANALYSIS & DESTINATION VALENCE AGGREGATION")
        # Sample reviews for sentiment processing
        # Taking up to 60,000 reviews for high-fidelity statistical sentiment analysis
        clean_cnt = df_clean.count()
        sample_size = min(25000, clean_cnt)
        logger.info(f"Collecting {sample_size:,} reviews for sentiment analysis...")
        pdf_sample = df_clean.limit(sample_size).select(
            "City", "Place", "Rating", "Review", "review_word_count"
        ).toPandas()

        logger.info(f"Scoring sentiment on {len(pdf_sample):,} review texts...")
        sentiment_scores = [score_text_sentiment(t) for t in pdf_sample["Review"]]

        pdf_sample["sentiment_compound"] = [s["compound"] for s in sentiment_scores]
        pdf_sample["sentiment_label"] = [s["label"] for s in sentiment_scores]

        # Save sentiment sample parquet
        SENTIMENT_SAMPLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        pdf_sample.to_parquet(SENTIMENT_SAMPLE_PARQUET, index=False)
        logger.info(f"Sentiment review sample saved to {SENTIMENT_SAMPLE_PARQUET}")

        # Compute destination sentiment aggregates
        dest_sentiment_df = aggregate_destination_sentiment(pdf_sample)

        # Load and merge destination profiles with sentiment
        import pandas as pd
        dest_profiles = pd.read_parquet(DESTINATION_PROFILES_PARQUET)
        dest_profiles = dest_profiles.merge(
            dest_sentiment_df, on=["City", "Place"], how="left"
        )
        dest_profiles["avg_sentiment"] = dest_profiles["avg_sentiment"].fillna(0.15)
        dest_profiles["sentiment_summary"] = dest_profiles["sentiment_summary"].fillna("Positive Feedback")
        dest_profiles.to_parquet(DESTINATION_PROFILES_PARQUET, index=False)
        logger.info(f"Updated destination profiles with sentiment in {DESTINATION_PROFILES_PARQUET}")

        # 5. Recommendation System Indexing & Evaluation
        logger.info("\n>>> STAGE 5: RECOMMENDATION ENGINE EVALUATION")
        recommender = TravelRecommender(dest_profiles)
        eval_report = evaluate_recommendation_system(recommender, top_k=10, min_reviews=min_reviews)

        # 6. Hypothesis Testing
        logger.info("\n>>> STAGE 6: STATISTICAL HYPOTHESIS TESTING")
        ht_city = run_city_rating_comparison_test(pdf_sample, top_n_cities=5)
        ht_length = run_review_length_correlation_test(pdf_sample)
        ht_sent = run_sentiment_rating_association_test(pdf_sample)

        ht_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tests": {
                "city_rating_comparison": ht_city,
                "review_length_correlation": ht_length,
                "sentiment_rating_association": ht_sent,
            }
        }
        ht_report_path = REPORTS_DIR / "hypothesis_testing_report.json"
        save_json_report(ht_report, ht_report_path)
        logger.info(f"Hypothesis testing report saved to {ht_report_path}")

        total_elapsed = round(time.time() - start_total_time, 2)
        logger.info("=" * 65)
        logger.info(f"  PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed}s  ")
        logger.info(f"  Processed Reviews : {prep_meta['cleaned_record_count']:,}")
        logger.info(f"  Unique Places     : {len(dest_profiles):,}")
        logger.info(f"  Analytics Outputs : {DESTINATION_PROFILES_PARQUET}")
        logger.info("=" * 65)

    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BharatExplore Big Data pipeline.")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of raw dataset to process (0.0 to 1.0)")
    parser.add_argument("--dev", action="store_true", help="Run quick development mode on 5% sample")
    parser.add_argument("--min-reviews", type=int, default=MIN_REVIEWS_DEFAULT, help="Minimum review threshold")
    args = parser.parse_args()

    run_pipeline(
        sample_fraction=args.sample,
        dev_mode=args.dev,
        min_reviews=args.min_reviews,
    )
