"""Data access layer for Streamlit dashboard and analytics reporting."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.config import (
    ANALYSIS_SUMMARY_REPORT,
    CITY_STATS_PARQUET,
    DATA_QUALITY_REPORT,
    DESTINATION_PROFILES_PARQUET,
    MODEL_EVALUATION_REPORT,
    SENTIMENT_SAMPLE_PARQUET,
)
from src.recommendation_engine import TravelRecommender
from src.utils import get_logger, load_json_report

logger = get_logger("DataAccess")


def load_destination_profiles() -> Optional[pd.DataFrame]:
    """Loads destination profile analytics Parquet table."""
    if not DESTINATION_PROFILES_PARQUET.exists():
        logger.warning(f"Destination profiles parquet not found at {DESTINATION_PROFILES_PARQUET}")
        return None
    try:
        return pd.read_parquet(DESTINATION_PROFILES_PARQUET)
    except Exception as e:
        logger.error(f"Error reading destination profiles parquet: {e}")
        return None


def load_city_stats() -> Optional[pd.DataFrame]:
    """Loads city-level statistical summary Parquet table."""
    if not CITY_STATS_PARQUET.exists():
        logger.warning(f"City stats parquet not found at {CITY_STATS_PARQUET}")
        return None
    try:
        return pd.read_parquet(CITY_STATS_PARQUET)
    except Exception as e:
        logger.error(f"Error reading city stats parquet: {e}")
        return None


def load_sentiment_sample() -> Optional[pd.DataFrame]:
    """Loads sentiment sample review dataset."""
    if not SENTIMENT_SAMPLE_PARQUET.exists():
        return None
    try:
        return pd.read_parquet(SENTIMENT_SAMPLE_PARQUET)
    except Exception as e:
        logger.error(f"Error reading sentiment sample parquet: {e}")
        return None


def load_quality_report() -> Optional[Dict[str, Any]]:
    """Loads data quality and validation JSON report."""
    return load_json_report(DATA_QUALITY_REPORT)


def load_analysis_summary() -> Optional[Dict[str, Any]]:
    """Loads exploratory analysis summary JSON report."""
    return load_json_report(ANALYSIS_SUMMARY_REPORT)


def load_model_evaluation() -> Optional[Dict[str, Any]]:
    """Loads recommendation model evaluation JSON report."""
    return load_json_report(MODEL_EVALUATION_REPORT)


_CACHED_RECOMMENDER: Optional[TravelRecommender] = None


def get_recommender() -> Optional[TravelRecommender]:
    """Returns singleton cached instance of TravelRecommender."""
    global _CACHED_RECOMMENDER
    if _CACHED_RECOMMENDER is not None:
        return _CACHED_RECOMMENDER

    dest_df = load_destination_profiles()
    if dest_df is None or len(dest_df) == 0:
        return None

    _CACHED_RECOMMENDER = TravelRecommender(dest_df)
    return _CACHED_RECOMMENDER
