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

import streamlit as st

logger = get_logger("DataAccess")


@st.cache_data(show_spinner=False)
def load_destination_profiles() -> Optional[pd.DataFrame]:
    """Loads destination profile analytics Parquet table (cached)."""
    if not DESTINATION_PROFILES_PARQUET.exists():
        logger.warning(f"Destination profiles parquet not found at {DESTINATION_PROFILES_PARQUET}")
        return None
    try:
        return pd.read_parquet(DESTINATION_PROFILES_PARQUET)
    except Exception as e:
        logger.error(f"Error reading destination profiles parquet: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_city_stats() -> Optional[pd.DataFrame]:
    """Loads city-level statistical summary Parquet table (cached)."""
    if not CITY_STATS_PARQUET.exists():
        logger.warning(f"City stats parquet not found at {CITY_STATS_PARQUET}")
        return None
    try:
        return pd.read_parquet(CITY_STATS_PARQUET)
    except Exception as e:
        logger.error(f"Error reading city stats parquet: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_sentiment_sample() -> Optional[pd.DataFrame]:
    """Loads sentiment sample review dataset (cached)."""
    if not SENTIMENT_SAMPLE_PARQUET.exists():
        return None
    try:
        return pd.read_parquet(SENTIMENT_SAMPLE_PARQUET)
    except Exception as e:
        logger.error(f"Error reading sentiment sample parquet: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_quality_report() -> Optional[Dict[str, Any]]:
    """Loads data quality and validation JSON report (cached)."""
    return load_json_report(DATA_QUALITY_REPORT)


@st.cache_data(show_spinner=False)
def load_analysis_summary() -> Optional[Dict[str, Any]]:
    """Loads exploratory analysis summary JSON report (cached)."""
    return load_json_report(ANALYSIS_SUMMARY_REPORT)


@st.cache_data(show_spinner=False)
def load_model_evaluation() -> Optional[Dict[str, Any]]:
    """Loads recommendation model evaluation JSON report (cached)."""
    return load_json_report(MODEL_EVALUATION_REPORT)


@st.cache_data(show_spinner=False)
def get_unique_review_cities() -> list[str]:
    """Returns sorted unique city names for the review analytics sample."""
    df = load_sentiment_sample()
    if df is None:
        return ["All Cities"]
    return ["All Cities"] + sorted(df["City"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def get_destination_cities() -> list[str]:
    """Returns sorted unique city names for destination profiles."""
    df = load_destination_profiles()
    if df is None:
        return ["All Cities"]
    return ["All Cities"] + sorted(df["City"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def get_cached_city_rating_test(top_n_cities: int = 5, alpha: float = 0.05) -> Dict[str, Any]:
    """Cached execution of Hypothesis Test 1 (ANOVA / Kruskal-Wallis)."""
    from src.hypothesis_testing import run_city_rating_comparison_test
    df = load_sentiment_sample()
    if df is None:
        return {}
    return run_city_rating_comparison_test(df, top_n_cities=top_n_cities, alpha=alpha)


@st.cache_data(show_spinner=False)
def get_cached_correlation_test(alpha: float = 0.05) -> Dict[str, Any]:
    """Cached execution of Hypothesis Test 2 (Review Length vs Star Rating)."""
    from src.hypothesis_testing import run_review_length_correlation_test
    df = load_sentiment_sample()
    if df is None:
        return {}
    return run_review_length_correlation_test(df, alpha=alpha)


@st.cache_data(show_spinner=False)
def get_cached_sentiment_association_test(alpha: float = 0.05) -> Dict[str, Any]:
    """Cached execution of Hypothesis Test 3 (5-Star vs 1-Star Sentiment Divergence)."""
    from src.hypothesis_testing import run_sentiment_rating_association_test
    df = load_sentiment_sample()
    if df is None:
        return {}
    return run_sentiment_rating_association_test(df, alpha=alpha)


@st.cache_resource(show_spinner="Initializing travel recommendation model...")
def get_recommender() -> Optional[TravelRecommender]:
    """Returns singleton cached instance of TravelRecommender."""
    dest_df = load_destination_profiles()
    if dest_df is None or len(dest_df) == 0:
        return None
    return TravelRecommender(dest_df)
