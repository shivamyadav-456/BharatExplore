"""Unit tests for the hybrid recommendation engine."""

import pytest
import pandas as pd
from src.recommendation_engine import TravelRecommender


@pytest.fixture
def sample_destination_catalog():
    return pd.DataFrame([
        {
            "City": "Agra",
            "Place": "Taj Mahal",
            "avg_rating": 4.8,
            "bayesian_rating": 4.75,
            "review_count": 850,
            "avg_sentiment": 0.45,
            "sentiment_summary": "Highly Positive",
            "profile_text": "ancient marble wonder monument history heritage magnificent architecture",
        },
        {
            "City": "Goa",
            "Place": "Baga Beach",
            "avg_rating": 4.3,
            "bayesian_rating": 4.25,
            "review_count": 620,
            "avg_sentiment": 0.28,
            "sentiment_summary": "Moderately Positive",
            "profile_text": "sunset beach waves sand coastal night market coastal sea relaxing",
        },
        {
            "City": "Mumbai",
            "Place": "Gateway of India",
            "avg_rating": 4.5,
            "bayesian_rating": 4.45,
            "review_count": 780,
            "avg_sentiment": 0.35,
            "sentiment_summary": "Highly Positive",
            "profile_text": "historic arch monument sea harbor boats british architecture heritage",
        },
    ])


def test_recommender_initialization(sample_destination_catalog):
    rec = TravelRecommender(sample_destination_catalog)
    assert rec.tfidf_matrix.shape[0] == 3


def test_recommender_query_matching(sample_destination_catalog):
    rec = TravelRecommender(sample_destination_catalog)
    results = rec.recommend(
        city="All Cities",
        interests=["Beaches & Coastal"],
        min_rating=4.0,
        min_reviews=10,
        top_n=2
    )
    assert len(results) > 0
    assert results.iloc[0]["Place"] == "Baga Beach"


def test_recommender_empty_results(sample_destination_catalog):
    rec = TravelRecommender(sample_destination_catalog)
    results = rec.recommend(
        city="Chennai",  # Non-existent city in fixture
        min_rating=4.0,
        min_reviews=10
    )
    assert len(results) == 0
