"""Unit tests for data preprocessing and cleaning functions."""

import pytest
import pandas as pd
from src.utils import calculate_bayesian_rating, clean_text_for_nlp


def test_clean_text_for_nlp():
    raw = "  Beautiful place!! Visit http://example.com/tour   for more details... \n\nGreat view! "
    cleaned = clean_text_for_nlp(raw)
    assert "http" not in cleaned
    assert "beautiful place" in cleaned
    assert "great view" in cleaned
    assert "\n" not in cleaned


def test_clean_text_empty_input():
    assert clean_text_for_nlp(None) == ""
    assert clean_text_for_nlp("") == ""
    assert clean_text_for_nlp(12345) == ""


def test_bayesian_rating_calculation():
    # Scenario 1: Destination with few reviews (1 review of 5.0) vs global mean of 4.0
    r1 = calculate_bayesian_rating(rating=5.0, review_count=1, mean_rating=4.0, min_reviews_threshold=30)
    # The score should be pulled heavily towards the global mean of 4.0
    assert r1 < 4.1

    # Scenario 2: Destination with many reviews (500 reviews of 4.8) vs global mean of 4.0
    r2 = calculate_bayesian_rating(rating=4.8, review_count=500, mean_rating=4.0, min_reviews_threshold=30)
    # The score should remain close to 4.8
    assert r2 > 4.7
    assert r2 > r1
