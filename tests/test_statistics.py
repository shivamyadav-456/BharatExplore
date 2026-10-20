"""Unit tests for sentiment scoring and inferential hypothesis testing functions."""

import pytest
import pandas as pd
import numpy as np
from src.sentiment_analysis import score_text_sentiment
from src.hypothesis_testing import (
    run_city_rating_comparison_test,
    run_review_length_correlation_test,
    run_sentiment_rating_association_test,
)


def test_sentiment_scoring_known_sentences():
    # Strong positive
    pos_res = score_text_sentiment("This place was absolutely magnificent and wonderful! Breathtaking view!")
    assert pos_res["label"] == "Positive"
    assert pos_res["compound"] > 0.5

    # Strong negative
    neg_res = score_text_sentiment("The hotel was terrible, dirty, horribly overcrowded and a complete waste of money.")
    assert neg_res["label"] == "Negative"
    assert neg_res["compound"] < -0.3

    # Neutral
    neu_res = score_text_sentiment("The ticket counter is located on the left side.")
    assert neu_res["label"] == "Neutral"


def test_hypothesis_test_correlation_synthetic():
    np.random.seed(42)
    # Synthetic negative correlation: higher rating -> lower length
    ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=100)
    lengths = [int(150 - (r * 20) + np.random.normal(0, 10)) for r in ratings]

    df = pd.DataFrame({"Rating": ratings, "review_word_count": lengths})
    res = run_review_length_correlation_test(df)
    assert "spearman_rank_correlation" in res
    assert res["spearman_rank_correlation"]["rho"] < 0
    assert "interpretation" in res


def test_hypothesis_test_sentiment_divergence_synthetic():
    # 5-star with high sentiment, 1-star with low sentiment
    df = pd.DataFrame({
        "Rating": [5.0] * 50 + [1.0] * 50,
        "sentiment_compound": list(np.random.normal(0.6, 0.1, 50)) + list(np.random.normal(-0.4, 0.1, 50))
    })
    res = run_sentiment_rating_association_test(df)
    assert res["decision"] == "Reject H0"
    assert res["effect_size"]["value"] > 0.8
