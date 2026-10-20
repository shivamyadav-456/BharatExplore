"""Sentiment analysis module for tourist reviews in BharatExplore.

Implements a robust, self-contained, offline-compatible VADER-style sentiment
scoring engine that handles valence shifters, negations, booster words, and
computes continuous compound scores [-1.0, 1.0] and discrete sentiment labels.
"""

from functools import lru_cache
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils import clean_text_for_nlp, get_logger

logger = get_logger("SentimentAnalysis")

# Core Sentiment Valence Lexicon (VADER-compatible curated weights for travel & hospitality)
LEXICON = {
    # Strong Positive (+3.0 to +4.0)
    "excellent": 3.4, "exceptional": 3.8, "outstanding": 3.7, "superb": 3.5,
    "breathtaking": 3.8, "magnificent": 3.6, "spectacular": 3.7, "awesome": 3.2,
    "wonderful": 3.2, "fantastic": 3.4, "splendid": 3.2, "divine": 3.3,
    "masterpiece": 3.6, "mesmerizing": 3.7, "marvelous": 3.4, "paradise": 3.5,
    "perfection": 3.6, "blissful": 3.5, "unforgettable": 3.4, "flawless": 3.5,
    # Positive (+1.5 to +2.9)
    "good": 1.9, "great": 2.7, "beautiful": 2.8, "nice": 1.6, "clean": 2.0,
    "peaceful": 2.5, "scenic": 2.6, "enjoyed": 2.1, "pleasant": 2.2, "serene": 2.6,
    "calm": 2.0, "safe": 2.1, "fun": 2.3, "charming": 2.4, "picturesque": 2.7,
    "impressive": 2.5, "relaxing": 2.4, "friendly": 2.2, "helpful": 2.1,
    "loved": 2.9, "worth": 2.1, "historical": 1.5, "lush": 2.2, "green": 1.6,
    "must-visit": 3.0, "refreshing": 2.4, "vibrant": 2.3, "gem": 2.7,
    # Weak Positive (+0.8 to +1.4)
    "decent": 1.2, "okay": 0.8, "fine": 1.0, "fair": 0.9, "adequate": 1.0,
    "average": 0.5, "acceptable": 1.1,
    # Weak Negative (-0.8 to -1.5)
    "crowded": -1.2, "noisy": -1.3, "costly": -1.4, "expensive": -1.4,
    "commercialized": -1.3, "hectic": -1.2, "rush": -0.8, "steep": -0.9,
    # Moderate Negative (-1.6 to -2.5)
    "dirty": -2.2, "unclean": -2.1, "rude": -2.4, "poor": -2.1, "bad": -2.3,
    "disappointing": -2.4, "lacking": -1.8, "unpleasant": -2.2, "unsafe": -2.5,
    "boring": -1.9, "overrated": -2.2, "scam": -2.8, "loot": -2.6, "cheat": -2.7,
    # Strong Negative (-2.6 to -4.0)
    "terrible": -3.4, "horrible": -3.5, "awful": -3.3, "disaster": -3.4,
    "pathetic": -3.3, "worst": -3.6, "disgusting": -3.5, "horrendous": -3.6,
    "waste": -2.9, "nightmare": -3.7, "miserable": -3.2,
}

# Negations that invert following valence
NEGATIONS = {
    "not", "no", "never", "hardly", "barely", "scarcely", "without", "rarely",
    "neither", "nowhere", "isnt", "wasnt", "werent", "wont", "dont", "cant",
    "couldnt", "shouldnt"
}

# Booster words that magnify or diminish following valence
BOOSTERS = {
    "very": 0.35, "extremely": 0.45, "absolutely": 0.40, "completely": 0.35,
    "totally": 0.35, "highly": 0.30, "really": 0.30, "super": 0.35,
    "especially": 0.25, "incredibly": 0.40, "hugely": 0.35,
    # Diminishers
    "slightly": -0.20, "somewhat": -0.20, "little": -0.15, "barely": -0.25,
}


@lru_cache(maxsize=512)
def score_text_sentiment(text: Optional[str]) -> Dict[str, Any]:
    """
    Computes VADER-style compound sentiment score [-1.0, 1.0] and sentiment category.
    Returns: {
        'compound': float,
        'label': 'Positive' | 'Neutral' | 'Negative',
        'pos_score': float,
        'neg_score': float
    }
    """
    if not text or not isinstance(text, str):
        return {"compound": 0.0, "label": "Neutral", "pos_score": 0.0, "neg_score": 0.0}

    cleaned = clean_text_for_nlp(text)
    words = cleaned.split()
    if not words:
        return {"compound": 0.0, "label": "Neutral", "pos_score": 0.0, "neg_score": 0.0}

    valences = []
    i = 0
    while i < len(words):
        word = words[i]
        # Base valence lookup
        val = LEXICON.get(word, 0.0)

        if val != 0.0:
            # Check preceding context (up to 3 words back) for negations and boosters
            multiplier = 1.0
            is_negated = False
            for j in range(max(0, i - 3), i):
                prev_word = words[j]
                if prev_word in NEGATIONS:
                    is_negated = True
                elif prev_word in BOOSTERS:
                    multiplier += BOOSTERS[prev_word]

            if is_negated:
                val = -val * 0.74  # standard VADER negation damping factor
            else:
                val = val * max(0.2, multiplier)

            valences.append(val)
        i += 1

    if not valences:
        return {"compound": 0.0, "label": "Neutral", "pos_score": 0.0, "neg_score": 0.0}

    # Sum valences and normalize using VADER standard formula:
    # compound = sum_s / sqrt(sum_s^2 + alpha), alpha = 15
    sum_s = sum(valences)
    alpha = 15.0
    compound = sum_s / math.sqrt((sum_s * sum_s) + alpha)
    compound = round(max(-1.0, min(1.0, compound)), 3)

    pos_sum = sum(v for v in valences if v > 0)
    neg_sum = abs(sum(v for v in valences if v < 0))
    total_val = (pos_sum + neg_sum + 1e-6)

    pos_score = round(pos_sum / total_val, 3)
    neg_score = round(neg_sum / total_val, 3)

    # Classification thresholds (standard VADER standard convention)
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "compound": compound,
        "label": label,
        "pos_score": pos_score,
        "neg_score": neg_score,
    }


def aggregate_destination_sentiment(pandas_reviews_df) -> "pd.DataFrame":
    """
    Computes destination-level sentiment aggregates:
    average compound score, positive/neutral/negative proportions, and sentiment label.
    """
    import pandas as pd

    grouped = pandas_reviews_df.groupby(["City", "Place"]).agg(
        avg_sentiment=("sentiment_compound", "mean"),
        pos_ratio=("sentiment_label", lambda s: (s == "Positive").mean()),
        neg_ratio=("sentiment_label", lambda s: (s == "Negative").mean()),
        neu_ratio=("sentiment_label", lambda s: (s == "Neutral").mean()),
    ).reset_index()

    grouped["avg_sentiment"] = grouped["avg_sentiment"].round(3)
    grouped["pos_ratio"] = (grouped["pos_ratio"] * 100).round(1)
    grouped["neg_ratio"] = (grouped["neg_ratio"] * 100).round(1)
    grouped["neu_ratio"] = (grouped["neu_ratio"] * 100).round(1)

    def label_sentiment(compound: float) -> str:
        if compound >= 0.35:
            return "Highly Positive"
        elif compound >= 0.05:
            return "Moderately Positive"
        elif compound > -0.05:
            return "Neutral / Mixed"
        else:
            return "Needs Improvement / Critical"

    grouped["sentiment_summary"] = grouped["avg_sentiment"].apply(label_sentiment)
    return grouped
