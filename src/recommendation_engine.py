"""Hybrid Content-Based Recommendation Engine for BharatExplore.

Uses TF-IDF vectorization over destination-level review corpora combined with
Bayesian confidence-adjusted ratings, sentiment indicators, and review volume evidence.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    REC_WEIGHT_BAYESIAN_RATING,
    REC_WEIGHT_POPULARITY,
    REC_WEIGHT_SENTIMENT,
    REC_WEIGHT_SIMILARITY,
)
from src.utils import clean_text_for_nlp, get_logger

logger = get_logger("RecommendationEngine")

# Supported thematic interest tags mapped to key descriptive review tokens
CURATED_INTERESTS = {
    "Heritage & History": "history historical monument ancient heritage architecture fort palace dynasty archaeological",
    "Nature & Landscapes": "nature scenic valley hills lush green greenery waterfall mountain trees flora fauna picturesque",
    "Spiritual & Temples": "temple darshan peaceful spiritual holy divine prayers morning arti quiet calm shrine religious",
    "Beaches & Coastal": "beach sea waves sand coastal sunset sunset breeze boat water marine shore coastline",
    "Forts & Architecture": "fort walls gate stone architecture grandeur view cannons ramparts climb defense carvings",
    "Adventure & Trekking": "trek trekking trail hiking adventure climb rocks cave view outdoor thrilling camping",
    "Wildlife & Safari": "wildlife tiger safari animals sanctuary forest national park birds deer nature reserve",
    "Peace & Relaxation": "peaceful quiet serene calm relaxing meditation fresh air tranquility secluded resting",
    "Family & Picnics": "family kids children garden park picnic weekend outing fun relaxing boating play",
    "Food & Culture": "food street local stalls cuisine taste shopping bazaar handicrafts cultural folk",
}


class TravelRecommender:
    """
    Hybrid content-based travel destination recommender.
    Combines text cosine similarity, Bayesian ratings, sentiment valence, and review volume.
    """

    def __init__(
        self,
        destinations_df: pd.DataFrame,
        weight_sim: float = REC_WEIGHT_SIMILARITY,
        weight_rating: float = REC_WEIGHT_BAYESIAN_RATING,
        weight_sentiment: float = REC_WEIGHT_SENTIMENT,
        weight_pop: float = REC_WEIGHT_POPULARITY,
    ):
        self.dest_df = destinations_df.copy()
        self.w_sim = weight_sim
        self.w_rating = weight_rating
        self.w_sent = weight_sentiment
        self.w_pop = weight_pop

        # Ensure required columns exist
        if "profile_text" not in self.dest_df.columns:
            self.dest_df["profile_text"] = self.dest_df["Place"] + " " + self.dest_df["City"]

        self.dest_df["profile_text"] = self.dest_df["profile_text"].fillna("").astype(str)

        # Initialize and fit TF-IDF vectorizer
        logger.info(f"Fitting TF-IDF on {len(self.dest_df)} destination profiles...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.dest_df["profile_text"])
        logger.info("TF-IDF matrix built successfully.")

        # Precompute normalized base metrics
        self._precompute_normalized_features()

    def _precompute_normalized_features(self):
        """Normalizes Bayesian rating, sentiment, and popularity into [0.0, 1.0]."""
        # 1. Rating normalization (scale 1-5 to 0-1)
        if "bayesian_rating" in self.dest_df.columns:
            r = self.dest_df["bayesian_rating"]
        else:
            r = self.dest_df["avg_rating"]
        self.dest_df["norm_rating"] = np.clip((r - 1.0) / 4.0, 0.0, 1.0)

        # 2. Sentiment normalization (scale [-1.0, 1.0] to [0.0, 1.0])
        if "avg_sentiment" in self.dest_df.columns:
            s = self.dest_df["avg_sentiment"].fillna(0.0)
            self.dest_df["norm_sentiment"] = np.clip((s + 1.0) / 2.0, 0.0, 1.0)
        else:
            self.dest_df["norm_sentiment"] = 0.5  # Neutral default

        # 3. Popularity normalization (log-transformed review count)
        log_reviews = np.log1p(self.dest_df["review_count"].fillna(1))
        max_log = log_reviews.max() if log_reviews.max() > 0 else 1.0
        self.dest_df["norm_popularity"] = log_reviews / max_log

    def recommend(
        self,
        city: Optional[str] = None,
        interests: Optional[List[str]] = None,
        custom_query: Optional[str] = None,
        min_rating: float = 3.5,
        min_reviews: int = 15,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Generates personalized destination recommendations based on query, filters,
        and multi-factor hybrid scoring.
        """
        filtered_indices = self.dest_df.index

        # Filter by City
        if city and city.lower() != "all cities":
            filtered_indices = filtered_indices[
                self.dest_df.loc[filtered_indices, "City"].str.lower() == city.lower()
            ]

        # Filter by Minimum Rating and Minimum Review Count
        rating_col = "avg_rating" if "avg_rating" in self.dest_df.columns else "bayesian_rating"
        filtered_indices = filtered_indices[
            (self.dest_df.loc[filtered_indices, rating_col] >= min_rating) &
            (self.dest_df.loc[filtered_indices, "review_count"] >= min_reviews)
        ]

        if len(filtered_indices) == 0:
            return pd.DataFrame()

        # Construct query text from selected interests and custom keywords
        query_parts = []
        if interests:
            for item in interests:
                if item in CURATED_INTERESTS:
                    query_parts.append(CURATED_INTERESTS[item])
                else:
                    query_parts.append(item)
        if custom_query and custom_query.strip():
            query_parts.append(custom_query.strip())

        full_query = " ".join(query_parts).strip()

        # Compute Text Cosine Similarity
        if full_query:
            query_vec = self.vectorizer.transform([clean_text_for_nlp(full_query)])
            sub_tfidf = self.tfidf_matrix[filtered_indices]
            sim_scores = cosine_similarity(query_vec, sub_tfidf).flatten()
        else:
            sim_scores = np.zeros(len(filtered_indices))

        candidates = self.dest_df.loc[filtered_indices].copy()
        candidates["sim_score"] = sim_scores

        # Adjust weights if no text query provided (fallback to popularity + quality)
        w_sim = self.w_sim if full_query else 0.0
        total_w = w_sim + self.w_rating + self.w_sent + self.w_pop
        norm_w_sim = w_sim / total_w
        norm_w_rating = self.w_rating / total_w
        norm_w_sent = self.w_sent / total_w
        norm_w_pop = self.w_pop / total_w

        # Compute Composite Match Score
        composite_score = (
            norm_w_sim * candidates["sim_score"] +
            norm_w_rating * candidates["norm_rating"] +
            norm_w_sent * candidates["norm_sentiment"] +
            norm_w_pop * candidates["norm_popularity"]
        )

        candidates["match_score"] = (composite_score * 100).round(1)
        candidates["match_score"] = np.clip(candidates["match_score"], 0.0, 100.0)

        # Generate Evidence-Based Explanation
        def explain_recommendation(row) -> str:
            reasons = []
            if row["sim_score"] > 0.15:
                reasons.append("matches your preferred travel theme")
            if row.get("avg_rating", 0) >= 4.5:
                reasons.append(f"consistently outstanding rating ({row.get('avg_rating', 0):.1f}★)")
            elif row.get("avg_rating", 0) >= 4.0:
                reasons.append(f"strong visitor rating ({row.get('avg_rating', 0):.1f}★)")
            if row.get("review_count", 0) >= 100:
                reasons.append(f"backed by {int(row['review_count'])} reviews")
            if row.get("avg_sentiment", 0) >= 0.25:
                reasons.append("strongly positive visitor feedback")
            
            if reasons:
                return "Recommended because it " + ", ".join(reasons) + "."
            return "Popular destination matching your location and rating criteria."

        candidates["explanation"] = candidates.apply(explain_recommendation, axis=1)

        # Sort and return Top N
        sorted_candidates = candidates.sort_values(
            by=["match_score", "bayesian_rating", "review_count"],
            ascending=[False, False, False]
        ).head(top_n)

        return sorted_candidates

    def get_baseline_recommendations(
        self,
        city: Optional[str] = None,
        min_reviews: int = 15,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Baseline recommender: Ranks strictly by Bayesian rating and review count.
        Represents the popular/highest-rated benchmark.
        """
        filtered = self.dest_df.copy()
        if city and city.lower() != "all cities":
            filtered = filtered[filtered["City"].str.lower() == city.lower()]
        filtered = filtered[filtered["review_count"] >= min_reviews]
        return filtered.sort_values(
            by=["bayesian_rating", "review_count"],
            ascending=[False, False]
        ).head(top_n)
