"""Recommendation engine offline evaluation and diagnostic module."""

import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.config import MODEL_EVALUATION_REPORT
from src.recommendation_engine import CURATED_INTERESTS, TravelRecommender
from src.utils import get_logger, save_json_report

logger = get_logger("ModelEvaluation")


def evaluate_recommendation_system(
    recommender: TravelRecommender,
    top_k: int = 10,
    min_reviews: int = 15,
) -> Dict[str, Any]:
    """
    Evaluates the content-based hybrid recommender across standard benchmark queries
    and compares against the popularity baseline on coverage, diversity, and rating confidence.
    """
    logger.info("Starting recommendation engine offline evaluation...")
    start_time = time.time()

    dest_df = recommender.dest_df
    total_destinations = len(dest_df)

    test_queries = [
        {"persona": "History & Heritage", "interests": ["Heritage & History"], "city": "All Cities"},
        {"persona": "Nature & Scenic", "interests": ["Nature & Landscapes"], "city": "All Cities"},
        {"persona": "Spiritual Pilgrim", "interests": ["Spiritual & Temples"], "city": "All Cities"},
        {"persona": "Coastal & Beach", "interests": ["Beaches & Coastal"], "city": "All Cities"},
        {"persona": "Adventure Trekker", "interests": ["Adventure & Trekking"], "city": "All Cities"},
        {"persona": "Wildlife Enthusiast", "interests": ["Wildlife & Safari"], "city": "All Cities"},
    ]

    all_recommended_places = set()
    persona_results = []

    for t in test_queries:
        recs = recommender.recommend(
            city=t["city"],
            interests=t["interests"],
            min_rating=3.5,
            min_reviews=min_reviews,
            top_n=top_k,
        )

        if len(recs) > 0:
            all_recommended_places.update(recs["Place"].tolist())
            avg_match = float(recs["match_score"].mean())
            avg_rating = float(recs["avg_rating"].mean())
            avg_reviews = float(recs["review_count"].mean())
            distinct_cities = int(recs["City"].nunique())

            persona_results.append({
                "persona": t["persona"],
                "results_returned": len(recs),
                "avg_match_score": round(avg_match, 2),
                "avg_star_rating": round(avg_rating, 2),
                "avg_review_count": round(avg_reviews, 1),
                "unique_cities_in_top_k": distinct_cities,
                "top_recommended_places": recs["Place"].head(3).tolist(),
            })

    # Catalog Coverage
    catalog_coverage_pct = round((len(all_recommended_places) / total_destinations) * 100, 2) if total_destinations > 0 else 0.0

    # Baseline Comparison
    baseline_top_k = recommender.get_baseline_recommendations(top_n=top_k, min_reviews=min_reviews)
    baseline_places = set(baseline_top_k["Place"].tolist())

    # Measure novelty / divergence from pure baseline
    overlap_count = len(all_recommended_places.intersection(baseline_places))
    novelty_divergence_pct = round((1.0 - (overlap_count / max(1, len(all_recommended_places)))) * 100, 2)

    duration = round(time.time() - start_time, 2)

    evaluation_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": duration,
        "methodology": {
            "type": "Content-Based Hybrid Multi-Persona Diagnostic",
            "scoring_components": ["TF-IDF Cosine Similarity", "Bayesian Adjusted Rating", "VADER Sentiment Valence", "Review Volume"],
            "top_k_evaluated": top_k,
            "min_reviews_filter": min_reviews,
        },
        "system_diagnostics": {
            "total_destinations_catalog": total_destinations,
            "distinct_destinations_recommended_across_personas": len(all_recommended_places),
            "catalog_coverage_percentage": catalog_coverage_pct,
            "novelty_vs_popularity_baseline_pct": novelty_divergence_pct,
        },
        "persona_evaluations": persona_results,
        "baseline_benchmark": {
            "baseline_type": "Bayesian Rating Ranking with Min Reviews",
            "top_places": baseline_top_k["Place"].tolist(),
            "avg_baseline_rating": round(float(baseline_top_k["avg_rating"].mean()), 2) if len(baseline_top_k) > 0 else 0.0,
            "avg_baseline_reviews": round(float(baseline_top_k["review_count"].mean()), 1) if len(baseline_top_k) > 0 else 0.0,
        },
        "academic_disclosure": (
            "Because this public dataset contains anonymous unlinked reviews without persistent user profiles, "
            "recommendation evaluation is performed via content relevance, coverage, and rank stability diagnostics "
            "rather than fabricating simulated user click logs for collaborative-filtering precision/recall."
        )
    }

    save_json_report(evaluation_report, MODEL_EVALUATION_REPORT)
    logger.info(f"Model evaluation report saved to {MODEL_EVALUATION_REPORT}")

    return evaluation_report
