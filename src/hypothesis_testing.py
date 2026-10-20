"""Hypothesis testing module for BharatExplore.

Implements rigorous inferential statistical tests on review ratings, text lengths,
and sentiment distributions following Mumbai University BDA Lab specifications.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger

logger = get_logger("HypothesisTesting")


def run_city_rating_comparison_test(
    reviews_df: pd.DataFrame,
    top_n_cities: int = 5,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Hypothesis Test 1: City-Wise Rating Differences.
    H0: Mean tourist ratings are equal across top tourist cities.
    H1: At least one city's mean tourist rating differs significantly.
    Tests: Levene's test for equal variance, then One-Way ANOVA and Kruskal-Wallis H-test.
    """
    logger.info("Executing Hypothesis Test 1: City-Wise Rating Differences...")

    # Identify top N cities by review volume
    city_counts = reviews_df["City"].value_counts()
    selected_cities = city_counts.head(top_n_cities).index.tolist()

    groups = [
        reviews_df[reviews_df["City"] == city]["Rating"].dropna().values
        for city in selected_cities
    ]

    city_stats_summary = [
        {
            "city": city,
            "sample_size": len(grp),
            "mean_rating": round(float(np.mean(grp)), 3),
            "std_dev": round(float(np.std(grp, ddof=1)), 3),
        }
        for city, grp in zip(selected_cities, groups)
    ]

    # Assumption check: Levene's test for homogeneity of variance
    levene_stat, levene_p = stats.levene(*groups)

    # Parametric: One-Way ANOVA
    f_stat, anova_p = stats.f_oneway(*groups)

    # Non-parametric: Kruskal-Wallis H-test
    kw_stat, kw_p = stats.kruskal(*groups)

    # Effect size: Eta-squared (η²) for ANOVA
    # η² = SS_between / SS_total
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = sum(len(grp) * (np.mean(grp) - grand_mean) ** 2 for grp in groups)
    eta_squared = round(float(ss_between / ss_total), 4) if ss_total > 0 else 0.0

    # Decision
    recommended_test = "Kruskal-Wallis" if levene_p < alpha else "One-Way ANOVA"
    chosen_p = kw_p if recommended_test == "Kruskal-Wallis" else anova_p
    rejected = chosen_p < alpha

    effect_label = "small" if eta_squared < 0.06 else "moderate"

    if rejected:
        interpretation = (
            f"At significance level α = {alpha}, we REJECT the null hypothesis "
            f"({recommended_test}: p = {chosen_p:.4f}). "
            f"At least one city shows a statistically significant difference in mean tourist rating. "
            f"However, the effect size (η² = {eta_squared}) indicates a {effect_label} practical impact; "
            f"other factors (place type, seasonality, individual preference) also contribute to rating variance."
        )
    else:
        interpretation = (
            f"At significance level α = {alpha}, we FAIL TO REJECT the null hypothesis "
            f"({recommended_test}: p = {chosen_p:.4f}). "
            f"The data do not provide sufficient evidence of statistically significant rating differences "
            f"across the selected cities: {', '.join(selected_cities)}. "
            f"This does not prove the groups are identical — sample size and within-group variance may limit "
            f"statistical power. Effect size η² = {eta_squared} ({effect_label})."
        )

    return {
        "test_name": "City-Wise Tourist Rating Comparison",
        "null_hypothesis": "H0: The mean tourist ratings are identical across selected major cities.",
        "alt_hypothesis": "H1: At least one city has a significantly different mean tourist rating.",
        "significance_level": alpha,
        "selected_cities": selected_cities,
        "group_summary": city_stats_summary,
        "assumptions": {
            "levene_statistic": round(float(levene_stat), 4),
            "levene_p_value": float(levene_p),
            "equal_variances_assumed": bool(levene_p >= alpha),
        },
        "anova_results": {
            "f_statistic": round(float(f_stat), 4),
            "p_value": float(anova_p),
        },
        "kruskal_results": {
            "h_statistic": round(float(kw_stat), 4),
            "p_value": float(kw_p),
        },
        "recommended_test": recommended_test,
        "effect_size": {"metric": "Eta-squared (η²)", "value": eta_squared},
        "decision": "Reject H0" if rejected else "Fail to Reject H0",
        "interpretation": interpretation,
    }


def run_review_length_correlation_test(
    reviews_df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Hypothesis Test 2: Association between Review Word Count and Star Rating.
    H0: There is no correlation between review length and rating (rho = 0).
    H1: There is a statistically significant correlation between review length and rating (rho != 0).
    """
    logger.info("Executing Hypothesis Test 2: Review Length vs Rating Correlation...")

    sub = reviews_df[["Rating", "review_word_count"]].dropna()
    ratings = sub["Rating"].values
    lengths = sub["review_word_count"].values

    # Spearman rank correlation
    spearman_rho, spearman_p = stats.spearmanr(ratings, lengths)
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(ratings, lengths)

    rejected = spearman_p < alpha

    p_display = f"{spearman_p:.4e}" if spearman_p < 0.001 else f"{spearman_p:.4f}"

    if rejected:
        if spearman_rho < 0:
            direction = (
                "Negative correlation: dissatisfied visitors tend to write longer, more detailed reviews "
                "explaining their grievances, while highly satisfied visitors frequently leave brief commendations."
            )
        else:
            direction = "Positive correlation: higher-rated reviews tend to be longer in this sample."
        interpretation = (
            f"At α = {alpha}, we REJECT the null hypothesis (Spearman ρ = {spearman_rho:.3f}, p = {p_display}). "
            f"{direction}"
        )
    else:
        interpretation = (
            f"At α = {alpha}, we FAIL TO REJECT the null hypothesis "
            f"(Spearman ρ = {spearman_rho:.3f}, p = {p_display}). "
            f"No statistically significant monotonic correlation between review word count and star rating "
            f"was detected in this sample."
        )

    return {
        "test_name": "Review Length vs Tourist Rating Correlation",
        "null_hypothesis": "H0: True correlation between review word count and rating is zero (ρ = 0).",
        "alt_hypothesis": "H1: True correlation between review word count and rating is non-zero (ρ ≠ 0).",
        "sample_size": len(sub),
        "significance_level": alpha,
        "spearman_rank_correlation": {
            "rho": round(float(spearman_rho), 4),
            "p_value": float(spearman_p),
        },
        "pearson_correlation": {
            "r": round(float(pearson_r), 4),
            "p_value": float(pearson_p),
        },
        "decision": "Reject H0" if rejected else "Fail to Reject H0",
        "interpretation": interpretation,
    }


def run_sentiment_rating_association_test(
    reviews_df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Hypothesis Test 3: Sentiment Valence in 5-Star vs 1-Star Reviews.
    H0: Mean sentiment score of 5-star reviews <= mean sentiment score of 1-star reviews.
    H1: Mean sentiment score of 5-star reviews > mean sentiment score of 1-star reviews.
    """
    logger.info("Executing Hypothesis Test 3: Sentiment vs Rating Association...")

    sub = reviews_df.dropna(subset=["Rating", "sentiment_compound"])
    g5 = sub[sub["Rating"] == 5.0]["sentiment_compound"].values
    g1 = sub[sub["Rating"] == 1.0]["sentiment_compound"].values

    if len(g5) == 0 or len(g1) == 0:
        return {"error": "Insufficient 1-star or 5-star samples for hypothesis test."}

    # Two-sample Welch's t-test (unequal variance)
    t_stat, t_p = stats.ttest_ind(g5, g1, equal_var=False, alternative="greater")

    # Mann-Whitney U test (non-parametric rank sum)
    u_stat, u_p = stats.mannwhitneyu(g5, g1, alternative="greater")

    # Cohen's d effect size
    n1, n5 = len(g1), len(g5)
    s1, s5 = np.var(g1, ddof=1), np.var(g5, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n5 - 1) * s5) / (n1 + n5 - 2))
    cohens_d = round(float((np.mean(g5) - np.mean(g1)) / pooled_sd), 3) if pooled_sd > 0 else 0.0

    rejected = t_p < alpha

    t_p_display = f"{t_p:.4e}" if t_p < 0.001 else f"{t_p:.4f}"
    mean_g5 = round(float(np.mean(g5)), 3)
    mean_g1 = round(float(np.mean(g1)), 3)
    effect_magnitude = (
        "large" if abs(cohens_d) >= 0.8 else
        "medium" if abs(cohens_d) >= 0.5 else
        "small"
    )

    if rejected:
        interpretation = (
            f"At α = {alpha}, we REJECT the null hypothesis (Welch's t = {t_stat:.2f}, p = {t_p_display}). "
            f"5-star reviews exhibit significantly higher positive sentiment (mean = {mean_g5}) compared to "
            f"1-star reviews (mean = {mean_g1}). Cohen's d = {cohens_d} indicates a {effect_magnitude} effect size, "
            f"confirming that the VADER-based lexicon sentiment scores align strongly with explicit star ratings. "
            f"Note: VADER is a lexicon- and rule-based method, not a generative AI model."
        )
    else:
        interpretation = (
            f"At α = {alpha}, we FAIL TO REJECT the null hypothesis (Welch's t = {t_stat:.2f}, p = {t_p_display}). "
            f"No statistically significant difference in sentiment between 5-star (mean = {mean_g5}) "
            f"and 1-star (mean = {mean_g1}) reviews was detected. Cohen's d = {cohens_d}."
        )

    return {
        "test_name": "Sentiment Score Divergence (5-Star vs 1-Star)",
        "null_hypothesis": "H0: Mean sentiment compound score of 5-star reviews ≤ 1-star reviews.",
        "alt_hypothesis": "H1: Mean sentiment compound score of 5-star reviews > 1-star reviews.",
        "significance_level": alpha,
        "sample_size_5_star": len(g5),
        "sample_size_1_star": len(g1),
        "mean_sentiment_5_star": mean_g5,
        "mean_sentiment_1_star": mean_g1,
        "welch_t_test": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(t_p),
        },
        "mann_whitney_u_test": {
            "u_statistic": float(u_stat),
            "p_value": float(u_p),
        },
        "effect_size": {"metric": "Cohen's d", "value": cohens_d},
        "decision": "Reject H0" if rejected else "Fail to Reject H0",
        "interpretation": interpretation,
    }
