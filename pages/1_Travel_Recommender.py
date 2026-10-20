"""Travel Recommender Page for BharatExplore.

Interactive Content-Based Hybrid Recommender with Explainable Match Scores
and Baseline Comparison.

Workflow:
  1. User fills in preferences inside a form (city, interests, filters, weights).
  2. User clicks "Recommend Destinations" to generate results.
  3. Results stay stable until the user submits again.
  4. "Clear Preferences" button resets all selections to neutral defaults.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_access import get_destination_cities, get_recommender
from src.recommendation_engine import CURATED_INTERESTS

st.set_page_config(page_title="Travel Recommender — BharatExplore", page_icon="🧭", layout="wide")

st.markdown("""
<style>
    .page-title { font-size: 2.1rem; font-weight: 800; color: #0F766E; margin-bottom: 0.2rem; }
    .rec-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.15s ease-in-out;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    .dest-title { font-size: 1.35rem; font-weight: 700; color: #0F766E; }
    .dest-city  { font-size: 0.95rem; font-weight: 600; color: #64748B; }
    .match-badge {
        background: #CCFBF1;
        color: #0F766E;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .evidence-text {
        font-size: 0.9rem;
        color: #334155;
        background: #F8FAFC;
        border-left: 3px solid #0F766E;
        padding: 0.5rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin-top: 0.6rem;
    }
    .pref-summary {
        background: #F0FDFA;
        border: 1px solid #99F6E4;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        color: #134E4A;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🧭 Hybrid Travel Recommender</div>', unsafe_allow_html=True)
st.caption(
    "Personalized destination rankings using TF-IDF text cosine similarity, "
    "Bayesian confidence-adjusted ratings (IMDb formula), and VADER sentiment valence. "
    "Recommendations are drawn from 7,798 destination profiles built by the PySpark pipeline."
)

recommender = get_recommender()

if recommender is None:
    st.warning("⚠️ **Analytics models are not yet generated.**")
    st.info("Please run the pipeline first:\n```bash\npython scripts/run_pipeline.py\n```")
    st.stop()

dest_df = recommender.dest_df
cities_list = get_destination_cities()
interests_list = list(CURATED_INTERESTS.keys())

# ── Session state: store submitted preferences and results ──────────────────
# We only compute recommendations on explicit form submission, so results are
# stable across reruns triggered by Streamlit's internal refresh mechanism.

if "submitted_prefs" not in st.session_state:
    st.session_state.submitted_prefs = None   # None = not yet submitted
if "rec_results" not in st.session_state:
    st.session_state.rec_results = None
if "baseline_results" not in st.session_state:
    st.session_state.baseline_results = None
if "rec_form_version" not in st.session_state:
    st.session_state.rec_form_version = 0

version = st.session_state.rec_form_version

# ── Preference Form ─────────────────────────────────────────────────────────
st.markdown("### 🎯 Set Your Travel Preferences")
st.caption(
    "Choose your preferences below and click **Recommend Destinations**. "
    "Changing a control does *not* update results until you submit."
)

with st.form(f"recommender_form_{version}", clear_on_submit=False):
    col1, col2 = st.columns([1, 1])

    with col1:
        selected_city = st.selectbox(
            "📍 Preferred City / Region",
            cities_list,
            index=0,
            key=f"rec_form_city_{version}",
            help="Select a specific city or leave as 'All Cities' to search nationwide.",
        )
        selected_interests = st.multiselect(
            "✨ Desired Experiences & Themes (optional)",
            interests_list,
            default=[],
            key=f"rec_form_interests_{version}",
            help=(
                "Select one or more themes to personalise results using TF-IDF text similarity. "
                "Leave empty to rank by Bayesian rating and sentiment only."
            ),
        )
        custom_query = st.text_input(
            "🔍 Custom Search Keywords (optional)",
            value="",
            placeholder="e.g. peaceful sunset boating photography lake views",
            key=f"rec_form_query_{version}",
        )

    with col2:
        min_rating = st.slider(
            "⭐ Minimum Average Rating",
            1.0, 5.0, 3.5, 0.1,
            key=f"rec_form_min_rating_{version}",
            help="Destinations with an average rating below this threshold are excluded.",
        )
        min_reviews = st.slider(
            "📊 Minimum Review Count",
            5, 200, 15, 5,
            key=f"rec_form_min_reviews_{version}",
            help="Ensures recommendations have sufficient reviewer evidence.",
        )
        top_n = st.select_slider(
            "🔢 Number of Results",
            options=[5, 10, 15, 20, 25],
            value=10,
            key=f"rec_form_top_n_{version}",
        )

    with st.expander("⚙️ Advanced: Scoring Weight Configuration"):
        st.caption(
            "Weights are renormalised to sum to 1.0 automatically. "
            "If no interests or keywords are provided, the similarity weight is redistributed to rating and sentiment."
        )
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        with w_col1:
            w_sim = st.slider("Text Similarity", 0.0, 1.0, 0.40, 0.05, key=f"rec_form_w_sim_{version}")
        with w_col2:
            w_rating = st.slider("Bayesian Rating", 0.0, 1.0, 0.35, 0.05, key=f"rec_form_w_rating_{version}")
        with w_col3:
            w_sent = st.slider("Sentiment Valence", 0.0, 1.0, 0.15, 0.05, key=f"rec_form_w_sent_{version}")
        with w_col4:
            w_pop = st.slider("Review Volume", 0.0, 1.0, 0.10, 0.05, key=f"rec_form_w_pop_{version}")

    submit_col, reset_col = st.columns([3, 1])
    with submit_col:
        submitted = st.form_submit_button(
            "🔍 Recommend Destinations",
            use_container_width=True,
            type="primary",
        )
    with reset_col:
        reset_clicked = st.form_submit_button(
            "🗑️ Clear",
            use_container_width=True,
            type="secondary",
            help="Reset all preferences and clear results.",
        )

# ── Handle form actions (outside the form block) ────────────────────────────
if reset_clicked:
    st.session_state.submitted_prefs = None
    st.session_state.rec_results = None
    st.session_state.baseline_results = None
    st.session_state.rec_form_version += 1
    st.rerun()

if submitted:
    # Validate: warn if no query context but still allow submission
    has_query = bool(selected_interests or (custom_query and custom_query.strip()))

    # Apply custom weights to recommender
    recommender.w_sim = w_sim
    recommender.w_rating = w_rating
    recommender.w_sent = w_sent
    recommender.w_pop = w_pop

    results = recommender.recommend(
        city=selected_city,
        interests=selected_interests,
        custom_query=custom_query.strip() if custom_query else "",
        min_rating=min_rating,
        min_reviews=min_reviews,
        top_n=top_n,
    )

    baseline = recommender.get_baseline_recommendations(
        city=selected_city,
        min_reviews=min_reviews,
        top_n=top_n,
    )

    # Save to session state so results persist across Streamlit reruns
    st.session_state.submitted_prefs = {
        "city": selected_city,
        "interests": selected_interests,
        "custom_query": custom_query.strip() if custom_query else "",
        "min_rating": min_rating,
        "min_reviews": min_reviews,
        "top_n": top_n,
        "weights": {"similarity": w_sim, "rating": w_rating, "sentiment": w_sent, "popularity": w_pop},
        "has_query": has_query,
    }
    st.session_state.rec_results = results
    st.session_state.baseline_results = baseline

# ── Results Area ────────────────────────────────────────────────────────────
st.markdown("---")

if st.session_state.submitted_prefs is None:
    # No submission yet
    st.info(
        "👆 **Fill in your preferences above and click 'Recommend Destinations'** to generate personalised results.\n\n"
        "- **No interests selected?** Results will be ranked by Bayesian rating confidence and visitor sentiment.\n"
        "- **Interests selected?** TF-IDF text similarity against destination review profiles is used to personalise rankings.\n"
        "- **Custom keywords?** Combined with interests for a richer semantic query."
    )
else:
    prefs = st.session_state.submitted_prefs
    results = st.session_state.rec_results
    baseline = st.session_state.baseline_results

    # Show applied preferences summary
    interests_str = ", ".join(prefs["interests"]) if prefs["interests"] else "None (ranking by rating & sentiment)"
    query_str = f'"{prefs["custom_query"]}"' if prefs["custom_query"] else "—"
    st.markdown(f"""
    <div class="pref-summary">
        <strong>Applied preferences:</strong>
        City = <strong>{prefs['city']}</strong> &nbsp;|&nbsp;
        Interests = <strong>{interests_str}</strong> &nbsp;|&nbsp;
        Keywords = <strong>{query_str}</strong><br>
        Min Rating ≥ <strong>{prefs['min_rating']}</strong> &nbsp;|&nbsp;
        Min Reviews ≥ <strong>{prefs['min_reviews']}</strong> &nbsp;|&nbsp;
        Results = <strong>{prefs['top_n']}</strong>
    </div>
    """, unsafe_allow_html=True)

    if not prefs["has_query"]:
        st.info(
            "ℹ️ No interests or keywords selected — results ranked by **Bayesian rating confidence** "
            "and **visitor sentiment** only. Select themes above and re-submit for personalised rankings."
        )

    tab_recs, tab_baseline = st.tabs(
        ["🌟 Personalised Recommendations", "📊 Baseline (Bayesian Rating Benchmark)"]
    )

    with tab_recs:
        if results is None or len(results) == 0:
            st.warning("No destinations found matching all chosen criteria.")
            st.info(
                "💡 Suggestions:\n"
                "- Lower the **Minimum Average Rating** (currently {:.1f}).\n"
                "- Lower the **Minimum Review Count** (currently {:,}).\n"
                "- Switch city to **All Cities** if a specific city is selected.\n"
                "- Check that the selected filters are not too restrictive together.".format(
                    prefs["min_rating"], prefs["min_reviews"]
                )
            )
        else:
            st.markdown(f"**Found {len(results)} destination(s) matching your preferences:**")

            for _, row in results.iterrows():
                rating_val = row.get("avg_rating", 0)
                bayes_val = row.get("bayesian_rating", rating_val)
                reviews_val = int(row.get("review_count", 0))
                sentiment_val = row.get("sentiment_summary", "—")
                explanation = row.get("explanation", "Matches your criteria.")
                match_score = row.get("match_score", 0.0)

                st.markdown(f"""
                <div class="rec-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span class="dest-title">{row['Place']}</span> &nbsp;
                            <span class="dest-city">📍 {row['City']}</span>
                        </div>
                        <div class="match-badge">{match_score:.1f}% Match</div>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.95rem;">
                        <strong>Rating:</strong> {rating_val:.2f} ★ &nbsp;|&nbsp;
                        <strong>Bayesian Score:</strong> {bayes_val:.2f} ★ &nbsp;|&nbsp;
                        <strong>Reviews:</strong> {reviews_val:,} &nbsp;|&nbsp;
                        <strong>Sentiment:</strong> {sentiment_val}
                    </div>
                    <div class="evidence-text">
                        💡 <strong>Evidence:</strong> {explanation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_baseline:
        st.markdown("#### ⚖️ Baseline Benchmark: Pure Bayesian Rating Ranking")
        st.caption(
            "Ranks destinations strictly by Bayesian-adjusted rating (no text personalisation). "
            "Used as an unpersonalized benchmark to evaluate the personalisation benefit of TF-IDF text similarity and sentiment scoring."
        )
        if baseline is None or len(baseline) == 0:
            st.info("No baseline results available for the selected city and minimum review count.")
        else:
            display_cols = [c for c in [
                "Place", "City", "avg_rating", "bayesian_rating", "review_count", "sentiment_summary"
            ] if c in baseline.columns]
            st.dataframe(
                baseline[display_cols].rename(columns={
                    "Place": "Destination",
                    "avg_rating": "Mean Rating",
                    "bayesian_rating": "Bayesian Rating",
                    "review_count": "Reviews Count",
                    "sentiment_summary": "Sentiment",
                }),
                use_container_width=True,
                hide_index=True,
            )
