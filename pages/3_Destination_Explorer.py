"""Destination Explorer Page for BharatExplore."""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_access import get_destination_cities, load_destination_profiles

st.set_page_config(page_title="Destination Explorer — BharatExplore", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
    .page-title { font-size: 2.1rem; font-weight: 800; color: #0F766E; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🗺️ Indian Destination Explorer</div>', unsafe_allow_html=True)
st.caption("Browse, search, and inspect statistical profiles for thousands of verified tourist destinations across India.")

dest_df = load_destination_profiles()

if dest_df is None:
    st.warning("⚠️ Destination profiles not found.")
    st.info("Please run the pipeline script:\n```bash\npython scripts/run_pipeline.py\n```")
    st.stop()

if "dest_filter_version" not in st.session_state:
    st.session_state.dest_filter_version = 0

version = st.session_state.dest_filter_version

# Search and Filters
col1, col2, col3, col4, col5 = st.columns([1.5, 1.1, 0.9, 0.9, 0.6])

with col1:
    search_term = st.text_input(
        "🔍 Search Destination by Name",
        placeholder="e.g. Taj Mahal, Gateway, Qutub...",
        key=f"dest_search_{version}",
    )
with col2:
    all_cities = get_destination_cities()
    selected_city = st.selectbox("Filter City", all_cities, key=f"dest_city_{version}")
with col3:
    min_rating = st.slider("Min Rating", 1.0, 5.0, 3.5, 0.1, key=f"dest_rating_{version}")
with col4:
    min_reviews = st.number_input("Min Reviews", min_value=1, max_value=500, value=15, step=5, key=f"dest_reviews_{version}")
with col5:
    st.write("")
    st.write("")
    if st.button("🔄 Reset", use_container_width=True, help="Reset all filters to defaults"):
        st.session_state.dest_filter_version += 1
        st.rerun()

filtered = dest_df.copy()
if search_term.strip():
    filtered = filtered[filtered["Place"].str.contains(search_term.strip(), case=False, na=False)]
if selected_city != "All Cities":
    filtered = filtered[filtered["City"] == selected_city]
filtered = filtered[filtered["avg_rating"] >= min_rating]
filtered = filtered[filtered["review_count"] >= min_reviews]

st.markdown(f"**Found {len(filtered):,} destinations matching filters:**")

# Summary Metrics for Filtered Set
if len(filtered) > 0:
    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.metric("Destinations Found", f"{len(filtered):,}")
    with sm2:
        st.metric("Mean Rating", f"{filtered['avg_rating'].mean():.2f} ★")
    with sm3:
        st.metric("Total Reviews in Selection", f"{int(filtered['review_count'].sum()):,}")
    with sm4:
        st.metric("Average Sentiment", f"{filtered['avg_sentiment'].mean():.2f}" if 'avg_sentiment' in filtered.columns else "N/A")

    # Display Table
    display_cols = ["Place", "City", "avg_rating", "bayesian_rating", "review_count"]
    if "sentiment_summary" in filtered.columns:
        display_cols.append("sentiment_summary")
    if "avg_sentiment" in filtered.columns:
        display_cols.append("avg_sentiment")

    rename_map = {
        "Place": "Destination",
        "City": "City",
        "avg_rating": "Observed Rating",
        "bayesian_rating": "Bayesian Rating",
        "review_count": "Reviews Count",
        "sentiment_summary": "Sentiment Profile",
        "avg_sentiment": "Compound Valence",
    }

    st.dataframe(
        filtered[display_cols].rename(columns=rename_map).sort_values(by="Reviews Count", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Note on Map and Geographic Integrity
    st.info(
        "ℹ️ **Geographic Data Integrity Note:** The public Kaggle dataset provides city and place textual identifiers "
        "without verified GPS coordinates. In adherence to data integrity and transparency standards, synthetic or "
        "approximate GPS coordinates are intentionally NOT fabricated for map plotting."
    )
else:
    st.warning("No destinations match the chosen filter combination. Try loosening minimum rating or review count filters.")
