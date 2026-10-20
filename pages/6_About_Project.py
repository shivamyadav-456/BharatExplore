"""About BharatExplore — System Overview, Architecture, and Methodology."""

import sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="About BharatExplore", page_icon="🧭", layout="wide")

st.markdown("""
<style>
    .page-title { font-size: 2.1rem; font-weight: 800; color: #0F766E; margin-bottom: 0.2rem; }
    .about-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🧭 About BharatExplore & System Architecture</div>', unsafe_allow_html=True)
st.caption("Distributed travel analytics, recommendation methodology, and data governance specifications.")

st.markdown("""
<div class="about-card">
    <h3 style="color: #0F766E;">🎯 Purpose & Problem Statement</h3>
    <p>
        Exploring India's vast travel landscape presents tourists with information overload, fragmented reviews,
        and unverified star ratings where single-review attractions often distort recommendations.
    </p>
    <p>
        <strong>BharatExplore</strong> is an end-to-end Big Data analytics and travel intelligence platform.
        It processes large-scale public review datasets using distributed computing, extracts sentiment valence
        from visitor feedback, and generates transparent, evidence-weighted destination recommendations
        across thousands of tourist attractions in India.
    </p>
</div>

<div class="about-card">
    <h3 style="color: #0F766E;">🏗️ Data & Processing Architecture</h3>
    <p>BharatExplore utilizes a production-grade distributed data processing and analytics architecture:</p>
    <ol>
        <li><strong>Distributed Ingestion:</strong> Apache Spark (PySpark) multi-line CSV streaming parser handling 541 MB of raw tourist feedback with explicit typing.</li>
        <li><strong>Preprocessing & Privacy Layer:</strong> Distributed text cleaning, deduplication, schema validation, and visitor PII anonymization.</li>
        <li><strong>Storage Layer:</strong> Snappy-compressed columnar Parquet files for high-throughput analytical querying and lightweight serve-time execution.</li>
        <li><strong>NLP Sentiment Engine:</strong> Offline-capable VADER-compatible sentiment intensity valence analyzer computing continuous compound scores [-1.0, 1.0].</li>
        <li><strong>Hybrid Recommendation Engine:</strong> Content-based TF-IDF vectorizer coupled with Bayesian rating confidence (IMDb formula) to ensure small-sample outliers do not outrank well-verified destinations.</li>
        <li><strong>Statistical Inference Engine:</strong> SciPy-powered inferential testing suite (One-way ANOVA, Kruskal-Wallis, Spearman rank correlation, Welch's t-test with effect sizes).</li>
        <li><strong>Interactive Dashboard:</strong> Multi-page Streamlit application with responsive Plotly visualizations and real-time parameter tuning.</li>
    </ol>
</div>

<div class="about-card">
    <h3 style="color: #0F766E;">🤖 Recommendation & Ranking Methodology</h3>
    <p>
        Recommendations combine four normalized signals into a composite score:
    </p>
    <ul>
        <li><strong>Text Similarity (TF-IDF):</strong> Cosine similarity between user travel interests/queries and aggregated destination visitor reviews.</li>
        <li><strong>Bayesian Confidence Rating:</strong> Weighted rating balancing destination observed mean with national average rating based on review volume threshold.</li>
        <li><strong>Sentiment Valence:</strong> Average compound sentiment extracted from actual visitor review texts.</li>
        <li><strong>Review Volume Popularity:</strong> Log-normalized review counts providing evidence confidence.</li>
    </ul>
    <p>
        All scoring weights are exposed to the user for explainability, with unpersonalized Bayesian ranking provided as a benchmark comparison.
    </p>
</div>

<div class="about-card">
    <h3 style="color: #0F766E;">📚 Dataset Attribution & Ethical Standards</h3>
    <ul>
        <li><strong>Dataset Source:</strong> <em>Indian Places to Visit Reviews Data</em> by Ritvik1909 on Kaggle.</li>
        <li><strong>Scope:</strong> Over 1.48 million review records across 14,400+ destinations in 1,700+ Indian cities and hubs.</li>
        <li><strong>Ethical & Scientific Safeguards:</strong>
            <ul>
                <li>Reviewer names are stripped and anonymized to protect personal identifiable information.</li>
                <li>Because the source <code>Date</code> column is completely unpopulated, temporal trends and seasonality claims are scientifically omitted.</li>
                <li>Geographical GPS coordinates are not fabricated, preserving mapping authenticity.</li>
                <li>No fake collaborative filtering interactions or fabricated accuracy metrics are reported.</li>
            </ul>
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)
