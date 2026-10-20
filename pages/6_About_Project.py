"""About Project and Academic Methodology Page for BharatExplore."""

import sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="About Project — BharatExplore", page_icon="🎓", layout="wide")

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

st.markdown('<div class="page-title">🎓 About BharatExplore & Academic Background</div>', unsafe_allow_html=True)
st.caption("College Mini Project Documentation — Big Data Analysis Lab (CSL702), Mumbai University.")

st.markdown("""
<div class="about-card">
    <h3 style="color: #0F766E;">🏛️ Academic Alignment & Syllabus Details</h3>
    <ul>
        <li><strong>Degree:</strong> Bachelor of Engineering (B.E.) in Computer Engineering</li>
        <li><strong>University:</strong> University of Mumbai (MU)</li>
        <li><strong>Semester:</strong> Semester VII</li>
        <li><strong>Course:</strong> Big Data Analytics Lab (CSL702)</li>
        <li><strong>Experiment 11 Objective:</strong> Design and implement a real-life application using standard datasets, performing data acquisition, big data cleaning, distributed exploratory data analysis, hypothesis testing, machine learning recommendation modeling, NLP sentiment analysis, and model evaluation.</li>
    </ul>
</div>

<div class="about-card">
    <h3 style="color: #0F766E;">🏗️ Big Data System Architecture</h3>
    <p>BharatExplore utilizes a production-grade distributed Big Data analytics architecture:</p>
    <ol>
        <li><strong>Ingestion Engine:</strong> Apache Spark (PySpark 4.2.0) with multi-line CSV streaming parser handling 541 MB of raw tourist feedback.</li>
        <li><strong>Preprocessing & Privacy Layer:</strong> Distributed text cleaning, deduplication, schema validation, and PII masking.</li>
        <li><strong>Storage Layer:</strong> Columnar, snappy-compressed Parquet files for high-throughput distributed I/O.</li>
        <li><strong>NLP Sentiment Engine:</strong> Offline-capable VADER-compatible sentiment intensity valence analyzer computing continuous compound scores [-1.0, 1.0].</li>
        <li><strong>Recommender Engine:</strong> Content-based TF-IDF vectorizer coupled with Bayesian confidence ratings (IMDb formula) to prevent sparse 1-review anomalies from outranking established destinations.</li>
        <li><strong>Statistical Inference Engine:</strong> SciPy-powered inferential hypothesis testing (One-way ANOVA, Kruskal-Wallis, Spearman Rank correlation, Welch's t-test with effect sizes).</li>
        <li><strong>Presentation Dashboard:</strong> Interactive Streamlit multi-page application with responsive Plotly visualizations.</li>
    </ol>
</div>

<div class="about-card">
    <h3 style="color: #0F766E;">📚 Dataset Attribution & Ethical Standards</h3>
    <ul>
        <li><strong>Dataset Source:</strong> <em>Indian Places to Visit Reviews Data</em> by Ritvik1909 on Kaggle.</li>
        <li><strong>Scope:</strong> Over 1.48 million review records across 14,400+ destinations in 1,700+ Indian cities and hubs.</li>
        <li><strong>Ethical Safeguards:</strong>
            <ul>
                <li>Reviewer names are strictly stripped from all dashboard displays to protect user privacy.</li>
                <li>Because the source <code>Date</code> column is completely unpopulated, temporal analysis and seasonality claims are scientifically withheld.</li>
                <li>Geographical GPS coordinates are not fabricated, preserving mapping authenticity.</li>
                <li>No fake collaborative filtering interactions or fabricated accuracy metrics are reported.</li>
            </ul>
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)
