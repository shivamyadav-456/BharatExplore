"""Data Quality and Methodology Page for BharatExplore."""

import json
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_access import load_analysis_summary, load_quality_report

st.set_page_config(page_title="Data Quality — BharatExplore", page_icon="📋", layout="wide")

st.markdown("""
<style>
    .page-title { font-size: 2.1rem; font-weight: 800; color: #0F766E; margin-bottom: 0.2rem; }
    .quality-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📋 Data Profiling, Quality & Ingestion Auditing</div>', unsafe_allow_html=True)
st.caption("Comprehensive data-quality assessment, schema validation, and missingness audit following Big Data engineering practices.")

quality_rep = load_quality_report()

if quality_rep is None:
    st.warning("⚠️ Data quality report not found.")
    st.info("Please run the validation script:\n```bash\npython scripts/validate_dataset.py\n```")
    st.stop()

# Key Data Quality Indicators
m1, m2, m3, m4 = st.columns(4)
metrics = quality_rep.get("metrics", {})
null_analysis = quality_rep.get("null_value_analysis", {})

with m1:
    st.metric("Total Ingested Records", f"{metrics.get('total_rows', 0):,}")
with m2:
    st.metric("Total Schema Columns", f"{metrics.get('total_columns', 0)}")
with m3:
    st.metric("Invalid Ratings Detected", f"{metrics.get('invalid_ratings_count', 0)}")
with m4:
    st.metric("Duplicate Review Clusters", f"{metrics.get('duplicate_review_groups', 0):,}")

st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.subheader("🔍 Column Missingness & Null Value Audit")
    null_rows = [
        {
            "Column": col,
            "Null/Empty Records": data.get("null_or_empty_count", 0),
            "Null Percentage (%)": data.get("null_percentage", 0.0),
        }
        for col, data in null_analysis.items()
    ]
    null_df = pd.DataFrame(null_rows)
    st.dataframe(null_df, use_container_width=True, hide_index=True)

    st.info(
        "ℹ️ **Missing Data Disclosure (Date Column):** "
        "The `Date` column in the source dataset is 100% empty (null). "
        "In accordance with strict scientific integrity and data truthfulness principles, "
        "temporal trends, monthly fluctuations, or seasonality analyses are intentionally omitted rather than fabricated."
    )

with c2:
    st.subheader("🛡️ PII Sanitization & Data Protection")
    st.info(
        "🔒 **Reviewer Privacy Protection:** "
        "The raw column `Name` contains user handles and potential personally identifiable information (PII). "
        "Our PySpark preprocessing pipeline strips and masks all names to `Verified Visitor` before any "
        "downstream analytical aggregation or dashboard presentation."
    )
    st.markdown("##### ⚙️ Data Ingestion Metadata")
    ds_meta = quality_rep.get("dataset_metadata", {})
    st.markdown(f"""
    - **Source File:** `{ds_meta.get('source_file')}`
    - **Uncompressed Size:** {ds_meta.get('file_size_mb', 0)} MB
    - **Ingestion Time (Spark):** {ds_meta.get('ingest_duration_seconds', 0)} seconds
    - **Execution Mode:** {'Dev Sample' if quality_rep.get('is_sample') else 'Full Big Data Run'}
    """)

st.markdown("---")
st.subheader("📥 Export Auditing Reports")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.download_button(
        "Download Data Quality Report (JSON)",
        data=json.dumps(quality_rep, indent=2),
        file_name="data_quality_report.json",
        mime="application/json"
    )
with col_d2:
    st.download_button(
        "Download Null Value Analysis (CSV)",
        data=null_df.to_csv(index=False),
        file_name="null_value_analysis.csv",
        mime="text/csv"
    )
