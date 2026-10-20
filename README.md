# BharatExplore — Big Data-Powered Indian Travel Recommendation System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-PySpark%204.2-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit%201.30+-red.svg)](https://streamlit.io/)
[![University](https://img.shields.io/badge/Mumbai%20University-CSL702%20Exp%2011-teal.svg)](https://mu.ac.in/)

> **Academic Alignment:**  
> **Course:** Bachelor of Engineering (B.E.) in Computer Engineering, Semester VII  
> **Subject:** Big Data Analytics Lab (CSL702), Experiment 11: Real-life Application using Standard Big Data Dataset  
> **University:** University of Mumbai (MU)

---

## 1. Project Overview

**BharatExplore** is an end-to-end Big Data analytics and intelligent recommendation platform designed for exploring Indian tourist destinations. Processing over **1.48 million public traveler reviews** across **14,400+ destinations** in **1,794 Indian cities**, the application addresses small-sample rating bias and provides explainable, personalized travel recommendations.

### Key Capabilities
- **Distributed Big Data Processing:** Ingests and cleans 516+ MB raw multi-line CSVs using **Apache Spark (PySpark)** and writes Snappy-compressed Parquet.
- **Bayesian Confidence Ratings:** Eliminates the "1-review 5.0-star" anomaly via empirical Bayes calibration (IMDb formula).
- **Offline NLP Sentiment Engine:** VADER-compatible sentiment intensity scoring capturing continuous valence $[-1.0, 1.0]$, handling negations and booster tokens completely offline.
- **Explainable Hybrid Recommender:** Multi-factor scoring combining TF-IDF cosine similarity, Bayesian rating confidence, sentiment valence, and review volume.
- **Inferential Statistical Hypothesis Testing:** Rigorous tests (One-Way ANOVA, Kruskal-Wallis, Spearman Rank correlation, Welch's unequal variance $t$-test) complete with effect size reporting ($\eta^2$, Cohen's $d$).
- **Interactive Multi-Page Web UI:** Polished, responsive Streamlit dashboard with custom India-inspired theme (forest green/teal accents, warm off-white canvas, saffron highlights) and Plotly charts.

---

## 2. System Architecture & Tech Stack

```
   Raw Dataset (archive.zip / Review_db.csv - 516 MB)
                      │
                      ▼
   PySpark Ingestion & Multi-line Parsing (Explicit Schema)
                      │
                      ▼
   Distributed Preprocessing & Deduplication (PII Masking)
                      │
                      ▼
   Columnar Storage Marts (Snappy Parquet)
      ├── cleaned_reviews.parquet
      ├── destination_profiles.parquet
      └── city_stats.parquet
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   NLP Sentiment Engine      TF-IDF & Bayesian Recommender
   (VADER Valence Scoring)   (Multi-Factor Match Score)
        │                           │
        └─────────────┬─────────────┘
                      ▼
   Inferential Hypothesis Testing (SciPy ANOVA / Kruskal / Spearman)
                      │
                      ▼
   Streamlit Multi-Page Interactive Dashboard (Plotly Visualizations)
```

### Technology Stack
- **Big Data Processing:** Apache Spark (PySpark 4.2.0)
- **Data Storage:** Apache Parquet (Columnar, Snappy compressed) via PyArrow
- **NLP & Text Mining:** Scikit-learn (TF-IDF vectorization), VADER sentiment intensity algorithm
- **Inferential Statistics:** SciPy, NumPy, Pandas
- **Frontend & Visualization:** Streamlit, Plotly Express & Graph Objects
- **Automated Testing:** Pytest

---

## 3. Repository Directory Structure

```
BDA_project/
├── app.py                         # Executive Dashboard entry point
├── pages/
│   ├── 1_Travel_Recommender.py    # Hybrid personalized recommender & baseline
│   ├── 2_Review_Analytics.py      # Sentiment distributions & NLP sandbox
│   ├── 3_Destination_Explorer.py  # Destination catalog search & profiles
│   ├── 4_Hypothesis_Testing.py    # ANOVA, Kruskal-Wallis, Spearman, Welch's t-test
│   ├── 5_Data_Quality.py          # Data profiling, missingness audit, JSON download
│   └── 6_About_Project.py         # Academic context, architecture & ethics
├── src/
│   ├── config.py                  # System paths, JVM flags & hyperparameters
│   ├── spark_session.py           # PySpark builder with modern JVM add-opens
│   ├── ingest.py                  # ZIP extraction & explicit StructType schema
│   ├── preprocessing.py           # Distributed cleaning, deduplication & PII masking
│   ├── exploratory_analysis.py    # Spark SQL aggregations & Bayesian ratings
│   ├── sentiment_analysis.py      # Offline VADER sentiment valence engine
│   ├── recommendation_engine.py   # TF-IDF cosine matching & hybrid scoring
│   ├── hypothesis_testing.py      # Statistical tests with effect sizes
│   ├── evaluation.py              # Catalog coverage & baseline divergence metrics
│   ├── data_access.py             # High-speed cached Parquet/JSON loader
│   └── utils.py                   # Logging, JSON encoder & text cleaning
├── scripts/
│   ├── validate_dataset.py        # Standalone schema and data-quality profiler
│   ├── run_pipeline.py            # Master end-to-end Big Data pipeline
│   └── export_demo_data.py        # Demo dataset generator for quick evaluation
├── data/
│   ├── raw/                       # archive.zip & Review_db.csv (516 MB)
│   ├── processed/                 # cleaned_reviews.parquet
│   └── analytics/                 # destination_profiles.parquet, city_stats.parquet
├── reports/
│   ├── data_quality_report.json   # Machine-readable schema & null audit
│   ├── analysis_summary.json      # National rating & city-level metrics
│   ├── model_evaluation.json      # Recommendation persona diagnostic results
│   └── hypothesis_testing_report.json # Statistical test statistics & p-values
├── docs/
│   ├── methodology.md             # Formal scientific methodology
│   ├── architecture.md            # Mermaid diagrams & component flow
│   └── findings_template.md       # Empirical observations & research outcomes
├── tests/
│   ├── conftest.py                # Pytest pythonpath configuration
│   ├── test_preprocessing.py      # Preprocessing & Bayesian rating tests
│   ├── test_recommendations.py    # TF-IDF matching & filter tests
│   └── test_statistics.py         # Hypothesis testing & sentiment tests
├── .streamlit/
│   └── config.toml                # Custom travel-tech color theme
├── requirements.txt               # Pinned, tested dependencies
├── .gitignore                     # Git ignore rules for large raw/parquet data
└── README.md                      # Complete execution manual
```

---

## 4. Prerequisites

- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python:** Python 3.10, 3.11, 3.12, or 3.13.
- **Java:** Java Runtime Environment (JRE) / JDK 11, 17, 21, or 23 (required for Apache Spark).
  - Verify Java installation in terminal: `java -version`

---

## 5. Step-by-Step Installation & Setup

### Step 1: Open Terminal in Workspace
In VS Code, press ``Ctrl + ` `` to open the integrated terminal.

### Step 2: (Optional) Set Up Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows Command Prompt (cmd)
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Dataset Placement
The project automatically checks `data/raw/Review_db.csv` or `data/raw/archive.zip` (and falls back to `~/Downloads/archive.zip`).
If needed, download the dataset from Kaggle:
[https://www.kaggle.com/datasets/ritvik1909/indian-places-to-visit-reviews-data](https://www.kaggle.com/datasets/ritvik1909/indian-places-to-visit-reviews-data)
and place `archive.zip` into the `data/raw/` directory.

---

## 6. Execution Commands

### 1. Validate Dataset Schema & Integrity
```bash
python scripts/validate_dataset.py --sample 0.05
```
*Outputs:* `reports/data_quality_report.json`

### 2. Run Master PySpark Big Data Pipeline
```bash
# Quick Development Mode (processes 5% sample - ~74,000+ reviews in ~45 seconds)
python scripts/run_pipeline.py --dev

# Full Big Data Mode (processes all 1.48+ million reviews)
python scripts/run_pipeline.py
```
*Outputs:* Cleaned Parquet tables in `data/processed/` and `data/analytics/`, plus JSON reports in `reports/`.

### 3. Run Automated Unit Tests
```bash
python -m pytest tests/
```
All unit tests validate text cleaning, Bayesian confidence mathematics, recommendation ranking, and statistical tests.

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
After running, open your web browser at:  
👉 **http://localhost:8501**

---

## 7. Application Walkthrough & Screenshots

### Page 1: Executive Dashboard (`app.py`)
- Real-time KPI metric tiles: Total Ingested Reviews, Unique Destinations, Indian Cities, National Mean Rating.
- Plotly interactive charts: Rating frequency distribution, top tourist hubs by volume, and Bayesian confidence scatter plot.

### Page 2: Travel Recommender (`pages/1_Travel_Recommender.py`)
- User filters: Preferred city, travel themes (Heritage, Nature, Spiritual, Beach, Adventure, etc.), minimum rating slider, minimum review threshold.
- Explainable recommendation cards displaying match percentages and transparent evidence statements.
- Baseline tab comparing personalized results against naive popularity rankings.

### Page 3: Review Analytics (`pages/2_Review_Analytics.py`)
- Sentiment valence distributions across star ratings.
- Review word-count dynamics illustrating complaint elaboration.
- Live Sentiment Analysis Sandbox: Test any tourist sentence in real-time.
- Anonymized visitor review excerpts (names scrubbed to protect privacy).

### Page 4: Destination Explorer (`pages/3_Destination_Explorer.py`)
- Searchable, filterable directory of thousands of Indian tourist places.
- Filter by city, rating, and minimum review threshold.

### Page 5: Hypothesis Testing (`pages/4_Hypothesis_Testing.py`)
- Dynamic selection between three inferential tests with live significance level adjustment.
- Explicit null/alternative hypothesis statements, assumption validations, $p$-values, effect sizes, and academic interpretations.

### Page 6: Data Quality & Profiling (`pages/5_Data_Quality.py`)
- Audit report showing missing values, deduplication counts, and PII masking.
- Direct download buttons for machine-readable JSON and CSV audit reports.

### Page 7: About Project (`pages/6_About_Project.py`)
- College project details, BDA lab curriculum alignment, and ethical data disclosures.

---

## 8. Troubleshooting Guide

### 1. Java / JVM Issues on Windows
If you encounter `JAVA_TOOL_OPTIONS` or JVM reflection warnings on Java 17/21/23, `src/config.py` and `src/spark_session.py` automatically inject `--add-opens` flags for modern JVM support. Ensure `java -version` returns a valid Java installation.

### 2. Hadoop `winutils.exe` on Windows
PySpark on Windows may emit warnings regarding `HADOOP_HOME` or `winutils.exe`. The pipeline includes an automatic PyArrow fallback writer, ensuring seamless Parquet persistence on Windows without needing external binaries.

### 3. Missing Dataset Error
If `archive.zip` or `Review_db.csv` is missing, `src/ingest.py` displays a clear error with the exact Kaggle download link.

---

## 9. Academic & Ethical Disclosures

1. **No Data Fabrication:** The public dataset contains an unpopulated `Date` column (100% null). In accordance with academic honesty, time-series curves and seasonality claims are scientifically omitted.
2. **Geographical Authenticity:** Exact GPS coordinates are not fabricated from place names.
3. **Privacy by Design:** Reviewer personal names from the raw dataset are strictly scrubbed to eliminate PII exposure.
4. **Transparent Recommender:** Recommendations are derived strictly from content semantics and Bayesian rating confidence without fabricating synthetic user click histories.
