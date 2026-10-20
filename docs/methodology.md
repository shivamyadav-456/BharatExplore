# BharatExplore: Experimental Methodology & System Design

**Subject:** Big Data Analysis Lab (CSL702), Experiment 11  
**Curriculum:** BE Computer Engineering, Semester VII, Mumbai University  
**Project:** BharatExplore — Big Data-Powered Indian Travel Recommendation System  

---

## 1. Problem Formulation
In modern tourism analytics, traveler decision-making suffers from two primary bottlenecks:
1. **Information Overload and Uncalibrated Ratings:** Popular platforms feature thousands of attractions, but raw star ratings suffer from small-sample variance (e.g., an attraction with one 5-star review artificially outranks an internationally acclaimed destination with 10,000 reviews averaging 4.7 stars).
2. **Generic Keyword Recommendations:** Traditional filter systems rely on static category tags rather than dynamic traveler sentiment and semantic review context.

**Objective:** Build an end-to-end Big Data processing, analytics, and recommendation system utilizing Apache Spark (PySpark), distributed Parquet storage, natural language processing, inferential hypothesis testing, and an explainable hybrid recommendation engine to assist tourists exploring India.

---

## 2. Dataset Acquisition and Source
* **Source:** *Indian Places to Visit Reviews Data* published by Ritvik1909 on Kaggle.
* **Volume:** Over 1.48 million raw tourist review records across 14,400+ destinations in 1,700+ Indian cities and hubs (~541 MB uncompressed CSV).
* **Source Schema:** `City`, `Place`, `Review`, `Rating`, `Name`, `Date`, `Raw_Review`.

---

## 3. Dataset Profiling & Initial Schema Validation
A dedicated PySpark profiling script (`scripts/validate_dataset.py`) audits:
- **Null Value Analysis:** Systematically counts empty and null records across all columns.
  - *Critical Empirical Finding:* The `Date` column in the public dataset is **100% empty (null)** across all records. Consequently, temporal trends, monthly seasonality, and year-over-year growth analyses are scientifically withheld to prevent data fabrication.
- **Data Protection & PII:** The `Name` column contains user handles. All personal identifiers are masked and scrubbed to `Verified Visitor` before analytical aggregation.
- **Rating Scale Validation:** Ratings are audited to strictly conform to numeric values between 1.0 and 5.0.

---

## 4. PySpark Data Preprocessing & Cleaning Pipeline
1. **Normalization:** Cities and Destination names are trimmed, cleaned of redundant whitespace, and converted to standardized casing (`initcap`).
2. **Review Text Sanitization:** `Review` (cleaned token stream) and `Raw_Review` (punctuated original text) are trimmed, and records lacking both texts are discarded.
3. **Deduplication Rule:** Records with identical tuples of `(City, Place, Review, Rating)` are filtered, and the exact count of dropped records is recorded in the data quality audit report.
4. **Length Feature Engineering:** Computes character length and whitespace-tokenized word count for downstream correlation studies.
5. **Columnar Parquet Storage:** Cleaned records are written to columnar Parquet format using Snappy compression (`data/processed/cleaned_reviews.parquet`).

---

## 5. Exploratory Data Analysis (EDA)
Distributed Spark aggregations compute:
- Global rating distribution, mean rating, and national rating variance.
- City-wise review volume, destination density, and city mean ratings.
- Review word count distribution segmented by star rating.
- Bayesian confidence ratings calculated using the empirical Bayes (IMDb) weighting formula:
  $$WR = \frac{v}{v + m} \cdot R + \frac{m}{v + m} \cdot C$$
  where $v$ is destination review count, $m$ is the minimum threshold (default 25–30), $R$ is destination average rating, and $C$ is the global mean rating.

---

## 6. Sentiment Analysis & Review NLP
- **Methodology:** Implements a rule-based VADER-compatible sentiment engine tailored for tourism, hospitality, and geographical features.
- **Contextual Processing:** Handles booster words (e.g., *extremely*, *super*), diminishing words (*barely*, *slightly*), and negative context inversion across 3-word preceding windows.
- **Normalization:** Standard VADER compound valence formula:
  $$\text{compound} = \frac{\sum s}{\sqrt{(\sum s)^2 + \alpha}}, \quad \alpha = 15$$
  producing continuous scores in $[-1.0, 1.0]$.
- **Offline Independence:** Operates completely offline without external network dependencies, ensuring 100% reproducibility in academic lab environments.

---

## 7. Recommendation Engine Architecture
- **Model Type:** Content-Based Hybrid with Bayesian Calibration.
- **Profile Construction:** For each destination, aggregated review texts form a consolidated destination document.
- **Feature Extraction:** Sublinear TF-IDF vectorization (unigrams and bigrams, stopword elimination, $L_2$ normalization).
- **Multi-Factor Composite Score:**
  $$\text{Score} = w_{\text{sim}} \cdot S_{\text{sim}} + w_{\text{rating}} \cdot S_{\text{rating}} + w_{\text{sent}} \cdot S_{\text{sent}} + w_{\text{pop}} \cdot S_{\text{pop}}$$
- **Explainability:** Each recommendation includes an evidence-based rationale citing the matched travel interest, star rating, review volume confidence, and sentiment profile.

---

## 8. Inferential Hypothesis Testing
Three rigorous statistical hypothesis tests address empirical travel behavior:
1. **City-Wise Rating Differences:**
   - $H_0$: $\mu_1 = \mu_2 = \dots = \mu_k$ (Mean tourist ratings are identical across major tourist cities).
   - $H_1$: At least one city's mean rating differs significantly.
   - *Assumptions:* Levene's test for equality of variance ($W$). When $p < 0.05$, non-parametric Kruskal-Wallis $H$-test is prioritized alongside One-Way ANOVA and Eta-Squared ($\eta^2$) effect size.
2. **Review Length vs Rating Association:**
   - $H_0$: True Spearman correlation $\rho = 0$.
   - $H_1$: $\rho \neq 0$.
   - Investigates the behavioral tendency of dissatisfied visitors to author longer explanatory feedback.
3. **Sentiment Valence Divergence (5-Star vs 1-Star):**
   - $H_0$: $\mu_{\text{sentiment, 5-star}} \le \mu_{\text{sentiment, 1-star}}$.
   - $H_1$: $\mu_{\text{sentiment, 5-star}} > \mu_{\text{sentiment, 1-star}}$.
   - Evaluated via Welch's two-sample unequal-variance $t$-test, Mann-Whitney $U$ test, and Cohen's $d$ effect size.

---

## 9. Evaluation Strategy
Because this public dataset contains anonymous unlinked reviews without persistent user session history, recommendation evaluation is conducted via:
- **Catalog Coverage:** Percentage of the destination catalog reachable across benchmark travel personas.
- **Persona Relevance:** Thematic alignment score of recommendations.
- **Baseline Divergence:** Quantifying novelty beyond a naive popularity/rating baseline ranking.

---

## 10. Ethical Considerations & Scientific Rigor
- **No Fabricated Data:** Missing dates and GPS coordinates are honestly disclosed rather than invented.
- **No Fabricated Collaborative Filtering:** User-item matrix factorization is not falsely claimed on anonymous unlinked review records.
- **Academic Transparency:** Full reproducibility via version-controlled PySpark scripts and self-contained test suites.
