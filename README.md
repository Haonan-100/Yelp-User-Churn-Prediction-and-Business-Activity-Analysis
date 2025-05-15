# Yelp User Churn Prediction & Business Activity Analysis

> **Project type:** End-to-end data-science case  
> **Data source:** [Yelp Open Dataset 2024](https://www.yelp.com/dataset)  
> **Focus:**  
> 1. **Predict** which users will churn (no review in the next 12 months).  
> 2. **Detect** businesses whose review activity and ratings have recently dropped.  
> 3. **Visualise** the findings in an interactive Tableau dashboard.

---

## Table of Contents
1. [Background & Goals](#background--goals)  
2. [Project Structure](#project-structure)  
3. [Workflow Overview](#workflow-overview)  
4. [Approach & Methods](#approach--methods)
5. [Data & Model Artefacts](#data--model-artefacts)
6. [Quick Start](#quick-start) 
7. [Tableau Dashboard](#tableau-dashboard)  
8. [Sample Results](#sample-results)  
9. [Credits](#credits)  

---

## Background & Goals
Customer retention is cheaper than acquisition; **predicting churn** lets a business intervene before a user disappears.  
At the same time, **business activity trends** reveal market dynamics—e.g., which categories or regions are losing popularity.

Using 8 M reviews, 2 M users and 200 K businesses from Yelp, we

* build an **XGBoost** model to score every user’s churn risk;  
* flag **“quiet & declining”** businesses: no new reviews in the past 12 months **and** a ≥ 0.3-star drop in recent ratings;  
* export analysis results to **Tableau dashboards** for interactive exploration.

---

## Project Structure
```

├── README.md
├── data
│   ├── raw/                       # 5 JSON files from Yelp
│   └── processed/
├── models/
│   ├── churn\_xgb\_model.pkl        # trained XGBoost model
│   └── churn\_xgb\_model.json       # feature list / params
├── notebooks/
│   ├── 1\_data\_exploration.ipynb
│   ├── 2\_user\_churn\_features.ipynb
│   ├── 3\_model\_training.ipynb
│   ├── 4\_business\_analysis.ipynb
│   └── 5\_visual\_prep.ipynb
├── src/                           
│   ├── data\_preprocessing.py
│   ├── feature\_engineering.py
│   ├── model.py
│   └── visualization\_prep.py
├── reports/                       # screenshots, diagrams
│   ├── Business Churn Distribution by Primary Category.png
│   ├── Churn Rate Trend by Year of Business Establishment.png
│   ├── User Engagement and Retention Rate dashboard.png
│   └── User retention rate by state.png
├── tableau/                       # Tableau dashboards (twbx)
│   ├── Business Churn Distribution by Primary Category.twbx
│   ├── Churn Rate Trend by Year of Business Establishment.twbx
│   ├── User Engagement and Retention Rate.twbx
│   └── User\_retention\_rate\_by\_state.twbx
└── requirements.txt

````

---

## Workflow Overview
| Step | Notebook / Script | Key Outputs |
|------|-------------------|-------------|
| **1. EDA & Cleaning** | `1_data_exploration.ipynb` | Initial data overview |
| **2. Feature Engineering** | `2_user_churn_features.ipynb` | `user_churn_features.csv` |
| **3. Model Training** | `3_model_training.ipynb` | `churn_xgb_model.pkl` |
| **4. Business Analysis** | `4_business_analysis.ipynb` | `business_churn_profile.csv` |
| **5. Viz Data Prep** | `5_visual_prep.ipynb` | Final `.csv` files |
| **6. Tableau** | `.twbx` files in `/tableau/` | 4 dashboards (see below) |

---
## Approach & Methods
This section answers two questions:

* **Why** did we choose a particular technique or rule?  
* **How** was it implemented in code (see notebooks for line-by-line detail)?

| Phase | Why | How (key code / logic) |
|-------|-----|------------------------|
| **Exploratory Data Analysis** | Validate data volume & date ranges, detect schema issues early—prevents downstream feature errors. | `1_data_exploration.ipynb` loads 1 M-row samples with `chunksize`, uses `df.info()` and `describe()` for quick sanity checks. |
| **User Churn Label** | A “full-year silence” threshold (no reviews in 365 days) mirrors real-world KPIs used by subscription platforms. It balances recall (catch true churners) with precision. | In `2_user_churn_features.ipynb` we compute `last_review_date` per `user_id`, compare to `cutoff_date = max(review.date) - 365 days`, then set `churn_label = 1` if beyond cutoff. |
| **Feature Engineering** | Mix **behavioural** (review_count, recency), **reputation** (votes, elite_years) and **social** (friends, fans) signals to capture multiple churn drivers (activity, satisfaction, network). | Functions in `feature_engineering.py`: <br>`build_activity_features()`, `aggregate_vote_features()`, `compute_social_metrics()`. Merged into `user_churn_features.csv` (≈ 35 cols). |
| **Model Choice – XGBoost** | Tree ensembles handle non-linear interactions and missing values with minimal preprocessing; XGBoost is fast on 2 M users and supports class-imbalance (`scale_pos_weight`). | `3_model_training.ipynb` uses `XGBClassifier` (max_depth = 6, n_estimators = 300, learning_rate = 0.05). Early-stopping on a 20 % validation split, AUC ≈ 0.80. |
| **Business “Quiet & Declining” Rule** | Simple, explainable heuristic > black-box model for first pass. Combines **no new reviews** (volume drop) with **≥ 0.3-star decline** (quality drop). | In `4_business_analysis.ipynb`: group reviews per `business_id`; flag `no_review_recent = last_review < cutoff`; compute `star_recent` (last N reviews) vs. `star_hist` (all-time). |
| **Aggregation for Tableau** | Pre-aggregating to CSV avoids heavy SQL/LOD calcs in Tableau → faster viz and smaller `.twbx`. | `5_visual_prep.ipynb` writes six CSVs (`business_churn_by_category.csv`, `user_activity_retention.csv`, …). Each file has only the dimensions/measures needed for one chart. |

**Key result metrics**

* **Model AUC:** 0.80  
* **Top predictors:** review_count, avg_stars, useful_votes, last_review_recency, account_age_days  
* **Business insight:** Food, Beauty, and Shopping account for 60 % of detected “quiet” businesses; venues opened before 2015 show 2-3× higher churn rate.

For full implementation details, see inline comments in each notebook or the functions within `/src`.

---

## Data & Model Artefacts

### Processed CSV files
| File | Tableau Component |
|------|-------------------|
| **business_churn_by_category.csv** | Bar – Churned businesses by category |
| **business_churn_by_open_year.csv** | Line – Churn rate by business open year |
| **user_activity_retention.csv** | Bar – Review activity vs. retention rate |
| **user_churn_by_state.csv** | Map – User churn by state |

### Model files
- `models/churn_xgb_model.pkl` – Trained XGBoost classifier  
- `models/churn_xgb_model.json` – Feature list and metadata  

---

## Quick Start

```bash
# 1 – clone repo & install dependencies
git clone https://github.com/your-handle/yelp-churn.git
cd yelp-churn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2 – place raw Yelp JSON files
#     inside data/raw/

# 3 – run notebooks 1 → 5 in order

# 4 – (optional) batch-score new users
python src/model.py \
  --model models/churn_xgb_model.pkl \
  --input data/processed/user_churn_features.csv \
  --output data/processed/user_scores.csv
````

---

## Tableau Dashboard

> Local `.twbx` files available in the `/tableau/` folder. Open using Tableau Desktop (or Tableau Reader).

### 4 Completed Dashboards:

| Chart                                | Workbook                                                  | PNG                                                                                       |
| ------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Business Churn by Category**       | `Business Churn Distribution by Primary Category.twbx`    | ![Category](reports/Business%20Churn%20Distribution%20by%20Primary%20Category.png)        |
| **Churn Rate by Business Open Year** | `Churn Rate Trend by Year of Business Establishment.twbx` | ![OpenYear](reports/Churn%20Rate%20Trend%20by%20Year%20of%20Business%20Establishment.png) |
| **User Engagement vs. Retention**    | `User Engagement and Retention Rate.twbx`                 | ![Engagement](reports/User%20Engagement%20and%20Retention%20Rate%20dashboard.png)         |
| **User Churn Rate by State**         | `User_retention_rate_by_state.twbx`                       | ![StateMap](reports/User%20retention%20rate%20by%20state.png)                             |

---

## Sample Results

* Businesses in **Food, Beauty**, and **Shopping** categories show higher churn rates.
* Older businesses (opened before 2015) have higher review drop-off.
* **Highly active users** are far more likely to remain active.
* States like **California, Texas, and Arizona** exhibit above-average churn.


---

## Credits

* **Data** – [Yelp Open Dataset 2024](https://www.yelp.com/dataset)

