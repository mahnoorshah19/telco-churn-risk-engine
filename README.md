# Telco Customer Retention & Revenue Optimization Engine

An end-to-end commercial analytics and machine learning system that transforms subscriber data into actionable customer retention strategies. The platform integrates unsupervised persona segmentation, supervised attrition modeling, and decision-threshold optimization based on unit economics.

## Executive Summary
- **Baseline Attrition Risk:** Over 26.5% of the subscriber network churns during tracking cycles, representing substantial lost monthly recurring revenue.
- **Onboarding Vulnerability:** Subscriber attrition is heavily concentrated in the first 1–5 months of contract creation, while accounts active beyond 20 months show significantly higher retention stability.
- **Unit Economic Optimization:** By moving from the default `0.50` decision threshold to an empirically tuned cutoff based on customer lifetime value and perk intervention costs, the business maximizes net preserved revenue.

## System Architecture & Features
1. **Data Ingestion & Hygiene (`src/data_loader.py`):** Automatically cleans whitespace-induced `NaN` values in `TotalCharges` for zero-tenure accounts, maps binary targets, and strips uninformative hash keys.
2. **Behavioral Clustering (`src/segmentation.py`):** Utilizes K-Means clustering across tenure, monthly spend, and total spend to segment subscribers into 3 distinct operational cohorts:
   - *At-Risk Month-to-Month Onboarders* (High churn, short tenure)
   - *Established High-Value Loyalists* (Low churn, long tenure)
   - *Mid-Tenure Tier Subscribers* (Moderate churn)
3. **Leakage-Free Predictive Engine (`src/model.py`):** Combines `StandardScaler` and `OneHotEncoder` within a unified Scikit-Learn `ColumnTransformer` and `RandomForestClassifier` with balanced class weights.
4. **Financial Threshold Calibration (`src/evaluate.py`):** Computes exact net retention revenue using an expected value loss matrix rather than raw accuracy.
5. **Interactive Dashboard (`app.py`):** Streamlit web application providing live retention economics modeling, cohort inspections, and single-subscriber risk scoring.

## Project Structure
```text
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── customer_risk_engine.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── segmentation.py
│   ├── model.py
│   └── evaluate.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md`