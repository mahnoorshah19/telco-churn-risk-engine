import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_and_preprocess_data
from src.segmentation import compute_customer_segments
from src.model import build_training_pipeline
from src.evaluate import optimize_decision_threshold

st.set_page_config(page_title="Telco Customer Retention Engine", layout="wide")

st.title("Telco Customer Retention & Financial Risk Engine")
st.markdown("Quantifying subscriber churn risk and optimizing retention intervention ROI.")

@st.cache_data
def get_processed_data():
    return load_and_preprocess_data()

@st.cache_resource
def get_model_and_segments(df):
    df_seg, summary, _, _ = compute_customer_segments(df)
    pipeline, X_test, y_test = build_training_pipeline(df)
    return df_seg, summary, pipeline, X_test, y_test

df = get_processed_data()
df_seg, summary, pipeline, X_test, y_test = get_model_and_segments(df)

# Sidebar Parameters
st.sidebar.header("Retention Unit Economics")
perk_cost = st.sidebar.slider("Intervention Cost per Subscriber ($)", 5, 50, 15, 5)
success_rate = st.sidebar.slider("Intervention Success Rate (%)", 10, 80, 45, 5) / 100.0
retention_window = st.sidebar.selectbox("Customer Retention Window (Months)", [3, 6, 12], index=1)

# Tab Navigation
tab1, tab2, tab3 = st.tabs(["Behavioral Personas", "Financial Threshold Optimizer", "Live Risk Scoring"])

with tab1:
    st.subheader("Subscriber Segmentation (K-Means)")
    st.dataframe(summary.style.format({
        'Avg_Tenure_Months': '{:.1f}',
        'Avg_Monthly_Charges': '${:.2f}',
        'Churn_Rate': '{:.1%}',
        'Revenue_At_Risk': '${:,.2f}'
    }), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary['Persona_Label'], summary['Churn_Rate'], color=['#d95f02', '#1b9e77', '#7570b3'])
    ax.set_ylabel("Churn Rate")
    ax.set_title("Attrition Probability by Behavioral Persona")
    plt.xticks(rotation=15, ha='right')
    st.pyplot(fig)

with tab2:
    st.subheader("Cost-Benefit Threshold Calibration")
    y_pred_probs = pipeline.predict_proba(X_test)[:, 1]
    opt_thresh, max_saving, thresholds, savings = optimize_decision_threshold(
        y_test, y_pred_probs, X_test, perk_cost, success_rate, retention_window
    )
    
    col1, col2 = st.columns(2)
    col1.metric("Optimal Probability Cutoff", f"{opt_thresh:.2f}", delta=f"{opt_thresh - 0.50:.2f} vs Default (0.50)")
    col2.metric(f"Net Revenue Preserved ({retention_window} Mo)", f"${max_saving:,.2f}")
    
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(thresholds, savings, color='#02818a', lw=2.5, label='Net Savings Curve')
    ax2.axvline(opt_thresh, color='#e31a1c', linestyle='--', label=f'Optimal Cutoff ({opt_thresh:.2f})')
    ax2.axvline(0.50, color='#636363', linestyle=':', label='Default Cutoff (0.50)')
    ax2.set_xlabel("Churn Probability Threshold")
    ax2.set_ylabel("Net Savings ($)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

with tab3:
    st.subheader("Single Subscriber Inference")
    with st.form("inference_form"):
        col_a, col_b, col_c = st.columns(3)
        tenure = col_a.number_input("Tenure (Months)", min_value=0, max_value=72, value=4)
        monthly_charges = col_b.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0)
        total_charges = col_c.number_input("Total Charges ($)", min_value=0.0, max_value=8500.0, value=300.0)
        
        contract = col_a.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = col_b.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment = col_c.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        submitted = st.form_submit_button("Score Subscriber Risk")
        
        if submitted:
            # Build sample payload matching schema
            sample = pd.DataFrame([{
                'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
                'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
                'InternetService': internet, 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
                'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
                'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': 'Yes',
                'PaymentMethod': payment, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
            }])
            
            prob = pipeline.predict_proba(sample)[0][1]
            risk_tier = "CRITICAL RISK" if prob >= opt_thresh else "LOW RISK"
            
            st.write(f"**Predicted Churn Probability:** `{prob:.2%}`")
            st.write(f"**Status:** `{risk_tier}`")
            if prob >= opt_thresh:
                st.warning(f"Exceeds target intervention threshold of {opt_thresh:.2f}. Recommend issuing customer perk voucher (${perk_cost}).")
            else:
                st.success("Subscriber risk is below target intervention threshold. No retention perk needed.")