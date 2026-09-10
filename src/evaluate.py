import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def optimize_decision_threshold(y_true: pd.Series, y_pred_probs: np.ndarray, test_features: pd.DataFrame, 
                                perk_cost: float = 15.0, success_rate: float = 0.45, retention_months: int = 6):
    """
    Financial Cost-Benefit Threshold Tuning:
    - Cost: perk_cost for every user targeted.
    - Benefit: MonthlyCharges * retention_months saved for True Positives * success_rate.
    """

    thresholds = np.linspace(0.05, 0.95, 91)
    net_dollar_savings = []

    monthly_revenue = test_features.loc[y_true.index, 'MonthlyCharges'].values
    y_true_array = y_true.values

    for thresh in thresholds:
        targeted_mask = (y_pred_probs >= thresh)
        
        # Total cost of customer engagement
        cost = np.sum(targeted_mask) * perk_cost
        
        # Realized retained revenue (True Positives)
        retained_mask = (targeted_mask & (y_true_array == 1))
        retained_revenue = np.sum(monthly_revenue[retained_mask] * retention_months * success_rate)
        
        net_dollar_savings.append(retained_revenue - cost)

    best_idx = np.argmax(net_dollar_savings)
    optimal_threshold = thresholds[best_idx]
    max_saving = net_dollar_savings[best_idx]

    return optimal_threshold, max_saving, thresholds, net_dollar_savings

def run_evaluation_suite(pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred_probs = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_probs)
    
    opt_thresh, max_saving, _, _ = optimize_decision_threshold(y_test, y_pred_probs, X_test)
    y_pred_tuned = (y_pred_probs >= opt_thresh).astype(int)
    
    print("* MODEL EVALUATION *")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Optimal Financial Threshold: {opt_thresh:.2f}")
    print(f"Estimated Net Dollar Retained: ${max_saving:,.2f}")
    print("\nClassification Report (At Optimal Threshold):")
    print(classification_report(y_test, y_pred_tuned))

if __name__ == "__main__":
    import joblib
    from src.data_loader import load_and_preprocess_data
    from src.model import build_training_pipeline
    
    df = load_and_preprocess_data()
    pipeline, X_test, y_test = build_training_pipeline(df)
    run_evaluation_suite(pipeline, X_test, y_test)