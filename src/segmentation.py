import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def compute_customer_segments(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, KMeans, StandardScaler]:
    """
    Applies K-Means clustering on tenure and spend metrics to identify subscriber personas.
    """

    df_segmented = df.copy()
    rfm_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_segmented[rfm_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_segmented['Cluster'] = kmeans.fit_predict(scaled_features)

    # Aggregate persona characteristics by cluster
    summary =df_segmented.groupby('Cluster').agg(
        Accounts=('Cluster', 'count'),
        Avg_Tenure_Months=('tenure', 'mean'),
        Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
        Churn_Rate=('Churn', 'mean'),
        Revenue_At_Risk=('MonthlyCharges', lambda x: x[df_segmented.loc[x.index, 'Churn'] == 1].sum())
    ).reset_index()

    persona_mapping = {
        0: "At Risk month-to-month subscribers",
        1: "Loyal long-term subscribers",
        2: "Mid-tenure tier subscribers"
    }
    summary['Persona_Label'] = summary['Cluster'].map(persona_mapping)
    df_segmented['Persona_Label'] = df_segmented['Cluster'].map(persona_mapping)

    return df_segmented, summary, kmeans, scaler

if __name__ == "__main__":
    from src.data_loader import load_and_preprocess_data
    df = load_and_preprocess_data()
    _, summary, _, _ = compute_customer_segments(df)
    print("BEHAVIORAL COHORT SUMMARY")
    print(summary)