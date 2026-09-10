import os
import pandas as pd

def load_and_preprocess_data(filepath: str = "data/Telco-Customer-Churn.csv") -> pd.DataFrame:
    """
    Loads Telco Customer Churn data and applies data preprocessing steps.
    - Resolves whitespac-induced NaNs in TotalCharges.
    - Encodes Churn to binary integers (0, 1).
    - Removes uniformative identifiers.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    # Resolve whitespace-induced NaNs in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0.0)

    # Map target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Remove unique alphanumeric customer IDs [cite: 1]
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    return df

if __name__ == "__main__":
    # Example usage
    data = load_and_preprocess_data()
    print(f"Data loaded and preprocessed. Shape: {data.shape}")