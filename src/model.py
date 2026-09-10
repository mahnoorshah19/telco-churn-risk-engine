import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def training_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Constructs a leakage-free preprocessing and modeling pipeline.
    """

    # Define features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth = 8,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
            ))
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model_pipeline.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/retention_pipeline.joblib')

    return model_pipeline, X_test, y_test

if __name__ == "__main__":
    from src.data_loader import load_and_preprocess_data
    df = load_and_preprocess_data()
    model_pipeline, X_test, y_test = training_pipeline(df)
    print(f"Pipeline trained successfully and exported to models/retention_pipeline.joblib")