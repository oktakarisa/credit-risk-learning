"""
Problem 4: Feature Engineering
Create new features and preprocessing strategies to improve AUC.
Train several model patterns, record AUC scores, and append summary to README.md.
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

def main():
    print("Running Problem 4 - Feature Engineering")

    # Paths
    data_folder = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_folder, exist_ok=True)
    train_path = os.path.join(data_folder, 'application_train.csv')
    readme_path = os.path.join(os.path.dirname(__file__), '../README.md')

    # Load training data
    try:
        df = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Training dataset not found at {train_path}. Please download application_train.csv from Kaggle.")
        return

    # Separate target
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')

    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median())

    # Add engineered features (Pattern 1–5)
    X_patterns = {}
    
    # Pattern 1: Baseline
    X_patterns['Baseline'] = X.copy()

    # Pattern 2: Income-to-Credit Ratio
    X2 = X.copy()
    if 'AMT_INCOME_TOTAL' in X2.columns and 'AMT_CREDIT' in X2.columns:
        X2['INCOME_CREDIT_RATIO'] = X2['AMT_INCOME_TOTAL'] / (X2['AMT_CREDIT'] + 1)
    X_patterns['Income_Credit_Ratio'] = X2

    # Pattern 3: Log transformations
    X3 = X.copy()
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']:
        if col in X3.columns:
            X3[col + '_LOG'] = np.log1p(X3[col])
    X_patterns['Log_Transform'] = X3

    # Pattern 4: Age-based (if DAYS_BIRTH exists)
    X4 = X.copy()
    if 'DAYS_BIRTH' in X4.columns:
        X4['AGE'] = abs(X4['DAYS_BIRTH']) / 365
        X4['AGE_BIN'] = pd.cut(X4['AGE'], bins=[20, 30, 40, 50, 60, 70, 80], labels=False)
    X_patterns['Age_Features'] = X4

    # Pattern 5: Combined ratio
    X5 = X.copy()
    if all(col in X5.columns for col in ['AMT_INCOME_TOTAL', 'AMT_ANNUITY']):
        X5['INCOME_ANNUITY_RATIO'] = X5['AMT_INCOME_TOTAL'] / (X5['AMT_ANNUITY'] + 1)
    X_patterns['Income_Annuity_Ratio'] = X5

    # Evaluate each pattern
    results = []
    for name, Xp in X_patterns.items():
        X_train, X_val, y_train, y_val = train_test_split(Xp, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        results.append({'Pattern': name, 'AUC_Score': round(auc, 4)})
        print(f"{name} AUC: {auc:.4f}")

    results_df = pd.DataFrame(results)

    # Save preview of results
    preview_path = os.path.join(data_folder, 'problem4_feature_engineering_results.csv')
    results_df.to_csv(preview_path, index=False)
    print(f"Feature engineering results saved to {preview_path}")

    # Update README.md with summary
    with open(readme_path, 'a') as f:
        f.write("\n## Problem 4 – Feature Engineering\n")
        f.write("Comparison of 5 feature patterns based on validation AUC:\n\n")
        f.write(tabulate(results_df, headers='keys', tablefmt='github'))
        f.write("\n")

if __name__ == "__main__":
    main()
