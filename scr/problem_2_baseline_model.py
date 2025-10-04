"""
Problem 2: Baseline Model Learning and Verification
Load Home Credit train data, preprocess minimally, train a simple model,
evaluate using AUC, save a preview CSV of predictions, and update README.md.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

def main():
    print("Running Problem 2 - Baseline Model Learning")

    # Paths
    data_folder = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_folder, exist_ok=True)
    train_path = os.path.join(data_folder, 'application_train.csv')
    preview_csv_path = os.path.join(data_folder, 'problem2_baseline_preview.csv')
    readme_path = os.path.join(os.path.dirname(__file__), '../README.md')

    # Load dataset
    try:
        df = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Dataset not found at {train_path}. Please download application_train.csv from Kaggle.")
        return

    # Separate target
    if 'TARGET' not in df.columns:
        print("TARGET column missing in training data.")
        return
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')

    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill missing values with median
    X = X.fillna(X.median())

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train simple baseline model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred_prob = model.predict_proba(X_val)[:,1]

    # Evaluate using AUC
    auc_score = roc_auc_score(y_val, y_pred_prob)
    print(f"Baseline model AUC on validation set: {auc_score:.4f}")

    # Save preview CSV with first 10 predictions
    preview_df = pd.DataFrame({
        'SK_ID_CURR': df.loc[X_val.index, 'SK_ID_CURR'],
        'TARGET_PRED_PROB': y_pred_prob
    }).head(10)
    preview_df.to_csv(preview_csv_path, index=False)
    print(f"Preview of predictions saved to {preview_csv_path}")

    # Update README.md
    with open(readme_path, 'a') as f:
        f.write("\n## Problem 2 - Baseline Model Learning\n")
        f.write(f"Baseline Random Forest model AUC (validation): {auc_score:.4f}\n\n")
        f.write("Preview of first 10 predictions:\n\n")
        f.write(tabulate(preview_df, headers='keys', tablefmt='github'))
        f.write("\n")

if __name__ == "__main__":
    main()
