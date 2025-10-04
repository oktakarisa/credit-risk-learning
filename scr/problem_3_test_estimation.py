"""
Problem 3: Estimation on Test Data
Apply the baseline model to unseen test data and save Kaggle submission CSV.
Updates README.md with a preview of predictions.
"""

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate

def main():
    print("Running Problem 3 - Test Data Estimation")

    # Paths
    data_folder = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_folder, exist_ok=True)
    train_path = os.path.join(data_folder, 'application_train.csv')
    test_path = os.path.join(data_folder, 'application_test.csv')
    submission_path = os.path.join(data_folder, 'problem3_test_submission.csv')
    readme_path = os.path.join(os.path.dirname(__file__), '../README.md')

    # Load training data to rebuild baseline model
    try:
        df_train = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Training dataset not found at {train_path}. Please download application_train.csv from Kaggle.")
        return

    y_train = df_train['TARGET']
    X_train = df_train.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')

    # Encode categorical variables in training
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = LabelEncoder().fit_transform(X_train[col].astype(str))

    # Fill missing values in training
    X_train = X_train.fillna(X_train.median())

    # Train baseline model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Load test data
    try:
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Test dataset not found at {test_path}. Please download application_test.csv from Kaggle.")
        return

    X_test = df_test.drop(columns=['SK_ID_CURR'], errors='ignore')

    # Encode categorical variables in test
    for col in X_test.select_dtypes(include=['object']).columns:
        X_test[col] = LabelEncoder().fit_transform(X_test[col].astype(str))

    # Fill missing values in test
    X_test = X_test.fillna(X_train.median())  # Use training median to avoid data leakage

    # Predict probabilities
    y_test_pred_prob = model.predict_proba(X_test)[:,1]

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'SK_ID_CURR': df_test['SK_ID_CURR'],
        'TARGET': y_test_pred_prob
    })

    # Save submission CSV
    submission_df.to_csv(submission_path, index=False)
    print(f"Kaggle submission CSV saved to {submission_path}")

    # Update README.md with preview of first 10 rows
    preview_df = submission_df.head(10)
    with open(readme_path, 'a') as f:
        f.write("\n## Problem 3 – Test Data Estimation\n")
        f.write("Preview of first 10 predictions for submission:\n\n")
        f.write(tabulate(preview_df, headers='keys', tablefmt='github'))
        f.write("\n")

if __name__ == "__main__":
    main()
