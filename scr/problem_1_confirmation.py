"""
Problem 1: Confirmation of Home Credit Default Risk Competition
Generates a Markdown report summarizing the competition overview, prediction target,
submission format, and evaluation metric. Saves report in the reports folder and updates README.md.
"""

import os

def main():
    print("Running Problem 1 - Competition Confirmation")

    # Paths
    repo_root = os.path.dirname(os.path.dirname(__file__))
    reports_folder = os.path.join(repo_root, 'reports')
    os.makedirs(reports_folder, exist_ok=True)
    report_path = os.path.join(reports_folder, 'problem_1_confirmation_report.md')
    readme_path = os.path.join(repo_root, 'README.md')

    # Competition summary text
    competition_summary = """
# Problem 1 – Confirmation of Home Credit Default Risk

**Goal:** Understand the Kaggle competition fully before touching the data.

**Competition Overview:**  
Home Credit aims to predict whether applicants can repay loans using alternative data. Many applicants lack formal credit histories. Kagglers are challenged to help Home Credit use this data to make better lending decisions.

**Prediction Target:**  
- `TARGET` = 1 if client is likely to default  
- `TARGET` = 0 if client is likely to repay

**Submission Format:**  
- CSV file with header: `SK_ID_CURR,TARGET`  
- Example:  
SK_ID_CURR,TARGET
100001,0.1
100005,0.9
100013,0.2

**Evaluation Metric:**  
- Area Under the Receiver Operating Characteristic Curve (AUC)

**Why Important:**  
Understanding these details ensures correct submission format and evaluation expectations.
"""

    # Save report
    with open(report_path, 'w') as f:
        f.write(competition_summary.strip())
    print(f"Report saved to {report_path}")

    # Append short summary to README.md
    readme_summary = """
## Problem 1 – Competition Confirmation
Full report available in [reports/problem_1_confirmation_report.md](reports/problem_1_confirmation_report.md)
"""
    with open(readme_path, 'a') as f:
        f.write(readme_summary.strip() + "\n")
    print("README.md updated with Problem 1 summary.")

if __name__ == "__main__":
    main()
