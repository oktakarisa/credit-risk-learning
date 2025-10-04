#!/usr/bin/env python3
# main.py
# Run all problem scripts for credit-risk-learning assignment

import sys
import os

# Add scr folder to sys.path so we can import scripts
sys.path.append(os.path.join(os.path.dirname(__file__), "scr"))

# Import each problem script (make sure each script has a main() function)
try:
    import problem_1_confirmation
    import problem_2_baseline_model
    import problem_3_test_estimation
    import problem_4_feature_engineering
except ImportError as e:
    print("Error importing problem scripts:", e)
    sys.exit(1)

def main():
    print("Running Problem 1: Confirmation of Competition Contents")
    problem_1_confirmation.main()
    print("\nRunning Problem 2: Baseline Model Learning and Verification")
    problem_2_baseline_model.main()
    print("\nRunning Problem 3: Test Data Estimation and Submission")
    problem_3_test_estimation.main()
    print("\nRunning Problem 4: Feature Engineering and Validation")
    problem_4_feature_engineering.main()
    print("\nAll scripts executed successfully.")

if __name__ == "__main__":
    main()
