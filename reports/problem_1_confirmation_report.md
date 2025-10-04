# Problem 1: Confirmation of Home Credit Default Risk

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