"""
Problem 4: Advanced Feature Engineering (IMPROVED VERSION)
Creates sophisticated features from multiple data sources, tests 10+ patterns,
uses LightGBM for better performance, and documents comprehensive results.
"""

import pandas as pd 
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_train(data_folder):
    """Load and basic preprocessing of training data"""
    train_path = os.path.join(data_folder, 'application_train.csv')
    df = pd.read_csv(train_path)
    return df

def aggregate_bureau(data_folder):
    """Create aggregated features from bureau and bureau_balance"""
    print("Processing bureau data...")
    bureau = pd.read_csv(os.path.join(data_folder, 'bureau.csv'))
    
    # Aggregate bureau data by SK_ID_CURR
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'CREDIT_TYPE': 'count'
    })
    
    # Flatten column names
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
    
    # Additional features
    bureau_agg['BUREAU_CREDIT_ACTIVE_COUNT'] = bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].apply(
        lambda x: (x == 'Active').sum()
    )
    bureau_agg['BUREAU_CREDIT_CLOSED_COUNT'] = bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].apply(
        lambda x: (x == 'Closed').sum()
    )
    
    return bureau_agg.reset_index()

def aggregate_previous_application(data_folder):
    """Create aggregated features from previous_application"""
    print("Processing previous application data...")
    prev = pd.read_csv(os.path.join(data_folder, 'previous_application.csv'))
    
    # Aggregate previous applications
    prev_agg = prev.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum']
    })
    
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    
    # Application status counts
    prev_agg['PREV_APPROVED_COUNT'] = prev.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].apply(
        lambda x: (x == 'Approved').sum()
    )
    prev_agg['PREV_REFUSED_COUNT'] = prev.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].apply(
        lambda x: (x == 'Refused').sum()
    )
    prev_agg['PREV_TOTAL_APPLICATIONS'] = prev.groupby('SK_ID_CURR').size()
    
    return prev_agg.reset_index()

def aggregate_installments(data_folder):
    """Create aggregated features from installments_payments"""
    print("Processing installments data...")
    inst = pd.read_csv(os.path.join(data_folder, 'installments_payments.csv'))
    
    # Payment difference and delay
    inst['PAYMENT_DIFF'] = inst['AMT_PAYMENT'] - inst['AMT_INSTALMENT']
    inst['PAYMENT_DELAY'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['LATE_PAYMENT'] = (inst['PAYMENT_DELAY'] > 0).astype(int)
    
    inst_agg = inst.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['max', 'mean'],
        'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum'],
        'PAYMENT_DELAY': ['min', 'max', 'mean'],
        'LATE_PAYMENT': ['sum', 'mean'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum']
    })
    
    inst_agg.columns = ['INST_' + '_'.join(col).upper() for col in inst_agg.columns]
    
    return inst_agg.reset_index()

def aggregate_credit_card(data_folder):
    """Create aggregated features from credit_card_balance"""
    print("Processing credit card data...")
    cc = pd.read_csv(os.path.join(data_folder, 'credit_card_balance.csv'))
    
    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'AMT_BALANCE': ['min', 'max', 'mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    })
    
    cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
    
    # Credit utilization
    cc['CREDIT_UTILIZATION'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    cc_agg['CC_CREDIT_UTILIZATION_MEAN'] = cc.groupby('SK_ID_CURR')['CREDIT_UTILIZATION'].mean()
    
    return cc_agg.reset_index()

def aggregate_pos_cash(data_folder):
    """Create aggregated features from POS_CASH_balance"""
    print("Processing POS cash data...")
    pos = pd.read_csv(os.path.join(data_folder, 'POS_CASH_balance.csv'))
    
    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'CNT_INSTALMENT': ['min', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    })
    
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    
    return pos_agg.reset_index()

def create_domain_features(df):
    """Create domain-specific financial features"""
    df_new = df.copy()
    
    # Income ratios
    df_new['INCOME_CREDIT_RATIO'] = df_new['AMT_INCOME_TOTAL'] / (df_new['AMT_CREDIT'] + 1)
    df_new['INCOME_ANNUITY_RATIO'] = df_new['AMT_INCOME_TOTAL'] / (df_new['AMT_ANNUITY'] + 1)
    df_new['CREDIT_ANNUITY_RATIO'] = df_new['AMT_CREDIT'] / (df_new['AMT_ANNUITY'] + 1)
    df_new['CREDIT_GOODS_RATIO'] = df_new['AMT_CREDIT'] / (df_new['AMT_GOODS_PRICE'] + 1)
    
    # Age and employment features
    df_new['AGE_YEARS'] = -df_new['DAYS_BIRTH'] / 365
    df_new['EMPLOYMENT_YEARS'] = -df_new['DAYS_EMPLOYED'] / 365
    df_new['EMPLOYMENT_AGE_RATIO'] = df_new['EMPLOYMENT_YEARS'] / (df_new['AGE_YEARS'] + 1)
    
    # Document flags
    doc_cols = [col for col in df_new.columns if 'FLAG_DOCUMENT' in col]
    df_new['DOCUMENT_COUNT'] = df_new[doc_cols].sum(axis=1)
    
    # Contact information
    df_new['CONTACT_INFO_AVAILABLE'] = (
        df_new['FLAG_MOBIL'] + df_new['FLAG_EMP_PHONE'] + 
        df_new['FLAG_WORK_PHONE'] + df_new['FLAG_CONT_MOBILE'] + 
        df_new['FLAG_PHONE'] + df_new['FLAG_EMAIL']
    )
    
    # External scores
    ext_cols = [col for col in df_new.columns if 'EXT_SOURCE' in col]
    if ext_cols:
        df_new['EXT_SOURCE_MEAN'] = df_new[ext_cols].mean(axis=1)
        df_new['EXT_SOURCE_MAX'] = df_new[ext_cols].max(axis=1)
        df_new['EXT_SOURCE_MIN'] = df_new[ext_cols].min(axis=1)
    
    return df_new

def prepare_data(df, label_encoders=None):
    """Prepare data for modeling"""
    df_prep = df.copy()
    
    # Separate target if exists
    if 'TARGET' in df_prep.columns:
        y = df_prep['TARGET']
        df_prep = df_prep.drop(columns=['TARGET'])
    else:
        y = None
    
    # Drop ID column
    if 'SK_ID_CURR' in df_prep.columns:
        df_prep = df_prep.drop(columns=['SK_ID_CURR'])
    
    # Encode categorical variables
    if label_encoders is None:
        label_encoders = {}
        for col in df_prep.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_prep[col] = le.fit_transform(df_prep[col].astype(str))
            label_encoders[col] = le
    else:
        for col in df_prep.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                df_prep[col] = label_encoders[col].transform(df_prep[col].astype(str))
    
    # Fill missing values
    df_prep = df_prep.fillna(df_prep.median())
    
    return df_prep, y, label_encoders

def train_lgb_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    return model

def main():
    print("=" * 60)
    print("Running IMPROVED Problem 4 - Advanced Feature Engineering")
    print("=" * 60)
    
    # Paths
    data_folder = os.path.join(os.path.dirname(__file__), '../data') 
    readme_path = os.path.join(os.path.dirname(__file__), '../README.md')
    
    # Load base training data
    print("\n[1/11] Loading base training data...")
    df_train = load_and_preprocess_train(data_folder)
    
    # Aggregate auxiliary datasets
    print("\n[2/11] Aggregating bureau data...")
    bureau_agg = aggregate_bureau(data_folder)
    
    print("\n[3/11] Aggregating previous application data...")
    prev_agg = aggregate_previous_application(data_folder)
    
    print("\n[4/11] Aggregating installments data...")
    inst_agg = aggregate_installments(data_folder)
    
    print("\n[5/11] Aggregating credit card data...")
    cc_agg = aggregate_credit_card(data_folder)
    
    print("\n[6/11] Aggregating POS cash data...")
    pos_agg = aggregate_pos_cash(data_folder)
    
    # Merge all data
    print("\n[7/11] Merging all datasets...")
    df_merged = df_train.copy()
    df_merged = df_merged.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df_merged = df_merged.merge(prev_agg, on='SK_ID_CURR', how='left')
    df_merged = df_merged.merge(inst_agg, on='SK_ID_CURR', how='left')
    df_merged = df_merged.merge(cc_agg, on='SK_ID_CURR', how='left')
    df_merged = df_merged.merge(pos_agg, on='SK_ID_CURR', how='left')
    
    print(f"   Merged dataset shape: {df_merged.shape}")
    
    # Create domain features
    print("\n[8/11] Creating domain-specific features...")
    df_full = create_domain_features(df_merged)
    
    print(f"   Final dataset shape: {df_full.shape}")
    
    # Define 10 feature patterns
    print("\n[9/11] Preparing 10 feature engineering patterns...")
    
    patterns = {}
    
    # Pattern 1: Baseline (only application_train features)
    patterns['P1_Baseline'] = df_train.copy()
    
    # Pattern 2: Baseline + Domain features
    patterns['P2_Domain_Features'] = create_domain_features(df_train)
    
    # Pattern 3: + Bureau features
    df_temp = df_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    patterns['P3_Bureau'] = create_domain_features(df_temp)
    
    # Pattern 4: + Previous application features
    df_temp = df_train.merge(prev_agg, on='SK_ID_CURR', how='left')
    patterns['P4_Previous_App'] = create_domain_features(df_temp)
    
    # Pattern 5: + Installments features
    df_temp = df_train.merge(inst_agg, on='SK_ID_CURR', how='left')
    patterns['P5_Installments'] = create_domain_features(df_temp)
    
    # Pattern 6: + Credit card features
    df_temp = df_train.merge(cc_agg, on='SK_ID_CURR', how='left')
    patterns['P6_Credit_Card'] = create_domain_features(df_temp)
    
    # Pattern 7: Bureau + Previous application
    df_temp = df_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df_temp = df_temp.merge(prev_agg, on='SK_ID_CURR', how='left')
    patterns['P7_Bureau_PrevApp'] = create_domain_features(df_temp)
    
    # Pattern 8: Bureau + Installments + Credit card
    df_temp = df_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df_temp = df_temp.merge(inst_agg, on='SK_ID_CURR', how='left')
    df_temp = df_temp.merge(cc_agg, on='SK_ID_CURR', how='left')
    patterns['P8_Bureau_Inst_CC'] = create_domain_features(df_temp)
    
    # Pattern 9: Previous app + Installments + POS
    df_temp = df_train.merge(prev_agg, on='SK_ID_CURR', how='left')
    df_temp = df_temp.merge(inst_agg, on='SK_ID_CURR', how='left')
    df_temp = df_temp.merge(pos_agg, on='SK_ID_CURR', how='left')
    patterns['P9_PrevApp_Inst_POS'] = create_domain_features(df_temp)
    
    # Pattern 10: All features combined
    patterns['P10_All_Features'] = df_full.copy()
    
    # Evaluate each pattern
    print("\n[10/11] Training and evaluating all patterns...")
    print("=" * 60)
    
    results = []
    
    for name, df_pattern in patterns.items():
        print(f"\nEvaluating {name}...")
        
        # Prepare data
        X, y, _ = prepare_data(df_pattern)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train LightGBM model
        model = train_lgb_model(X_train, y_train, X_val, y_val)
        
        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, y_pred)
        
        results.append({
            'Pattern': name,
            'Features': X.shape[1],
            'AUC_Score': round(auc, 4)
        })
        
        print(f"   Features: {X.shape[1]} | AUC: {auc:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AUC_Score', ascending=False)
    
    # Save results
    print("\n[11/11] Saving results...")
    results_path = os.path.join(data_folder, 'problem4_improved_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"   Results saved to {results_path}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
    
    best_pattern = results_df.iloc[0]
    baseline_auc = results_df[results_df['Pattern'] == 'P1_Baseline']['AUC_Score'].values[0]
    improvement = best_pattern['AUC_Score'] - baseline_auc
    
    print(f"\nBest Pattern: {best_pattern['Pattern']}")
    print(f"Best AUC: {best_pattern['AUC_Score']:.4f}")
    print(f"Improvement over baseline: +{improvement:.4f} ({improvement/baseline_auc*100:.2f}%)")
    print("=" * 60)
    
    # Update README.md
    print("\nUpdating README.md...")
    readme_update = f"""
-----------

## Problem 4: Advanced Feature Engineering (IMPROVED)

### Overview
This improved solution incorporates features from multiple auxiliary datasets:
- Bureau credit history
- Previous Home Credit applications
- Installment payment history
- Credit card balance history
- POS and cash loan history

### Results Summary
Trained and evaluated 10 different feature engineering patterns using LightGBM:

{tabulate(results_df, headers='keys', tablefmt='github', showindex=False)}

### Key Findings
- **Best Pattern**: {best_pattern['Pattern']} with **{best_pattern['Features']} features**
- **Best AUC Score**: **{best_pattern['AUC_Score']:.4f}**
- **Improvement over baseline**: **+{improvement:.4f}** ({improvement/baseline_auc*100:.2f}%)

### Feature Engineering Techniques Used
1. Aggregated statistics from auxiliary tables (mean, max, min, sum, count)
2. Domain-specific financial ratios (income/credit, credit utilization)
3. Time-based features (age, employment duration, payment delays)
4. Behavioral features (late payments, document submission counts)
5. External score combinations and interactions

Full results available in [data/problem4_improved_results.csv](data/problem4_improved_results.csv)
"""
    
    # Read existing README
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Remove old Problem 4 section if exists
    if "## Problem 4" in readme_content:
        parts = readme_content.split("## Problem 4")
        readme_content = parts[0]
    
    # Append new Problem 4 section
    with open(readme_path, 'w') as f:
        f.write(readme_content.rstrip() + readme_update)
    
    print("README.md updated successfully!")
    print("\n" + "=" * 60)
    print("IMPROVED FEATURE ENGINEERING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
    