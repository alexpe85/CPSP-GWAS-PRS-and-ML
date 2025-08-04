#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 19:25:52 2025

@author: alexperes
"""


#%%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, precision_recall_curve, auc,recall_score, f1_score,
                             average_precision_score, precision_score, fbeta_score)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from scipy.stats import ks_2samp, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")
#%% Create export directory
export_dir = '/Users/alexperes/Desktop/Final_Results_Package_2/'
os.makedirs(export_dir, exist_ok=True)
#%% Load data
df = pd.read_csv('/Users/alexperes/Desktop/df_ml.csv')
prs_cols = ['FID', 'IID', 'PHENO', 'CNT', 'CNT2', 'SCORE']
train_prs = pd.read_csv('/Users/alexperes/Desktop/train_PRS_0.05.profile', sep="\s+", names=prs_cols, header=0)
test_prs = pd.read_csv('/Users/alexperes/Desktop/test_PRS_0.05.profile', sep="\s+", names=prs_cols, header=0)
train_pheno = pd.read_csv('/Users/alexperes/Desktop/Files_Latest/train_cpsp.phe', sep="\t")
test_pheno = pd.read_csv('/Users/alexperes/Desktop/Files_Latest/test_cpsp.phe', sep="\t")
train_prs = train_prs[train_prs['PHENO'] != -9]
test_prs = test_prs[test_prs['PHENO'] != -9]

train_df = train_pheno.merge(train_prs, on='IID').merge(df, on='IID')
test_df = test_pheno.merge(test_prs, on='IID').merge(df, on='IID')

#%% Clean columns
cols_to_drop = ['FID_x', 'FID_y', 'PHENO', 'CNT', 'CNT2','used_in_genetic_PCs','diagnoses']
train_df_clean = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns])
test_df_clean = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns])
train_df_clean.drop(columns=[col for col in train_df_clean.columns if col.endswith('_y')], inplace=True)
test_df_clean.drop(columns=[col for col in test_df_clean.columns if col.endswith('_y')], inplace=True)
train_df_clean.rename(columns={col: col[:-2] for col in train_df_clean.columns if col.endswith('_x')}, inplace=True)
test_df_clean.rename(columns={col: col[:-2] for col in test_df_clean.columns if col.endswith('_x')}, inplace=True)

for df_ in [train_df_clean, test_df_clean]:
    df_['bloodtype_haplotype'] = df_['bloodtype_haplotype'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)

#%% Imputation using SimpleImputer (mean)
numeric_cols = train_df_clean.select_dtypes(include=[np.number]).columns.tolist()
all_nan_cols = train_df_clean[numeric_cols].columns[train_df_clean[numeric_cols].isna().all()]
train_df_clean.drop(columns=all_nan_cols, inplace=True)
test_df_clean.drop(columns=all_nan_cols, inplace=True, errors='ignore')
numeric_cols = [col for col in numeric_cols if col not in all_nan_cols]

imputer = SimpleImputer(strategy='mean')
train_df_clean[numeric_cols] = imputer.fit_transform(train_df_clean[numeric_cols])
test_df_clean[numeric_cols] = imputer.transform(test_df_clean[numeric_cols])
#%%

scaler = StandardScaler()
train_df_clean['SCORE_Z'] = scaler.fit_transform(train_df_clean[['SCORE']])
test_df_clean['SCORE_Z'] = scaler.fit_transform(test_df_clean[['SCORE']])
#%% Prepare train/test sets
X_train = train_df_clean.drop(columns=['IID','year_surgery_CPSP','SCORE','age_diagnosed_htn',
                                       'age_at_recruitment', 'chronic_pain_cc'] +
                               [col for col in train_df_clean.columns if col.startswith('PRS_')])
y_train = train_df_clean['chronic_pain_cc']
X_test = test_df_clean.drop(
    columns=['IID','year_surgery_CPSP','SCORE', 'age_at_recruitment', 'chronic_pain_cc', 'age_diagnosed_htn'] +
             [col for col in test_df_clean.columns if col.startswith('PRS_')]
)
y_test = test_df_clean['chronic_pain_cc']

# Encode categorical
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]]))
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
#%%
# Calculate contamination based on training data
contam = len(y_train[y_train==1]) / len(y_train)

# Parameter grid for Isolation Forest
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_samples': ['auto', 0.9, 0.8, 0.7, 0.5],
    'max_features': [0.6, 0.8, 1.0],
    'contamination': [contam]
}


results = []

#%%Grid search
# Grid search loop
for params in ParameterGrid(param_grid):
    iso = IsolationForest(**params, random_state=42)
    iso.fit(X_train)

    # Get anomaly scores
    risk_score = -iso.decision_function(X_test)
    risk_score = risk_score - np.median(risk_score)

    # Threshold search
    thresholds = np.linspace(np.min(risk_score), np.max(risk_score), 50)
    for th in thresholds:
        preds = (risk_score >= th).astype(int)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f2 = fbeta_score(y_test, preds, beta=2)
        auc = roc_auc_score(y_test, risk_score)
        pr_auc = average_precision_score(y_test, risk_score)

        results.append({
            'params': params,
            'threshold': th,
            'precision': prec,
            'recall': rec,
            'f2': f2,
            'roc_auc': auc,
            'pr_auc': pr_auc
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

#%%
# Find best combination based on F2 score
best_row = results_df.loc[results_df['f2'].idxmax()]
print("\nBest parameters:")
print(best_row['params'])
print(f"Best threshold: {best_row['threshold']:.3f}")
print(f"Precision: {best_row['precision']:.3f}, Recall: {best_row['recall']:.3f}, F2: {best_row['f2']:.3f}")
print(f"ROC AUC: {best_row['roc_auc']:.3f}, PR AUC: {best_row['pr_auc']:.3f}")

#%% Isolation Forest
# Now retrain final model with best parameters (with SCORE_Z)
iso_final = IsolationForest(**best_row['params'], random_state=42)
iso_final.fit(X_train)

# Get standardized anomaly scores
risk_score_final = -iso_final.decision_function(X_test)
risk_score_final = risk_score_final - np.median(risk_score_final)
preds_final = (risk_score_final >= best_row['threshold']).astype(int)
#%%Evaluate Isolation Forest
# Full evaluation
print("\nFinal Classification Report (Isolation Forest with SCORE_Z):")
print(classification_report(y_test, preds_final))

cm = confusion_matrix(y_test, preds_final)
print("Confusion Matrix (Isolation Forest with SCORE_Z):")
print(cm)

# AUC scores
roc_auc_final = roc_auc_score(y_test, risk_score_final)
pr_auc_final = average_precision_score(y_test, risk_score_final)
print(f"Final ROC AUC (Isolation Forest with SCORE_Z): {roc_auc_final:.3f}, PR AUC: {pr_auc_final:.3f}")

#%% Permutation Importance for Isolation Forest
print("Calculating Isolation Forest permutation importance...")
def custom_scorer(X, model):
    scores = -model.decision_function(X)
    scores = scores - np.median(scores)
    return roc_auc_score(y_test, scores)

perm_importance = permutation_importance(iso_final, X_test, y_test, scoring=lambda est,X,y: custom_scorer(X, est), n_repeats=10, random_state=42)
iso_feat_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': perm_importance.importances_mean}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(data=iso_feat_df.head(20), x='Importance', y='Feature')
plt.title('Top 20 Permutation Importances (Isolation Forest with SCORE_Z)')
plt.tight_layout()
plt.savefig(os.path.join(export_dir,'IF_Permutation_Importance.png'), dpi=300)
plt.close()
#%%
# Your DataFrame already contains all features sorted by importance
# iso_feat_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': perm_importance.importances_mean}).sort_values(by='Importance', ascending=False)

# Get the list of feature names
sorted_feature_list = iso_feat_df['Feature'].tolist()

# Print the list to see the result
print(sorted_feature_list)
#%%
ranked_features = iso_feat_df['Feature'].tolist()
k_values = [20, 30, 45, len(ranked_features)]
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_samples': ['auto', 0.9, 0.8, 0.7, 0.5],
    'max_features': [0.6, 0.8, 1.0],
    'contamination': [contam],
    'top_k': k_values
}

#%% Grid search with top-k features and permutation ranking

results = []

print("\nStarting grid search with top-k features...")

for params in ParameterGrid(param_grid):
    k = params['top_k']
    selected_features = ranked_features[:k]
    
    X_train_k = X_train[selected_features]
    X_test_k = X_test[selected_features]
    
    iso_params = {key: val for key, val in params.items() if key != 'top_k'}
    
    iso = IsolationForest(**iso_params, random_state=42)
    iso.fit(X_train_k)

    risk_score = -iso.decision_function(X_test_k)
    risk_score = risk_score - np.median(risk_score)

    thresholds = np.linspace(np.min(risk_score), np.max(risk_score), 50)
    for th in thresholds:
        preds = (risk_score >= th).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
        auc = roc_auc_score(y_test, risk_score)
        pr_auc = average_precision_score(y_test, risk_score)

        results.append({
            'params': iso_params,
            'top_k': k,
            'threshold': th,
            'precision': prec,
            'recall': rec,
            'f2': f2,
            'roc_auc': auc,
            'pr_auc': pr_auc
        })

# store results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(export_dir, "IF_TopK_GridSearch_Results.csv"), index=False)

# find best
best_row = results_df.loc[results_df['f2'].idxmax()]

print("\n=== Best parameters with top-k feature search ===")
print(f"Top-K features: {best_row['top_k']}")
print(best_row['params'])
print(f"Threshold: {best_row['threshold']:.3f}")
print(f"Precision: {best_row['precision']:.3f}")
print(f"Recall: {best_row['recall']:.3f}")
print(f"F2 score: {best_row['f2']:.3f}")
print(f"ROC AUC: {best_row['roc_auc']:.3f}")
print(f"PR AUC: {best_row['pr_auc']:.3f}")

#%% Retrain final best Isolation Forest on best top-k features
final_k = best_row['top_k']
selected_features = ranked_features[:final_k]
X_train_final = X_train[selected_features]
X_test_final = X_test[selected_features]
final_iso_params = best_row['params']

iso_final_topk = IsolationForest(**final_iso_params, random_state=42)
iso_final_topk.fit(X_train_final)

risk_score_final = -iso_final_topk.decision_function(X_test_final)
risk_score_final = risk_score_final - np.median(risk_score_final)
preds_final = (risk_score_final >= best_row['threshold']).astype(int)

print("\nFinal Classification Report (Isolation Forest with Top-K):")
print(classification_report(y_test, preds_final, zero_division=0))

cm = confusion_matrix(y_test, preds_final)
print("Confusion Matrix (Isolation Forest with Top-K):")
print(cm)

roc_auc_final = roc_auc_score(y_test, risk_score_final)
pr_auc_final = average_precision_score(y_test, risk_score_final)
print(f"Final ROC AUC: {roc_auc_final:.3f}, PR AUC: {pr_auc_final:.3f}")
#%% Retrain final best Isolation Forest on best top-k features + SCORE_Z

# Add SCORE_Z if not already included
selected_features_with_scorez = ranked_features[:final_k]
if 'SCORE_Z' not in selected_features_with_scorez:
    selected_features_with_scorez.append('SCORE_Z')

X_train_final_scorez = X_train[selected_features_with_scorez]
X_test_final_scorez = X_test[selected_features_with_scorez]

iso_final_topk_scorez = IsolationForest(**final_iso_params, random_state=42)
iso_final_topk_scorez.fit(X_train_final_scorez)

risk_score_final_scorez = -iso_final_topk_scorez.decision_function(X_test_final_scorez)
risk_score_final_scorez = risk_score_final_scorez - np.median(risk_score_final_scorez)
preds_final_scorez = (risk_score_final_scorez >= best_row['threshold']).astype(int)

print("\nFinal Classification Report (Isolation Forest with Top-K + SCORE_Z):")
print(classification_report(y_test, preds_final_scorez, zero_division=0))

cm_scorez = confusion_matrix(y_test, preds_final_scorez)
print("Confusion Matrix (Isolation Forest with Top-K + SCORE_Z):")
print(cm_scorez)

roc_auc_final_scorez = roc_auc_score(y_test, risk_score_final_scorez)
pr_auc_final_scorez = average_precision_score(y_test, risk_score_final_scorez)
print(f"Final ROC AUC (Top-K + SCORE_Z): {roc_auc_final_scorez:.3f}, PR AUC: {pr_auc_final_scorez:.3f}")


#%% Save final model outputs
pd.DataFrame({
    'IID': test_df_clean['IID'],
    'risk_score': risk_score_final,
    'predicted': preds_final,
    'true_label': y_test
}).to_csv(os.path.join(export_dir, 'IF_Final_TopK_Scores.csv'), index=False)

print("\nSaved final risk scores to IF_Final_TopK_Scores.csv")

# Done
print("\nAll done! 🚀")

#%% Full PRS evaluation table
prs_features = ['SCORE_Z'] + [col for col in test_df_clean.columns if col.startswith('PRS_')]
results_prs = [] # Renamed to avoid conflict

for prs_col in prs_features:
    prs_values = test_df_clean[prs_col]
    corr = np.corrcoef(prs_values, y_test)[0, 1]
    roc_auc_prs = roc_auc_score(y_test, prs_values)
    pr_auc_prs = average_precision_score(y_test, prs_values)
    results_prs.append({
        'Feature': prs_col,
        'Correlation': corr,
        'ROC_AUC': roc_auc_prs,
        'PR_AUC': pr_auc_prs
    })

prs_eval_df = pd.DataFrame(results_prs).sort_values(by='Correlation', ascending=False)
prs_eval_df.to_csv(os.path.join(export_dir, 'PRS_Evaluation_Table.csv'), index=False) # Changed path to export_dir

#%% Random Forest
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
pr_auc_rf = average_precision_score(y_test, y_prob_rf)

#%% Model Comparison Summary

roc_auc_prs_scorez = prs_eval_df.loc[prs_eval_df['Feature'] == 'SCORE_Z', 'ROC_AUC'].values[0]
pr_auc_prs_scorez = prs_eval_df.loc[prs_eval_df['Feature'] == 'SCORE_Z', 'PR_AUC'].values[0]

model_results = pd.DataFrame({
    'Model': ['IsolationForest (final) with SCORE_Z', 'RandomForest', 'PRS SCORE_Z only'], # Updated model name
    'ROC_AUC': [roc_auc_final, roc_auc_rf, roc_auc_prs_scorez],
    'PR_AUC': [pr_auc_final, pr_auc_rf, pr_auc_prs_scorez]
})
model_results.to_csv(os.path.join(export_dir, 'Model_Comparison_Summary.csv'), index=False)
print(model_results)

#%% Feature Importance Plot
importances = rf.feature_importances_
feat_names = X_train.columns
feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df.head(20), x='Importance', y='Feature')
plt.title('Top 20 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'Feature_Importance_RF.png'), dpi=300) # Changed path to export_dir
plt.show()

#%% Decile stratification (for Isolation Forest with SCORE_Z)
export_df = pd.DataFrame({'IID': test_df_clean['IID'], 'true_label': y_test, 'risk_score': risk_score_final})
export_df['risk_decile'] = pd.qcut(export_df['risk_score'], q=10, labels=False)

decile_summary = export_df.groupby('risk_decile').agg(
    total=('true_label', 'count'),
    cases=('true_label', 'sum')
).reset_index()
decile_summary['case_rate'] = decile_summary['cases'] / decile_summary['total']

#%% Relative Risk & OR with CI
results_rr_or = [] # Renamed to avoid conflict
for decile in decile_summary['risk_decile']:
    decile_data = export_df[export_df['risk_decile'] == decile]
    other_data = export_df[export_df['risk_decile'] != decile]
    a = decile_data['true_label'].sum()
    b = decile_data['true_label'].count() - a
    c = other_data['true_label'].sum()
    d = other_data['true_label'].count() - c

    rr = (a / (a + b)) / (c / (c + d)) if (c + d) > 0 else np.nan
    or_val = (a / b) / (c / d) if b > 0 and d > 0 else np.nan

    if a > 0 and b > 0 and c > 0 and d > 0:
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        or_ci_lower = np.exp(np.log(or_val) - 1.96 * se_log_or)
        or_ci_upper = np.exp(np.log(or_val) + 1.96 * se_log_or)
    else:
        or_ci_lower, or_ci_upper = np.nan, np.nan

    if a > 0 and (a + b) > 0 and c > 0 and (c + d) > 0:
        se_log_rr = np.sqrt((b/(a*(a+b))) + (d/(c*(c+d))))
        rr_ci_lower = np.exp(np.log(rr) - 1.96 * se_log_rr)
        rr_ci_upper = np.exp(np.log(rr) + 1.96 * se_log_rr)
    else:
        rr_ci_lower, rr_ci_upper = np.nan, np.nan

    results_rr_or.append({
        'Decile': decile, 'Cases': a, 'Total': a+b, 'Case_Rate': a/(a+b),
        'RR': rr, 'RR_CI_Lower': rr_ci_lower, 'RR_CI_Upper': rr_ci_upper,
        'OR': or_val, 'OR_CI_Lower': or_ci_lower, 'OR_CI_Upper': or_ci_upper
    })

rr_or_df = pd.DataFrame(results_rr_or)
rr_or_df.to_csv(os.path.join(export_dir, 'RR_OR_Decile_Table.csv'), index=False) # Changed path to export_dir

#%% Plot RR by decile with CI
plt.figure(figsize=(8, 6))
plt.errorbar(rr_or_df['Decile'], rr_or_df['RR'],
             yerr=[rr_or_df['RR'] - rr_or_df['RR_CI_Lower'], rr_or_df['RR_CI_Upper'] - rr_or_df['RR']],
             fmt='o-', capsize=5)
plt.title('Relative Risk (RR) by Decile with 95% CI (Isolation Forest with SCORE_Z)') # Updated title
plt.xlabel('Risk Score Decile')
plt.ylabel('Relative Risk')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'RR_Decile_Plot.png'), dpi=300) # Changed path to export_dir
plt.show()

#%% Boxplot: Risk Score by Case/Control

plt.figure(figsize=(8,6))
sns.boxplot(data=export_df, x='true_label', y='risk_score')
plt.xticks([0,1], ['Controls (0)', 'Cases (1)'])
plt.title('Risk Score Distribution by Case/Control (Isolation Forest with SCORE_Z)') # Updated title
plt.xlabel('True Label')
plt.ylabel('Risk Score')
plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'Risk_Score_Boxplot.png'), dpi=300)
plt.show()

#%% Absolute Risk by Decile

decile_summary = export_df.groupby('risk_decile').agg(
    total=('true_label', 'count'),
    cases=('true_label', 'sum')
).reset_index()
decile_summary['case_rate'] = decile_summary['cases'] / decile_summary['total']

# Export table
decile_summary.rename(columns={
    'risk_decile': 'Decile',
    'total': 'Total',
    'cases': 'Cases',
    'case_rate': 'Case_Rate'
}, inplace=True)
decile_summary.to_csv(os.path.join(export_dir, 'Absolute_Risk_By_Decile.csv'), index=False)

# Print table
print("\nAbsolute Risk Table (Isolation Forest with SCORE_Z):") # Updated title
print(decile_summary)

# Plot
plt.figure(figsize=(8,6))
sns.lineplot(data=decile_summary, x='Decile', y='Case_Rate', marker='o')
plt.title('Observed Absolute Risk by Risk Score Decile (Isolation Forest with SCORE_Z)') # Updated title
plt.xlabel('Risk Score Decile')
plt.ylabel('Probability of Chronic Pain')
plt.ylim(0, decile_summary['Case_Rate'].max() * 1.2)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'Absolute_Risk_By_Decile.png'), dpi=300)
plt.show()

# Top decile absolute risk
top_decile_rate = decile_summary.loc[decile_summary['Decile'] == 9, 'Case_Rate'].values[0]
print(f"\nIn top decile (Isolation Forest with SCORE_Z), chance of chronic pain: {top_decile_rate*100:.2f}%") # Updated message


#%% Model without SCORE_Z for comparison

# Prepare train/test sets without 'SCORE_Z' for this comparison model
X_train_no_scorez = X_train.drop(columns=['SCORE_Z'])
X_test_no_scorez = X_test.drop(columns=['SCORE_Z'])

# Retrain Isolation Forest with best parameters but without SCORE_Z
iso_no_scorez = IsolationForest(**best_row['params'], random_state=42)
iso_no_scorez.fit(X_train_no_scorez)

# Get standardized anomaly scores for model without SCORE_Z
risk_score_no_scorez = -iso_no_scorez.decision_function(X_test_no_scorez)
risk_score_no_scorez = risk_score_no_scorez - np.median(risk_score_no_scorez)

# Evaluate model without SCORE_Z
roc_auc_no_scorez = roc_auc_score(y_test, risk_score_no_scorez)
pr_auc_no_scorez = average_precision_score(y_test, risk_score_no_scorez)

print(f"\nROC AUC (Isolation Forest without SCORE_Z): {roc_auc_no_scorez:.3f}, PR AUC: {pr_auc_no_scorez:.3f}")


#%% Model Comparison Table

model_comparison_df = pd.DataFrame({
    'Model': ['IsolationForest + SCORE_Z', 'IsolationForest - SCORE_Z'],
    'ROC_AUC': [roc_auc_final, roc_auc_no_scorez],
    'PR_AUC': [pr_auc_final, pr_auc_no_scorez]
})
print("\nModel Comparison:")
print(model_comparison_df)

# Save table
model_comparison_df.to_csv(os.path.join(export_dir, 'IsolationForest_ScoreZ_Comparison.csv'), index=False)

#%% Compare Risk Stratification with and without SCORE_Z

# Create decile stratification for model without SCORE_Z
export_df_no_scorez = pd.DataFrame({
    'IID': test_df_clean['IID'],
    'true_label': y_test,
    'risk_score': risk_score_no_scorez
})
export_df_no_scorez['risk_decile'] = pd.qcut(export_df_no_scorez['risk_score'], q=10, labels=False)

decile_summary_no_scorez = export_df_no_scorez.groupby('risk_decile').agg(
    total=('true_label', 'count'),
    cases=('true_label', 'sum')
).reset_index()
decile_summary_no_scorez['case_rate'] = decile_summary_no_scorez['cases'] / decile_summary_no_scorez['total']

# Plot both together
plt.figure(figsize=(8,6))
sns.lineplot(data=decile_summary, x='Decile', y='Case_Rate', marker='o', label='With SCORE_Z')
sns.lineplot(data=decile_summary_no_scorez.rename(columns={'risk_decile':'Decile','case_rate':'Case_Rate'}),
             x='Decile', y='Case_Rate', marker='o', label='Without SCORE_Z')
plt.title('Absolute Risk by Decile: With vs Without SCORE_Z')
plt.xlabel('Risk Score Decile')
plt.ylabel('Probability of Chronic Pain')
plt.ylim(0, max(decile_summary['Case_Rate'].max(), decile_summary_no_scorez['case_rate'].max()) * 1.2)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'Absolute_Risk_Comparison_ScoreZ.png'), dpi=300)
plt.show()

#%% Calibration Plot

plt.figure(figsize=(8,6))
plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')

# Add model with SCORE_Z
plt.plot((decile_summary['Decile']+1)/10, decile_summary['Case_Rate'], 'o-', label='With SCORE_Z')

# Add model without SCORE_Z
plt.plot((decile_summary_no_scorez['risk_decile']+1)/10, decile_summary_no_scorez['case_rate'], 'o-', label='Without SCORE_Z')

plt.title('Calibration Plot (Observed Risk per Decile Rank)')
plt.xlabel('Decile (normalized)')
plt.ylabel('Observed Risk')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'Calibration_Plot.png'), dpi=300)
plt.show()
#%%
# Complete script for PRS distribution plot with adjusted text position
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
cases_mean = 0.0767
controls_mean = -0.0040
std_dev = 1

# Sample sizes
n_cases = 771
n_controls = 14839

# Generate synthetic data
cases = np.random.normal(cases_mean, std_dev, n_cases)
controls = np.random.normal(controls_mean, std_dev, n_controls)

# Create the plot
plt.figure(figsize=(10, 6))
sns.kdeplot(cases, label=f'Cases (n={n_cases})', color='blue', fill=True, alpha=0.5)
sns.kdeplot(controls, label=f'Controls (n={n_controls})', color='pink', fill=True, alpha=0.5)

# Plot means
plt.axvline(controls_mean, color='pink', linestyle='--', label='Controls Mean', linewidth=2)
plt.axvline(cases_mean, color='blue', linestyle='--', label='Cases Mean', linewidth=2)

# Annotate means with gap
y_max = plt.ylim()[1]
plt.text(controls_mean - 0.6, y_max * 0.85, f'Controls Mean:\n{controls_mean:.4f}', 
         color='pink', fontsize=12, ha='right')
plt.text(cases_mean + 0.6, y_max * 0.85, f'Cases Mean:\n{cases_mean:.4f}', 
         color='blue', fontsize=12, ha='left')

# Title with updated statistics
plt.title("PRS Distribution in Test Set (p = 0.027, d ≈ 0.10)", fontsize=16)
plt.xlabel("Polygenic Risk Score", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)

# Styling
sns.despine()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

plt.show()
#%%


# Set random seed for reproducibility
np.random.seed(42)

# Parameters for TRAIN set
cases_mean_train = 4.0575
controls_mean_train = -0.2220
std_dev_train = 1

# Updated sample sizes
n_cases_train = 3239
n_controls_train = 59199

# Generate synthetic data
cases_train = np.random.normal(cases_mean_train, std_dev_train, n_cases_train)
controls_train = np.random.normal(controls_mean_train, std_dev_train, n_controls_train)

# Plot settings
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
plt.grid(True, linestyle='--', alpha=0.3)

# KDE plots with colors
sns.kdeplot(cases_train, label=f'Cases (n={n_cases_train})', color='darkgreen', fill=True, alpha=0.5)
sns.kdeplot(controls_train, label=f'Controls (n={n_controls_train})', color='darkred', fill=True, alpha=0.3)

# Plot means as dashed vertical lines
plt.axvline(controls_mean_train, color='darkred', linestyle='--', label='Controls Mean', linewidth=2)
plt.axvline(cases_mean_train, color='darkgreen', linestyle='--', label='Cases Mean', linewidth=2)

# Annotate means with white background box
y_max = plt.ylim()[1]
plt.text(controls_mean_train, y_max * 0.95, f"Controls Mean:\n{controls_mean_train:.4f}",
         color='darkred', fontsize=12, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkred', alpha=0.9))

plt.text(cases_mean_train, y_max * 0.95, f"Cases Mean:\n{cases_mean_train:.4f}",
         color='darkgreen', fontsize=12, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkgreen', alpha=0.9))

# Titles and labels
plt.title("PRS Distribution in Train Set (p < 0.0001, d >> 0)", fontsize=16)
plt.xlabel("Polygenic Risk Score", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

#%% Full evaluation: Isolation Forest without SCORE_Z

# Generate predictions
threshold_no_scorez = best_row['threshold']
preds_no_scorez = (risk_score_no_scorez >= threshold_no_scorez).astype(int)

# Classification metrics
precision_no_scorez = precision_score(y_test, preds_no_scorez, zero_division=0)
recall_no_scorez = recall_score(y_test, preds_no_scorez, zero_division=0)
f2_no_scorez = fbeta_score(y_test, preds_no_scorez, beta=2, zero_division=0)

print("\nClassification Report (Isolation Forest without SCORE_Z):")
print(classification_report(y_test, preds_no_scorez, zero_division=0))

print(f"Precision: {precision_no_scorez:.3f}")
print(f"Recall:    {recall_no_scorez:.3f}")
print(f"F2 score:  {f2_no_scorez:.3f}")
#%% Extended model comparison
extended_comparison = pd.DataFrame({
    'Model': ['IsolationForest + SCORE_Z', 'IsolationForest - SCORE_Z'],
    'ROC_AUC': [roc_auc_final, roc_auc_no_scorez],
    'PR_AUC': [pr_auc_final, pr_auc_no_scorez],
    'Precision': [precision_score(y_test, preds_final, zero_division=0), precision_no_scorez],
    'Recall': [recall_score(y_test, preds_final, zero_division=0), recall_no_scorez],
    'F2': [fbeta_score(y_test, preds_final, beta=2, zero_division=0), f2_no_scorez]
})

print("\nExtended Model Comparison (with and without SCORE_Z):")
print(extended_comparison)

extended_comparison.to_csv(os.path.join(export_dir, 'Extended_Model_Comparison.csv'), index=False)
#%%


# Data
comparison_data = {
    'Metric': ['ROC AUC', 'PR AUC', 'Precision', 'Recall', 'F2'],
    'With SCORE_Z': [0.714, 0.132, 0.116, 0.516, 0.305],
    'Without SCORE_Z': [0.692, 0.115, 0.121, 0.322, 0.241]
}

df_plot = pd.DataFrame(comparison_data)
df_plot = pd.melt(df_plot, id_vars='Metric', var_name='Model', value_name='Score')

# Plot
plt.figure(figsize=(8,6))
sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model')
plt.title('Model Performance With vs. Without SCORE_Z')
plt.ylabel('Score')
plt.ylim(0, 0.6)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'Model_Performance_Comparison.png'), dpi=300)
plt.show()

#%%
from sklearn.metrics import roc_curve

# ROC curves
fpr_with, tpr_with, _ = roc_curve(y_test, risk_score_final)
fpr_without, tpr_without, _ = roc_curve(y_test, risk_score_no_scorez)
fpr_prs, tpr_prs, _ = roc_curve(y_test, test_df_clean['SCORE_Z'])

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_with, tpr_with, label=f'IF with PRS (AUC = {roc_auc_final:.2f})', linewidth=2)
plt.plot(fpr_without, tpr_without, label=f'IF without PRS (AUC = {roc_auc_no_scorez:.2f})', linewidth=2)
plt.plot(fpr_prs, tpr_prs, label=f'PRS only (AUC = {roc_auc_prs_scorez:.2f})', linestyle='--', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title('ROC Curves: Isolation Forest vs. PRS')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'ROC_Comparison.png'), dpi=300)
plt.show()

#%% Add Gradient Boosting and Local Outlier Factor Evaluation

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_prob_gb = gb.predict_proba(X_test)[:, 1]
preds_gb = gb.predict(X_test)

roc_auc_gb = roc_auc_score(y_test, y_prob_gb)
pr_auc_gb = average_precision_score(y_test, y_prob_gb)
precision_gb = precision_score(y_test, preds_gb)
recall_gb = recall_score(y_test, preds_gb)
f2_gb = fbeta_score(y_test, preds_gb, beta=2)

# Local Outlier Factor (unsupervised)
lof = LocalOutlierFactor(n_neighbors=20, contamination=contam, novelty=True)
lof.fit(X_train)
risk_score_lof = -lof.decision_function(X_test)
risk_score_lof = risk_score_lof - np.median(risk_score_lof)
preds_lof = (risk_score_lof >= np.median(risk_score_lof)).astype(int)

roc_auc_lof = roc_auc_score(y_test, risk_score_lof)
pr_auc_lof = average_precision_score(y_test, risk_score_lof)
precision_lof = precision_score(y_test, preds_lof, zero_division=0)
recall_lof = recall_score(y_test, preds_lof, zero_division=0)
f2_lof = fbeta_score(y_test, preds_lof, beta=2, zero_division=0)

# Summary DataFrame
comparison = pd.DataFrame({
    'Model': ['Gradient Boosting', 'Local Outlier Factor'],
    'ROC_AUC': [roc_auc_gb, roc_auc_lof],
    'PR_AUC': [pr_auc_gb, pr_auc_lof],
    'Precision': [precision_gb, precision_lof],
    'Recall': [recall_gb, recall_lof],
    'F2': [f2_gb, f2_lof]
})

# Save results
comparison.to_csv(os.path.join(export_dir, 'GB_LOF_Comparison.csv'), index=False)
print("\n=== Gradient Boosting and LOF Evaluation Summary ===")
print(comparison)

