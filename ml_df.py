
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:11:02 2025

@author: alexperes
"""
#%% libaries
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, precision_recall_curve, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import numpy as np
import xgboost as xgb
import shap
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

#%% Loading files
df = pd.read_csv('/Users/alexperes/Desktop/df_ml.csv', sep = ",")
prs_cols = ['FID', 'IID', 'PHENO', 'CNT', 'CNT2', 'SCORE']
train_prs = pd.read_csv('/Users/alexperes/Desktop/train_PRS_0.05.profile', sep="\s+", names=prs_cols, header=0)
test_prs = pd.read_csv('/Users/alexperes/Desktop/test_PRS_0.05.profile', sep="\s+", names=prs_cols, header=0)
train_pheno = pd.read_csv('/Users/alexperes/Desktop/Files_Latest/train_cpsp.phe', sep="\t")
test_pheno = pd.read_csv('/Users/alexperes/Desktop/Files_Latest/test_cpsp.phe', sep="\t")
#%%
# Remove rows where PHENO == -9 (usually indicates missing/invalid phenotype)
train_prs = train_prs[train_prs['PHENO'] != -9]
test_prs = test_prs[test_prs['PHENO'] != -9]


#%% Merging df
# Merge df_ml with train_prs and train_pheno to create train_df
train_df = (
    train_pheno
    .merge(train_prs, on='IID')
    .merge(df, on='IID')
)

# Merge df_ml with test_prs and test_pheno to create test_df
test_df = (
    test_pheno
    .merge(test_prs, on='IID')
    .merge(df, on='IID')
)

#%%
print(train_df['chronic_pain_cc_y'].value_counts())
print(test_df['chronic_pain_cc_y'].value_counts())

#%% Cleaning columns:
    # Columns to drop explicitly
cols_to_drop = [
    'FID_x', 'FID_y', 'PHENO', 'CNT', 'CNT2',
    'operations', 'noncancer_illnesses',  # <- fix here
    'used_in_genetic_PCs', 'diagnoses'
]

existing_cols_to_drop = [col for col in cols_to_drop if col in train_df.columns]


# Drop only existing ones
train_df_clean = train_df.drop(columns=existing_cols_to_drop)

# Drop all *_y columns
cols_to_drop_y = [col for col in train_df_clean.columns if col.endswith('_y')]
train_df_clean = train_df_clean.drop(columns=cols_to_drop_y)

# Rename *_x columns back to original names
train_df_clean = train_df_clean.rename(columns={col: col[:-2] for col in train_df_clean.columns if col.endswith('_x')})

# Drop only existing ones
existing_cols_to_drop = [col for col in cols_to_drop if col in test_df.columns]
test_df_clean = test_df.drop(columns=existing_cols_to_drop)

# Drop all *_y columns
cols_to_drop_y = [col for col in test_df_clean.columns if col.endswith('_y')]
test_df_clean = test_df_clean.drop(columns=cols_to_drop_y)

# Rename *_x columns back to original names
test_df_clean = test_df_clean.rename(columns={col: col[:-2] for col in test_df_clean.columns if col.endswith('_x')})

#%%
def count_list_items(val):
    if pd.isna(val):
        return 0
    try:
        return len(eval(val))
    except:
        return 0
#%%
train_df_clean['bloodtype_haplotype'] = train_df_clean['bloodtype_haplotype'].apply(count_list_items)
test_df_clean['bloodtype_haplotype'] = test_df_clean['bloodtype_haplotype'].apply(count_list_items)

#%%
print("Train shape:", train_df_clean.shape)
print("Test shape:", test_df_clean.shape)
print("Total missing values in train:", train_df_clean.isna().sum().sum())
print("Total missing values in test:", test_df_clean.isna().sum().sum())

#%% Impute NaNs

# Step 1: Identify numeric columns
numeric_cols = train_df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Step 2: Identify columns with all NaNs in training set
all_nan_cols = train_df_clean[numeric_cols].columns[train_df_clean[numeric_cols].isna().all()]

# Step 3: Drop all-NaN columns from train and test
train_df_clean = train_df_clean.drop(columns=all_nan_cols)
test_df_clean = test_df_clean.drop(columns=all_nan_cols, errors='ignore')

# Step 4: Update numeric columns after dropping
numeric_cols = [col for col in numeric_cols if col not in all_nan_cols]

# Step 5: Impute remaining missing values in numeric columns

imputer = SimpleImputer(strategy='mean')
train_df_clean[numeric_cols] = imputer.fit_transform(train_df_clean[numeric_cols])
test_df_clean[numeric_cols] = imputer.transform(test_df_clean[numeric_cols])

#%%

# Prepare training data
X_train = train_df_clean.drop(columns=['IID','year_surgery_CPSP', 'chronic_pain_cc'])
y_train = train_df_clean['chronic_pain_cc']

# Prepare test data
X_test = test_df_clean.drop(columns=['IID', 'year_surgery_CPSP', 'chronic_pain_cc'])
y_test = test_df_clean['chronic_pain_cc']

#%%
from sklearn.preprocessing import LabelEncoder

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        full_col = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(full_col)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

#%%edfine focal loss
def focal_loss(alpha=0.25, gamma=2.0):
    def fl_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        grad = alpha * y_true * (1 - p) ** gamma * (p - 1) + \
               (1 - alpha) * (1 - y_true) * p ** gamma * p
        hess = alpha * y_true * (1 - p) ** (gamma - 1) * (
            gamma * p * (1 - p) + (1 - p)) + \
               (1 - alpha) * (1 - y_true) * p ** (gamma - 1) * (
            gamma * p * (1 - p) + p)
        return grad, hess
    return fl_obj

#%%
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 6,
    'eta': 0.05,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtest, 'eval')],
    obj=focal_loss(alpha=0.25, gamma=2),
    early_stopping_rounds=50
)
#%%Predict and evaluate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred_proba = bst.predict(dtest)
y_pred = (y_pred_proba > 0.35).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
#%%
overlap = set(train_df_clean['IID']) & set(test_df_clean['IID'])
print(f"Overlapping IIDs: {len(overlap)}")

correlations = train_df_clean.corr(numeric_only=True)['chronic_pain_cc'].drop('chronic_pain_cc').sort_values(ascending=False)
print(correlations.head(10))

#%%
overlap = set(train_df_clean['SCORE']).intersection(set(test_df_clean['SCORE']))
print(f"Number of overlapping SCORE values: {len(overlap)}")

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(train_df_clean['SCORE'], label='Train SCORE', fill=True)
sns.kdeplot(test_df_clean['SCORE'], label='Test SCORE', fill=True)
plt.title('Distribution of SCORE in Train vs Test')
plt.xlabel('SCORE')
plt.legend()
plt.show()
#%%
test_corr = test_df_clean[['SCORE', 'chronic_pain_cc']].corr().iloc[0, 1]
print(f"Correlation between SCORE and label in test set: {test_corr:.3f}")
#%%
top_corr_features = (
    test_df_clean
    .corr(numeric_only=True)['chronic_pain_cc']
    .drop('chronic_pain_cc')
    .sort_values(ascending=False)
    .head(10)
    .index
    .tolist()
)

print(top_corr_features)
#%%
# Check correlations in train and test for all features
train_corr = train_df_clean.corr(numeric_only=True)['chronic_pain_cc'].drop('chronic_pain_cc').sort_values(ascending=False)
test_corr = test_df_clean.corr(numeric_only=True)['chronic_pain_cc'].drop('chronic_pain_cc').sort_values(ascending=False)

# Compare top correlated features in train and test
print("Top correlated features in TRAIN:")
print(train_corr.head(10))

print("\nTop correlated features in TEST:")
print(test_corr.head(10))

# Join correlations side-by-side
corr_compare = pd.DataFrame({
    'train_corr': train_corr,
    'test_corr': test_corr
}).sort_values(by='train_corr', ascending=False)

# Display features with large discrepancy
corr_compare['abs_diff'] = (corr_compare['train_corr'] - corr_compare['test_corr']).abs()
print("\nTop features with biggest train/test correlation difference:")
print(corr_compare.sort_values(by='abs_diff', ascending=False).head(10))
#%%
# Prepare results storage
results = {}
#%%

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)[:,1]
results['LogisticRegression'] = roc_auc_score(y_test, y_pred_proba_lr)

print("\nLogistic Regression:")
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:,1]
results['RandomForest'] = roc_auc_score(y_test, y_pred_proba_rf)

print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf))

# CatBoost
cat = CatBoostClassifier(verbose=0, iterations=500, depth=6, learning_rate=0.05, random_state=42)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
y_pred_proba_cat = cat.predict_proba(X_test)[:,1]
results['CatBoost'] = roc_auc_score(y_test, y_pred_proba_cat)

print("\nCatBoost:")
print(classification_report(y_test, y_pred_cat))

# LightGBM
lgbm = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
lgbm.fit(X_train, y_train)
y_pred_lgb = lgbm.predict(X_test)
y_pred_proba_lgb = lgbm.predict_proba(X_test)[:,1]
results['LightGBM'] = roc_auc_score(y_test, y_pred_proba_lgb)

print("\nLightGBM:")
print(classification_report(y_test, y_pred_lgb))

# XGBoost (without focal loss for benchmark)
xgb_model = xgb.XGBClassifier(
    max_depth=6, learning_rate=0.05, n_estimators=500, use_label_encoder=False, eval_metric='auc'
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:,1]
results['XGBoost'] = roc_auc_score(y_test, y_pred_proba_xgb)

print("\nXGBoost:")
print(classification_report(y_test, y_pred_xgb))

# Print ROC AUC summary
print("\nROC AUC summary:")
for model, auc in results.items():
    print(f"{model}: {auc:.3f}")
#%%
# Select top N features based on correlation
top_features = (
    train_df_clean
    .corr(numeric_only=True)['chronic_pain_cc']
    .abs()
    .drop('chronic_pain_cc')
    .sort_values(ascending=False)
    .head(20)  # you can adjust this number
    .index
    .tolist()
)

# Subset the data
X_train_fs = X_train[top_features]
X_test_fs = X_test[top_features]
#%%
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Fit Isolation Forest only on training features (unsupervised)
iso = IsolationForest(contamination=len(y_train[y_train==1]) / len(y_train), random_state=42)
iso.fit(X_train)

# Predict anomalies (-1: anomaly, 1: normal)
y_pred_iso_test = iso.predict(X_test)
y_pred_iso_test = np.where(y_pred_iso_test == -1, 1, 0)  # Map to same class labels

print(confusion_matrix(y_test, y_pred_iso_test))
print(classification_report(y_test, y_pred_iso_test))
#%% Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.ensemble import BalancedRandomForestClassifier

#%% Define your train and test data (already cleaned and imputed)
# Assume X_train, y_train, X_test, y_test are ready

results = {}

#%% 1️⃣ Random Forest (baseline)
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:,1]

results['RandomForest'] = {
    'confusion': confusion_matrix(y_test, y_pred_rf),
    'report': classification_report(y_test, y_pred_rf, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
}

#%% 2️⃣ Random Forest with class_weight balanced
rf_bal = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
rf_bal.fit(X_train, y_train)
y_pred_rf_bal = rf_bal.predict(X_test)
y_pred_proba_rf_bal = rf_bal.predict_proba(X_test)[:,1]

results['RF_balanced'] = {
    'confusion': confusion_matrix(y_test, y_pred_rf_bal),
    'report': classification_report(y_test, y_pred_rf_bal, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf_bal)
}

#%% 3️⃣ SMOTE + Random Forest
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

rf_sm = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_sm.fit(X_train_res, y_train_res)
y_pred_rf_sm = rf_sm.predict(X_test)
y_pred_proba_rf_sm = rf_sm.predict_proba(X_test)[:,1]

results['SMOTE_RF'] = {
    'confusion': confusion_matrix(y_test, y_pred_rf_sm),
    'report': classification_report(y_test, y_pred_rf_sm, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf_sm)
}

#%% 4️⃣ Isolation Forest (unsupervised anomaly detection)
contam = len(y_train[y_train==1]) / len(y_train)  # actual prevalence in train set

iso = IsolationForest(contamination=contam, n_estimators=300, random_state=42)
iso.fit(X_train)
y_pred_iso_test = iso.predict(X_test)
y_pred_iso_test = np.where(y_pred_iso_test == -1, 1, 0)

results['IsolationForest'] = {
    'confusion': confusion_matrix(y_test, y_pred_iso_test),
    'report': classification_report(y_test, y_pred_iso_test, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_iso_test)
}

#%% 5️⃣ One-Class SVM (unsupervised)
ocsvm = OneClassSVM(kernel='rbf', nu=contam, gamma='scale')
ocsvm.fit(X_train)
y_pred_ocsvm_test = ocsvm.predict(X_test)
y_pred_ocsvm_test = np.where(y_pred_ocsvm_test == -1, 1, 0)

results['OneClassSVM'] = {
    'confusion': confusion_matrix(y_test, y_pred_ocsvm_test),
    'report': classification_report(y_test, y_pred_ocsvm_test, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_ocsvm_test)
}

#%% Print all results
for model, res in results.items():
    print(f"\n--- {model} ---")
    print("Confusion matrix:")
    print(res['confusion'])
    print("Classification report:")
    print(res['report'])
    print(f"ROC AUC: {res['roc_auc']:.3f}")
#%%

# Example for Isolation Forest
y_scores_iso = iso.decision_function(X_test) * -1  # flip anomaly score
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_iso)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, marker='.')
plt.title(f'IsolationForest PR Curve (AUC={pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()


