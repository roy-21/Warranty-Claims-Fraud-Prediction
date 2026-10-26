import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc, f1_score, recall_score, precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap

# 1. Load Data
df = pd.read_csv('df_Clean.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("Data Loaded successfully. Total records:", len(df))

# 2. Advanced Feature Engineering
def feature_engineering(data):
    df_feat = data.copy()
    
    # Issues Aggregation
    df_feat['Total_AC_Issues'] = df_feat['AC_1001_Issue'] + df_feat['AC_1002_Issue'] + df_feat['AC_1003_Issue']
    df_feat['Total_TV_Issues'] = df_feat['TV_2001_Issue'] + df_feat['TV_2002_Issue'] + df_feat['TV_2003_Issue']
    df_feat['Total_Issues'] = df_feat['Total_AC_Issues'] + df_feat['Total_TV_Issues']
    df_feat['Has_No_Issue'] = (df_feat['Total_Issues'] == 0).astype(int)
    
    # Risk Indicators
    df_feat['Claim_Per_Product_Age'] = df_feat['Claim_Value'] / (df_feat['Product_Age'] + 1.0)
    df_feat['Claim_Per_Call_Duration'] = df_feat['Claim_Value'] / (df_feat['Call_details'] + 0.1)
    df_feat['Is_Early_Claim'] = (df_feat['Product_Age'] <= 45).astype(int)
    df_feat['Is_Short_Call'] = (df_feat['Call_details'] <= 2.0).astype(int)
    
    q75 = df_feat['Claim_Value'].quantile(0.75)
    df_feat['Is_High_Claim'] = (df_feat['Claim_Value'] > q75).astype(int)
    
    return df_feat

df_engineered = feature_engineering(df)

X = df_engineered.drop('Fraud', axis=1)
y = df_engineered['Fraud']

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols.append('Service_Centre')
X['Service_Centre'] = X['Service_Centre'].astype(str)

num_cols = [c for c in X.columns if c not in cat_cols]

# Fixed Preprocessor with known categories across entire dataset to avoid dimension mismatch
categories = [X[col].unique().tolist() for col in cat_cols]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),
        ('cat', OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# Fit preprocessor on full feature definitions
preprocessor.fit(X)

encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(encoded_cat_names)
print(f"Total features after One-Hot Encoding: {len(all_feature_names)}")

# Stratified Train-Test Split (25% test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

print(f"Training distribution before SMOTE: {np.bincount(y_train)} -> after SMOTE: {np.bincount(y_train_res)}")
print(f"Testing distribution: {np.bincount(y_test)} ({y_test.sum()} frauds)")

# Define Base Models
base_models = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42),
    "Random Forest (Balanced)": RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=42
    ),
    "XGBoost (SMOTE)": xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    ),
    "LightGBM (Balanced)": lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1
    )
}

results = []
trained_models = {}
probas = {}

for name, model in base_models.items():
    m = clone(model)
    if "SMOTE" in name:
        m.fit(X_train_res, y_train_res)
    else:
        m.fit(X_train_proc, y_train)
        
    trained_models[name] = m
    y_pred = m.predict(X_test_proc)
    y_proba = m.predict_proba(X_test_proc)[:, 1] if hasattr(m, "predict_proba") else y_pred
    probas[name] = y_proba
    
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_pts, precision_pts)
    
    results.append({
        "Model": name,
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4)
    })

res_df = pd.DataFrame(results)
print("\n" + "="*75)
print("[+] MODEL BENCHMARK COMPARISON (TEST SET - DEFAULT 0.5 THRESHOLD)")
print("="*75)
print(res_df.to_string(index=False))

# Optimal Probability Threshold Tuning for Fraud Detection
# In fraud detection, missed fraud is costlier than false alarm.
print("\n" + "="*75)
print("[+] THRESHOLD TUNING (FINDING OPTIMAL FRAUD RECALL)")
print("="*75)

xgb_test_proba = probas["XGBoost (SMOTE)"]
thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
thresh_results = []

for t in thresholds:
    t_pred = (xgb_test_proba >= t).astype(int)
    t_rec = recall_score(y_test, t_pred, zero_division=0)
    t_prec = precision_score(y_test, t_pred, zero_division=0)
    t_f1 = f1_score(y_test, t_pred, zero_division=0)
    cm = confusion_matrix(y_test, t_pred)
    tn, fp, fn, tp = cm.ravel()
    thresh_results.append({
        "Threshold": t,
        "Caught Frauds (TP)": f"{tp}/{tp+fn}",
        "False Alarms (FP)": fp,
        "Recall": f"{t_rec:.1%}",
        "Precision": f"{t_prec:.1%}",
        "F1-Score": round(t_f1, 3)
    })

print(pd.DataFrame(thresh_results).to_string(index=False))

# 5-Fold Stratified Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n" + "="*75)
print("[+] 5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY")
print("="*75)

for name in ["Random Forest (Balanced)", "XGBoost (SMOTE)", "LightGBM (Balanced)"]:
    fold_recalls = []
    fold_aucs = []
    fold_pr_aucs = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        X_tr_p = preprocessor.transform(X_tr)
        X_va_p = preprocessor.transform(X_va)
        
        m = clone(base_models[name])
        if "SMOTE" in name:
            sm = SMOTE(random_state=42)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_p, y_tr)
            m.fit(X_tr_res, y_tr_res)
        else:
            m.fit(X_tr_p, y_tr)
            
        y_val_pred = m.predict(X_va_p)
        y_val_proba = m.predict_proba(X_va_p)[:, 1]
        
        fold_recalls.append(recall_score(y_va, y_val_pred, zero_division=0))
        fold_aucs.append(roc_auc_score(y_va, y_val_proba))
        p_pts, r_pts, _ = precision_recall_curve(y_va, y_val_proba)
        fold_pr_aucs.append(auc(r_pts, p_pts))
        
    print(f"{name:26} | Mean Recall: {np.mean(fold_recalls):.3f} | Mean ROC-AUC: {np.mean(fold_aucs):.3f} | Mean PR-AUC: {np.mean(fold_pr_aucs):.3f}")

# SHAP Explainability
print("\n" + "="*75)
print("[+] COMPUTING SHAP EXPLAINABILITY ON XGBOOST...")
print("="*75)
xgb_final = trained_models["XGBoost (SMOTE)"]
explainer = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_test_proc)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:12]

print("Top 12 Most Influential Features Driving Fraud Prediction:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. {all_feature_names[idx]:35} : Mean |SHAP| = {mean_abs_shap[idx]:.4f}")

# Generate and Save High Quality Visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. ROC & PR Curves
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.2f})", lw=2)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
axes[0].set_title("ROC Curves Comparison", fontsize=14, fontweight='bold')
axes[0].set_xlabel("False Positive Rate", fontsize=11)
axes[0].set_ylabel("True Positive Rate", fontsize=11)
axes[0].legend(loc="lower right")

for name, p in probas.items():
    prec_pts, rec_pts, _ = precision_recall_curve(y_test, p)
    axes[1].plot(rec_pts, prec_pts, label=f"{name} (PR-AUC={auc(rec_pts, prec_pts):.2f})", lw=2)

axes[1].set_title("Precision-Recall Curves Comparison", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Recall", fontsize=11)
axes[1].set_ylabel("Precision", fontsize=11)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig('model_curves_comparison.png', dpi=300)
plt.close()
print("\n[+] Saved 'model_curves_comparison.png'")

# 2. Confusion Matrices Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cm_models = [
    ("Random Forest (Balanced)", probas["Random Forest (Balanced)"], 0.50),
    ("XGBoost (Default 0.50)", probas["XGBoost (SMOTE)"], 0.50),
    ("XGBoost (Optimized 0.25 Threshold)", probas["XGBoost (SMOTE)"], 0.25)
]

for idx, (m_title, p_vals, th) in enumerate(cm_models):
    preds = (p_vals >= th).astype(int)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Genuine (0)', 'Fraud (1)'],
                yticklabels=['Genuine (0)', 'Fraud (1)'])
    axes[idx].set_title(f"{m_title}\nRecall: {recall_score(y_test, preds):.1%} | Prec: {precision_score(y_test, preds):.1%}", fontsize=12)
    axes[idx].set_xlabel("Predicted", fontsize=11)
    axes[idx].set_ylabel("Actual", fontsize=11)

plt.tight_layout()
plt.savefig('confusion_matrices_advanced.png', dpi=300)
plt.close()
print("[+] Saved 'confusion_matrices_advanced.png'")

# 3. SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_proc, feature_names=all_feature_names, max_display=12, show=False)
plt.title("SHAP Feature Importance Summary (XGBoost)", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved 'shap_feature_importance.png'")

# 4. Save Cleaned Processed Dataset with Engineered Features
df_engineered.to_csv('df_engineered_advanced.csv', index=False)
print("[+] Saved engineered dataset as 'df_engineered_advanced.csv'")

print("\nAll advanced ML steps executed successfully!")
