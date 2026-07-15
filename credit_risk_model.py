"""
LTFS Vehicle Loan Default Prediction
=====================================
A credit risk model predicting probability of default on vehicle loans,
using real historical data from LTFS (L&T Financial Services), one of
India's leading NBFCs.

Dataset: L&T Vehicle Loan Default Prediction (Kaggle)
https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction

WHAT THIS SCRIPT DOES:
1. Load and explore the data
2. Engineer features from raw columns (convert text -> numbers, create
   meaningful flags for missing/special values)
3. Split into train/test sets
4. Train two models: Logistic Regression (interpretable baseline) and
   Random Forest (non-linear comparison)
5. Evaluate using metrics appropriate for imbalanced classification
   (AUC-ROC, precision/recall - NOT plain accuracy)
6. Interpret which factors drive default risk
7. Generate visualizations: ROC curve, feature importance, default rate by LTV

HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib
    python credit_risk_model.py

Before running: download train.csv from the Kaggle link above, then either
place it in the same folder as this script, or update DATA_PATH below to
point to wherever you saved it.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # renders to file without needing a display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)

pd.set_option('display.max_columns', None)

# ============================================================
# CONFIG - change this to match where YOU saved train.csv
# ============================================================
# Default assumes train.csv sits in the same folder as this script.
# Windows example: DATA_PATH = r'C:\Users\yourname\Downloads\train.csv'
# Mac/Linux example: DATA_PATH = '/Users/yourname/Downloads/train.csv'
DATA_PATH = 'train.csv'

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} loans")
print(f"Default rate: {df['loan_default'].mean()*100:.1f}%")
print("(This is an imbalanced classification problem - ~1 in 5 loans")
print(" default. This shapes how we evaluate the model later.)\n")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
# Real-world data rarely comes ready for modeling. Each step below
# converts a raw column into something a model can actually use.

# --- 2a. Age at disbursal, from Date of Birth + Disbursal Date ---
df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'], format='%d-%m-%y', errors='coerce')
# Two-digit years (e.g. '84') parse as 2084 by default; fix impossible
# future birth years back into the 1900s.
df.loc[df['Date.of.Birth'].dt.year > 2005, 'Date.of.Birth'] -= pd.DateOffset(years=100)
df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], format='%d-%m-%y', errors='coerce')
df['age_at_disbursal'] = (df['DisbursalDate'] - df['Date.of.Birth']).dt.days / 365.25

# --- 2b. Convert "Xyrs Ymon" text fields into total months ---
def parse_years_months(text):
    """Converts '1yrs 11mon' -> 23 (total months)."""
    try:
        years = int(text.split('yrs')[0].strip())
        months = int(text.split('yrs')[1].replace('mon', '').strip())
        return years * 12 + months
    except (ValueError, AttributeError, IndexError):
        return np.nan

df['avg_acct_age_months'] = df['AVERAGE.ACCT.AGE'].apply(parse_years_months)
df['credit_history_months'] = df['CREDIT.HISTORY.LENGTH'].apply(parse_years_months)

# --- 2c. Bureau score: separate "has a score" from "the score value" ---
# A raw score of 0 means "No Bureau History Available", NOT "terrible score".
# ~50% of borrowers have no bureau history - common in the Indian lending
# market. Feeding 0 directly into the model would wrongly teach it that
# 0 = worst possible risk, so we split this into two features instead.
df['has_bureau_score'] = (df['PERFORM_CNS.SCORE'] > 0).astype(int)
df['bureau_score'] = df['PERFORM_CNS.SCORE'].where(df['PERFORM_CNS.SCORE'] > 0, np.nan)

# --- 2d. Employment type ---
df['Employment.Type'] = df['Employment.Type'].fillna('Unknown')
df['is_self_employed'] = (df['Employment.Type'] == 'Self employed').astype(int)

# --- 2e. Document count (proxy for borrower formality/verification) ---
doc_cols = ['Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']
df['num_documents'] = df[doc_cols].sum(axis=1)

print("Data quality note: {:,} loans ({:.1f}%) have no bureau history.".format(
    (df['has_bureau_score'] == 0).sum(),
    (df['has_bureau_score'] == 0).mean() * 100
))
print()

# ============================================================
# STEP 3: SELECT FEATURES
# ============================================================
# A focused, interpretable feature set - easier to explain and defend
# than throwing in every available column.
feature_cols = [
    'disbursed_amount', 'asset_cost', 'ltv',
    'age_at_disbursal', 'num_documents',
    'is_self_employed', 'has_bureau_score', 'bureau_score',
    'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS',
    'PRI.CURRENT.BALANCE', 'PRIMARY.INSTAL.AMT',
    'avg_acct_age_months', 'credit_history_months',
    'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
    'NO.OF_INQUIRIES',
]

X = df[feature_cols].copy()
y = df['loan_default'].copy()

# Fill remaining missing values (e.g. bureau_score where no history exists,
# or unparseable account-age text) with the median - a simple, defensible
# default for a baseline model.
X = X.fillna(X.median())

# ============================================================
# STEP 4: TRAIN / TEST SPLIT
# ============================================================
# Hold out 20% the model never sees during training, to honestly evaluate
# how it performs on "new" loans. stratify=y keeps the same ~21.7% default
# rate in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {len(X_train):,} loans | Test set: {len(X_test):,} loans\n")

# ============================================================
# STEP 5: TRAIN MODEL 1 - LOGISTIC REGRESSION (baseline)
# ============================================================
# The industry-standard starting point for credit risk scoring: simple,
# fast, and interpretable (each feature gets a coefficient that can be
# explained to a non-technical credit committee).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
# class_weight='balanced' makes the model pay more attention to the
# minority class (defaults). Without this, a model could reach 78%
# "accuracy" just by predicting "no default" every time - useless in
# practice.
logreg.fit(X_train_scaled, y_train)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
logreg_pred = logreg.predict(X_test_scaled)
logreg_auc = roc_auc_score(y_test, logreg_proba)

# ============================================================
# STEP 6: TRAIN MODEL 2 - RANDOM FOREST (comparison)
# ============================================================
# A non-linear ensemble model, used to check whether feature interactions
# (e.g. "high LTV is only risky when the borrower ALSO has no bureau
# history") meaningfully improve on the linear baseline.
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)  # Random Forest doesn't require feature scaling
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)

# ============================================================
# STEP 7: EVALUATE
# ============================================================
# WHY NOT JUST "ACCURACY"? With a 78%-no-default / 22%-default split, a
# model that always predicts "no default" gets 78% accuracy while catching
# zero real defaults. Instead we use:
#   - AUC-ROC: how well the model ranks defaulters above non-defaulters
#     (0.5 = random guessing, 1.0 = perfect)
#   - Precision/Recall: of loans flagged risky, how many actually defaulted
#     (precision); of all actual defaults, how many were caught (recall)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"Logistic Regression AUC-ROC: {logreg_auc:.3f}")
print(f"Random Forest AUC-ROC:       {rf_auc:.3f}")
print(f"Improvement: {(rf_auc - logreg_auc)*100:.1f} percentage points")
print("(0.5 = random guessing, 1.0 = perfect. Published benchmarks on this")
print(" dataset typically reach 0.66-0.68 even with heavy tuning - real")
print(" credit data is genuinely noisy.)\n")

print("Logistic Regression - Classification Report (0.5 threshold):")
print(classification_report(y_test, logreg_pred, target_names=['No Default', 'Default']))

cm = confusion_matrix(y_test, logreg_pred)
print("Confusion Matrix (Logistic Regression):")
print(f"                 Predicted No-Default   Predicted Default")
print(f"Actual No-Default        {cm[0][0]:>8,}              {cm[0][1]:>8,}")
print(f"Actual Default            {cm[1][0]:>8,}              {cm[1][1]:>8,}\n")

# ============================================================
# STEP 8: INTERPRET - WHICH FACTORS DRIVE DEFAULT RISK
# ============================================================
logreg_coef = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': logreg.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Logistic Regression coefficients (positive = increases default risk,")
print("negative = decreases default risk / protective factor):")
print(logreg_coef.to_string(index=False))
print()
print("Random Forest feature importance (top drivers, no direction implied):")
print(rf_importance.to_string(index=False))
print()

logreg_coef.to_csv('logreg_coefficients.csv', index=False)
rf_importance.to_csv('rf_feature_importance.csv', index=False)

# ============================================================
# STEP 9: VISUALIZATIONS
# ============================================================

# --- 9a. ROC Curve comparing both models ---
fig, ax = plt.subplots(figsize=(7, 6))
for name, proba, auc_val, color in [
    ('Logistic Regression', logreg_proba, logreg_auc, '#2563eb'),
    ('Random Forest', rf_proba, rf_auc, '#dc2626'),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})', linewidth=2, color=color)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guessing (AUC = 0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve: Vehicle Loan Default Prediction')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.close()

# --- 9b. Feature importance (Random Forest, top 10) ---
fig, ax = plt.subplots(figsize=(8, 6))
top_features = rf_importance.head(10).sort_values('Importance')
ax.barh(top_features['Feature'], top_features['Importance'], color='#2563eb')
ax.set_xlabel('Importance')
ax.set_title('Top 10 Drivers of Default Risk (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()

# --- 9c. Default rate by LTV bucket (clearest business-facing chart) ---
df['ltv_bucket'] = pd.cut(df['ltv'], bins=[0, 60, 70, 80, 90, 100],
                            labels=['<60%', '60-70%', '70-80%', '80-90%', '90%+'])
default_by_ltv = df.groupby('ltv_bucket', observed=True)['loan_default'].mean() * 100

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(default_by_ltv.index.astype(str), default_by_ltv.values, color='#dc2626')
ax.set_xlabel('Loan-to-Value (LTV) Ratio')
ax.set_ylabel('Default Rate (%)')
ax.set_title('Default Rate Increases Sharply with LTV Ratio')
for i, v in enumerate(default_by_ltv.values):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('default_by_ltv.png', dpi=150)
plt.close()

print("Saved: roc_curve.png, feature_importance.png, default_by_ltv.png")
print("Saved: logreg_coefficients.csv, rf_feature_importance.csv")
print("\nDone.")
