"""
==============================================================================
Customer Churn & Monthly Charges: End-to-End ML Pipeline
Classification (KNN) + Regression (KNN) + Model Serialization
==============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
csv_filename = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

if os.path.exists(csv_filename):
    print(f"Loading dataset from: {csv_filename}")
    df = pd.read_csv(csv_filename)
else:
    print(f"'{csv_filename}' not found. Generating realistic synthetic dataset...")
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'customerID': [f'CUST-{i:05d}' for i in range(n)],
        'gender': np.random.choice(['Female', 'Male'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n),
        'tenure': np.random.randint(1, 73, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['No phone service', 'No', 'Yes'], n),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n),
        'MonthlyCharges': np.round(np.random.uniform(20.0, 118.0, n), 2),
    })
    df['TotalCharges'] = np.round(df['tenure'] * df['MonthlyCharges'] + np.random.normal(0, 15, n), 2)
    churn_prob = 0.35 + (df['MonthlyCharges'] / 300) - (df['tenure'] / 120)
    churn_prob += np.where(df['Contract'] == 'Month-to-month', 0.2, -0.15)
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    df['Churn'] = np.where(np.random.rand(n) < churn_prob, 'Yes', 'No')

print(f"Data Loaded! Shape: {df.shape}")

# ------------------------------------------------------------------------------
# 2. Data Cleaning & Encoding
# ------------------------------------------------------------------------------
df.drop('customerID', axis=1, inplace=True, errors='ignore')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0.0, inplace=True)
df.drop_duplicates(inplace=True)

# Encode Target for Classification
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])  # No -> 0, Yes -> 1

# One-hot encode remaining categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# ==============================================================================
# PART 1: CLASSIFICATION (Predicting Churn)
# ==============================================================================
print("\n" + "="*50)
print("1. CLASSIFICATION: PREDICTING CHURN (KNN)")
print("="*50)

X_clf = df_encoded.drop('Churn', axis=1)
y_clf = df_encoded['Churn']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# Cross-Validation to select optimal K
cv_acc_scores = []
for k in range(1, 21):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train_c_scaled, y_train_c, cv=5, scoring='accuracy')
    cv_acc_scores.append(scores.mean())

best_k_clf = np.argmax(cv_acc_scores) + 1
print(f"Best K for Classification: {best_k_clf}")

# Train final KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=best_k_clf)
knn_clf.fit(X_train_c_scaled, y_train_c)

y_pred_c = knn_clf.predict(X_test_c_scaled)
y_proba_c = knn_clf.predict_proba(X_test_c_scaled)[:, 1]
auc_score = roc_auc_score(y_test_c, y_proba_c)

print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print(f"ROC-AUC Score: {auc_score:.4f}")
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c))

# --- Plot Side-by-Side: Confusion Matrix and ROC Curve ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay (0)", "Churn (1)"])
disp.plot(ax=axes[0], cmap="Blues", values_format="d")
axes[0].set_title("Confusion Matrix")
axes[0].grid(False)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_c, y_proba_c)
axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.2f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance (AUC = 0.50)")
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate (Recall)")
axes[1].set_title("Receiver Operating Characteristic (ROC) Curve")
axes[1].legend(loc="lower right")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# ==============================================================================
# PART 2: REGRESSION (Predicting Monthly Charges)
# ==============================================================================
print("\n" + "="*50)
print("2. REGRESSION: PREDICTING MONTHLY CHARGES (KNN)")
print("="*50)

X_reg = df_encoded.drop(['MonthlyCharges', 'Churn'], axis=1)
y_reg = df['MonthlyCharges']

print(f"Features Shape (X_reg): {X_reg.shape}")
print(f"Target Shape (y_reg): {y_reg.shape}")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

# Cross-Validation to select optimal K for regression
cv_mse_scores = []
for k in range(1, 21):
    knn_reg_temp = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(
        knn_reg_temp, X_train_r_scaled, y_train_r, cv=5, scoring='neg_mean_squared_error'
    )
    cv_mse_scores.append(-scores.mean())

best_k_reg = np.argmin(cv_mse_scores) + 1
print(f"Best K for Regression: {best_k_reg}")

# --- Plot Cross-Validation Curve (Regression) ---
plt.figure(figsize=(8, 4))
plt.plot(range(1, 21), cv_mse_scores, marker='o', color='crimson')
plt.xlabel("K Value")
plt.ylabel("Mean Squared Error")
plt.title("Cross Validation Curve (Regression)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Train final KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=best_k_reg)
knn_reg.fit(X_train_r_scaled, y_train_r)

y_pred_r = knn_reg.predict(X_test_r_scaled)

mae = mean_absolute_error(y_test_r, y_pred_r)
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_r)

print("\n--- Regression Performance ---")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# --- Plot Actual vs. Predicted & Residual Distribution ---
residuals = y_test_r - y_pred_r
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Actual vs. Predicted
axes[0].scatter(y_test_r, y_pred_r, color="teal", alpha=0.5, edgecolor="k", s=30)
min_v = min(y_test_r.min(), y_pred_r.min())
max_v = max(y_test_r.max(), y_pred_r.max())
axes[0].plot([min_v, max_v], [min_v, max_v], color="crimson", lw=2, linestyle="--", label="Perfect Fit (y = x)")
axes[0].set_xlabel("Actual Monthly Charges ($)")
axes[0].set_ylabel("Predicted Monthly Charges ($)")
axes[0].set_title("Actual vs. Predicted Monthly Charges")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Subplot 2: Residuals Distribution
axes[1].hist(residuals, bins=30, color="cornflowerblue", edgecolor="black", alpha=0.7)
axes[1].axvline(0, color="crimson", linestyle="--", lw=2, label="Zero Error Line")
axes[1].set_xlabel("Residual Error (Actual - Predicted)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Residual Errors")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# ==============================================================================
# PART 3: SAVE MODEL ARTIFACTS
# ==============================================================================
artifacts = {
    'classification_model': knn_clf,
    'classification_scaler': scaler_c,
    'classification_features': list(X_clf.columns),
    'regression_model': knn_reg,
    'regression_scaler': scaler_r,
    'regression_features': list(X_reg.columns)
}

joblib.dump(artifacts, "customer_churn_pipeline.joblib")
print("\n" + "="*50)
print("Pipeline artifacts saved to: customer_churn_pipeline.joblib")
print("="*50)