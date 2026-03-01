import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ==============================
# Load Dataset
# ==============================

DATA_PATH = "data/Palo Alto Networks.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# ==============================
# Feature Engineering
# ==============================

# Income-to-Experience Ratio
df["IncomeExperienceRatio"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

# Promotion Delay Indicator
df["PromotionDelay"] = (df["YearsSinceLastPromotion"] > 3).astype(int)

# Engagement Score
df["EngagementScore"] = (
    df["JobInvolvement"] +
    df["JobSatisfaction"] +
    df["RelationshipSatisfaction"]
)

# Workload Stress Flag
df["WorkStressFlag"] = (
    (df["OverTime"] == "Yes") &
    (df["WorkLifeBalance"] <= 2)
).astype(int)

# ==============================
# Encode Categorical Variables
# ==============================

categorical_cols = df.select_dtypes(include=["object"]).columns

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ==============================
# Split Features & Target
# ==============================

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ==============================
# Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# Scaling
# ==============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Model Training
# ==============================

# Logistic Regression (Baseline)
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_scaled, y_train)

# Random Forest (Primary)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

# ==============================
# Evaluation
# ==============================

# Get probabilities first
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Adjust threshold for HR-focused recall
threshold = 0.30
y_pred = (y_prob >= threshold).astype(int)

print("\n--- Model Evaluation ---")
print("Threshold Used:", threshold)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# Save Feature Names
# ==============================

feature_columns = X.columns
os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/attrition_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

# ==============================
# Feature Importance
# ==============================

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"][:15],
         feature_importance_df["Importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png")

# ==============================
# SHAP Explainability (SAFE VERSION)
# ==============================

try:
    import shap

    print("\nGenerating SHAP summary plot...")

    # Use a smaller sample for performance
    X_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)

    explainer = shap.Explainer(rf_model, X_train)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    plt.close()

    print("SHAP summary saved successfully.")

except Exception as e:
    print("SHAP generation failed:", e)