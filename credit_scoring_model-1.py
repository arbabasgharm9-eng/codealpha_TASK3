# Credit Scoring Model
# Predicts creditworthiness of applicants using historical financial data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Generate Synthetic Dataset
# ─────────────────────────────────────────────
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "income":            np.random.randint(20000, 120000, n),
    "debt":              np.random.randint(1000,  50000,  n),
    "credit_history":    np.random.randint(0, 10, n),          # years
    "employment_status": np.random.randint(0, 2, n),           # 0=unemployed, 1=employed
    "age":               np.random.randint(21, 65, n),
})

# Label: 1 = Good credit, 0 = Bad credit
data["label"] = (
    (data["income"] > 50000) &
    (data["debt"] < 20000) &
    (data["credit_history"] > 3)
).astype(int)

print("Dataset shape:", data.shape)
print("Class distribution:\n", data["label"].value_counts())
print()


# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────

# Handle missing values (none here, but shown for real-world use)
data.fillna(data.median(numeric_only=True), inplace=True)

X = data.drop("label", axis=1)
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ─────────────────────────────────────────────
# 3. Define Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(eval_metric="logloss", random_state=42),
}


# ─────────────────────────────────────────────
# 4. Train and Evaluate
# ─────────────────────────────────────────────
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'ROC-AUC':>10}")
print("-" * 65)

results = {}
for name, model in models.items():
    # Tree-based models don't need scaling
    if name in ("Logistic Regression",):
        model.fit(X_train_scaled, y_train)
        y_pred  = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "ROC-AUC": auc}
    print(f"{name:<22} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {auc:>10.4f}")

print()


# ─────────────────────────────────────────────
# 5. Best Model Summary
# ─────────────────────────────────────────────
best_model = max(results, key=lambda m: results[m]["ROC-AUC"])
print(f"Best Model by ROC-AUC: {best_model}")
print(f"  ROC-AUC   : {results[best_model]['ROC-AUC']:.4f}")
print(f"  Accuracy  : {results[best_model]['Accuracy']:.4f}")
print(f"  Precision : {results[best_model]['Precision']:.4f}")
print(f"  Recall    : {results[best_model]['Recall']:.4f}")
