"""
train_advanced.py
-----------------
Trains Random Forest and XGBoost models for 30-day readmission prediction.
Compares against the Logistic Regression baseline.
Saves the best model to models/best_model.joblib


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "data/processed/"
MODELS_DIR  = "models/"
PLOTS_DIR   = "notebooks/plots/"
RANDOM_SEED = 42
BASELINE_AUC = 0.8467  # from Logistic Regression

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("=" * 50)
print("ADVANCED MODELS — RANDOM FOREST + XGBOOST")
print("=" * 50)

X_train = pd.read_csv(f"{DATA_DIR}X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}y_test.csv").squeeze()

print(f"\nTraining set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")

# ---------------------------------------------------------------------------
# Helper — evaluate any trained model
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test):
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc      = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{'=' * 50}")
    print(f"{name} — ROC-AUC: {roc_auc:.4f}", end="")
    if roc_auc > BASELINE_AUC:
        print(f"  ✅ beats baseline by +{roc_auc - BASELINE_AUC:.4f}")
    else:
        print(f"  ❌ below baseline by {roc_auc - BASELINE_AUC:.4f}")
    print(f"{'=' * 50}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Readmitted", "Readmitted"]
    ))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix breakdown:")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives:  {tp:,}")

    return roc_auc, y_pred, y_pred_proba


# ---------------------------------------------------------------------------
# 1. Random Forest
# ---------------------------------------------------------------------------

print("\n--- RANDOM FOREST ---")
print("Training... (this may take 1-2 minutes)")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,  # use all CPU cores
)
rf.fit(X_train, y_train)
print("Done.")

rf_auc, rf_pred, rf_proba = evaluate_model("Random Forest", rf, X_test, y_test)

# ---------------------------------------------------------------------------
# 2. XGBoost
# ---------------------------------------------------------------------------

print("\n--- XGBOOST ---")
print("Training...")

# Calculate scale_pos_weight for class imbalance
# = count(negative) / count(positive)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.2f} (handles class imbalance)")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    eval_metric="auc",
    verbosity=0,
)
xgb.fit(X_train, y_train)
print("Done.")

xgb_auc, xgb_pred, xgb_proba = evaluate_model("XGBoost", xgb, X_test, y_test)

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

print(f"\n{'=' * 50}")
print("MODEL COMPARISON SUMMARY")
print(f"{'=' * 50}")
print(f"{'Model':<25} {'ROC-AUC':>10} {'vs Baseline':>12}")
print(f"{'-' * 50}")
print(f"{'Logistic Regression':<25} {BASELINE_AUC:>10.4f} {'(baseline)':>12}")
print(f"{'Random Forest':<25} {rf_auc:>10.4f} {rf_auc - BASELINE_AUC:>+12.4f}")
print(f"{'XGBoost':<25} {xgb_auc:>10.4f} {xgb_auc - BASELINE_AUC:>+12.4f}")

best_name  = "Random Forest" if rf_auc >= xgb_auc else "XGBoost"
best_model = rf if rf_auc >= xgb_auc else xgb
best_auc   = max(rf_auc, xgb_auc)
print(f"\n🏆 Best model: {best_name} (AUC = {best_auc:.4f})")

# ---------------------------------------------------------------------------
# ROC Curve comparison plot
# ---------------------------------------------------------------------------

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Load baseline probabilities for comparison
baseline = joblib.load(f"{MODELS_DIR}baseline_lr.joblib")
baseline_proba = baseline.predict_proba(X_test)[:, 1]
baseline_auc   = roc_auc_score(y_test, baseline_proba)

ax = axes[0]
RocCurveDisplay.from_predictions(y_test, baseline_proba, ax=ax, name=f"Logistic Regression (AUC={baseline_auc:.4f})")
RocCurveDisplay.from_predictions(y_test, rf_proba,       ax=ax, name=f"Random Forest (AUC={rf_auc:.4f})")
RocCurveDisplay.from_predictions(y_test, xgb_proba,      ax=ax, name=f"XGBoost (AUC={xgb_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_title("ROC Curve — All Models")
ax.legend(fontsize=8)

# Feature importance — best model
ax = axes[1]
feature_names = X_train.columns.tolist()

if best_name == "Random Forest":
    importances = rf.feature_importances_
else:
    importances = xgb.feature_importances_

imp_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": importances,
}).sort_values("importance", ascending=True)

ax.barh(imp_df["feature"], imp_df["importance"], color="steelblue", edgecolor="white")
ax.set_title(f"Feature Importance — {best_name}")
ax.set_xlabel("Importance")

plt.suptitle("Advanced Model Evaluation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}advanced_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved plot → {PLOTS_DIR}advanced_evaluation.png")

# ---------------------------------------------------------------------------
# Save best model
# ---------------------------------------------------------------------------

best_path = f"{MODELS_DIR}best_model.joblib"
joblib.dump(best_model, best_path)
print(f"Best model saved → {best_path}")

print(f"\n{'=' * 50}")
print(f"Done. Best model: {best_name} | AUC = {best_auc:.4f}")
print(f"{'=' * 50}\n")