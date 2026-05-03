"""
train_baseline.py
-----------------
Trains a Logistic Regression baseline model for 30-day readmission prediction.
Evaluates with ROC-AUC, confusion matrix, and classification report.
Saves the trained model to models/baseline_lr.joblib

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
import joblib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "data/processed/"
MODELS_DIR  = "models/"
PLOTS_DIR   = "notebooks/plots/"
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("=" * 50)
print("BASELINE MODEL — LOGISTIC REGRESSION")
print("=" * 50)

X_train = pd.read_csv(f"{DATA_DIR}X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}y_test.csv").squeeze()

print(f"\nTraining set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")
print(f"Features:     {X_train.columns.tolist()}")

# ---------------------------------------------------------------------------
# Build pipeline
# StandardScaler is important for Logistic Regression —
# features on different scales (age=72 vs binary=0/1) need normalization
# ---------------------------------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",  # handles 90/10 class imbalance
        max_iter=1000,
        random_state=RANDOM_SEED,
    ))
])

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print("\nTraining Logistic Regression...")
pipeline.fit(X_train, y_train)
print("Done.")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

y_pred       = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
roc_auc      = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'=' * 50}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"{'=' * 50}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Not Readmitted", "Readmitted"]
))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix breakdown:")
print(f"  True Negatives  (correctly predicted no readmission): {tn:,}")
print(f"  False Positives (incorrectly flagged as readmitted):   {fp:,}")
print(f"  False Negatives (missed actual readmissions):          {fn:,}")
print(f"  True Positives  (correctly predicted readmission):     {tp:,}")

# ---------------------------------------------------------------------------
# Feature coefficients
# ---------------------------------------------------------------------------

feature_names = X_train.columns.tolist()
coefficients  = pipeline.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature":     feature_names,
    "coefficient": coefficients,
}).sort_values("coefficient", ascending=False)

print(f"\nFeature Coefficients (higher = stronger predictor of readmission):")
print(coef_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

RocCurveDisplay.from_predictions(
    y_test, y_pred_proba,
    ax=axes[0],
    name="Logistic Regression"
)
axes[0].plot([0, 1], [0, 1], "k--", label="Random classifier")
axes[0].set_title(f"ROC Curve (AUC = {roc_auc:.4f})")
axes[0].legend()

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Not Readmitted", "Readmitted"],
    cmap="Blues",
    ax=axes[1]
)
axes[1].set_title("Confusion Matrix — Logistic Regression")

plt.suptitle("Baseline Model Evaluation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}baseline_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved plot → {PLOTS_DIR}baseline_evaluation.png")

# Feature importance plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["tomato" if c > 0 else "steelblue" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression — Feature Coefficients")
ax.set_xlabel("Coefficient (positive = higher readmission risk)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}baseline_coefficients.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved plot → {PLOTS_DIR}baseline_coefficients.png")

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

model_path = f"{MODELS_DIR}baseline_lr.joblib"
joblib.dump(pipeline, model_path)
print(f"\nModel saved → {model_path}")

print(f"\n{'=' * 50}")
print(f"Baseline complete. ROC-AUC = {roc_auc:.4f}")
print(f"{'=' * 50}\n")