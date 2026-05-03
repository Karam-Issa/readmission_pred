"""
shap_analysis.py
----------------
Generates SHAP (SHapley Additive exPlanations) values for the best model
(XGBoost) to explain which features drive readmission predictions.

Produces:
    - SHAP summary plot (feature importance + direction)
    - SHAP bar plot (mean absolute importance)
    - SHAP waterfall plot (single patient explanation)


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = "data/processed/"
MODELS_DIR = "models/"
PLOTS_DIR  = "notebooks/plots/"

# ---------------------------------------------------------------------------
# Load data and model
# ---------------------------------------------------------------------------

print("=" * 50)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 50)

X_train = pd.read_csv(f"{DATA_DIR}X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}X_test.csv")
y_test  = pd.read_csv(f"{DATA_DIR}y_test.csv").squeeze()

model = joblib.load(f"{MODELS_DIR}best_model.joblib")
print(f"\nLoaded model: {type(model).__name__}")
print(f"Test set: {X_test.shape}")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Compute SHAP values
# XGBoost has a native TreeExplainer — fast and exact
# ---------------------------------------------------------------------------

print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("Done.")

# ---------------------------------------------------------------------------
# Plot 1 — Summary Plot (beeswarm)
# Shows each feature's impact on the model output
# Red = high feature value, Blue = low feature value
# X-axis = SHAP value (positive = pushes toward readmission)
# ---------------------------------------------------------------------------

print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X_test.columns.tolist(),
    show=False
)
plt.title("SHAP Summary Plot — Feature Impact on Readmission Risk",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {PLOTS_DIR}shap_summary.png")

# ---------------------------------------------------------------------------
# Plot 2 — Bar Plot (mean absolute SHAP values)
# Clean ranking of feature importance
# ---------------------------------------------------------------------------

print("Generating SHAP bar plot...")
plt.figure(figsize=(9, 5))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X_test.columns.tolist(),
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance — Mean |SHAP Value|",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {PLOTS_DIR}shap_importance.png")

# ---------------------------------------------------------------------------
# Plot 3 — Waterfall Plot (single patient explanation)
# Pick a high-risk patient and explain exactly why
# ---------------------------------------------------------------------------

print("Generating SHAP waterfall plot (single patient)...")

# Find a true positive — a patient who was actually readmitted
# and the model correctly flagged as high risk
y_pred_proba = model.predict_proba(X_test)[:, 1]
true_positives = np.where((y_test.values == 1) & (y_pred_proba > 0.6))[0]

if len(true_positives) > 0:
    patient_idx = true_positives[0]
else:
    # fallback: highest predicted risk patient
    patient_idx = np.argmax(y_pred_proba)

patient_risk = y_pred_proba[patient_idx]
patient_data = X_test.iloc[patient_idx]

print(f"\nExplaining patient at index {patient_idx}:")
print(f"  Predicted readmission risk: {patient_risk:.1%}")
print(f"  Actual outcome: {'Readmitted' if y_test.iloc[patient_idx] == 1 else 'Not Readmitted'}")
print(f"  Patient features:")
for col, val in patient_data.items():
    print(f"    {col}: {val}")

# Waterfall plot
shap_explanation = shap.Explanation(
    values        = shap_values[patient_idx],
    base_values   = explainer.expected_value,
    data          = X_test.iloc[patient_idx].values,
    feature_names = X_test.columns.tolist()
)

plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_explanation, show=False)
plt.title(f"SHAP Waterfall — Single Patient (Risk: {patient_risk:.1%})",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {PLOTS_DIR}shap_waterfall.png")

# ---------------------------------------------------------------------------
# Print feature importance summary
# ---------------------------------------------------------------------------

mean_shap = pd.DataFrame({
    "feature":         X_test.columns.tolist(),
    "mean_abs_shap":   np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print(f"\n{'=' * 50}")
print("SHAP FEATURE IMPORTANCE RANKING")
print(f"{'=' * 50}")
print(mean_shap.to_string(index=False))

print(f"\n{'=' * 50}")
print("SHAP analysis complete. Plots saved to notebooks/plots/")
print(f"{'=' * 50}\n")