"""
preprocess.py
-------------
Cleans and prepares the parsed FHIR feature matrix for ML training.

Steps:
    1. Filter to adults (age >= 18)
    2. Cap LOS outliers at 30 days
    3. Encode categorical columns (gender, enc_class)
    4. Drop ID columns not used in training
    5. Train/test split (80/20, stratified)
    6. Save X_train, X_test, y_train, y_test to data/processed/


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_PATH  = "data/processed/features.csv"
OUTPUT_DIR  = "data/processed/"
RANDOM_SEED = 42
TEST_SIZE   = 0.2
LOS_CAP     = 30  # days — cap length of stay outliers at 30 days

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("=" * 50)
print("PREPROCESSING PIPELINE")
print("=" * 50)

df = pd.read_csv(INPUT_PATH)
print(f"\nLoaded {len(df):,} rows from {INPUT_PATH}")

# ---------------------------------------------------------------------------
# Step 1 — Filter to adults
# ---------------------------------------------------------------------------

before = len(df)
df = df[df["age"] >= 18].copy()
print(f"\nStep 1 — Filter to adults (age >= 18)")
print(f"  Dropped {before - len(df):,} pediatric rows")
print(f"  Remaining: {len(df):,} rows")

# ---------------------------------------------------------------------------
# Step 2 — Cap LOS outliers
# ---------------------------------------------------------------------------

before_max = df["length_of_stay_days"].max()
df["length_of_stay_days"] = df["length_of_stay_days"].clip(upper=LOS_CAP)
print(f"\nStep 2 — Cap length_of_stay_days at {LOS_CAP} days")
print(f"  Max LOS before: {before_max:.1f} days")
print(f"  Max LOS after:  {df['length_of_stay_days'].max():.1f} days")

# ---------------------------------------------------------------------------
# Step 3 — Encode categorical columns
# ---------------------------------------------------------------------------

print(f"\nStep 3 — Encode categorical columns")

# gender: female=0, male=1
df["gender_male"] = (df["gender"] == "male").astype(int)
print(f"  gender → gender_male (female=0, male=1)")

# enc_class: EMER=0, IMP=1
df["is_inpatient"] = (df["enc_class"] == "IMP").astype(int)
print(f"  enc_class → is_inpatient (EMER=0, IMP=1)")

# ---------------------------------------------------------------------------
# Step 4 — Select features for training
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "age",
    "gender_male",
    "length_of_stay_days",
    "prior_admissions",
    "num_conditions",
    "num_medications",
    "num_procedures",
    "has_prior_ed_visit",
    "is_inpatient",
]

TARGET_COL = "readmitted_30d"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\nStep 4 — Feature selection")
print(f"  Features: {FEATURE_COLS}")
print(f"  Target:   {TARGET_COL}")
print(f"  X shape:  {X.shape}")

# ---------------------------------------------------------------------------
# Step 5 — Train/test split
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y  # preserve class balance in both splits
)

print(f"\nStep 5 — Train/test split (80/20, stratified)")
print(f"  Train: {len(X_train):,} rows | Readmission rate: {y_train.mean():.1%}")
print(f"  Test:  {len(X_test):,} rows  | Readmission rate: {y_test.mean():.1%}")

# ---------------------------------------------------------------------------
# Step 6 — Save
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train.to_csv(f"{OUTPUT_DIR}X_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}y_test.csv", index=False)

print(f"\nStep 6 — Saved to {OUTPUT_DIR}")
print(f"  X_train.csv, X_test.csv, y_train.csv, y_test.csv")

print(f"\n{'=' * 50}")
print("Preprocessing complete.")
print(f"{'=' * 50}\n")