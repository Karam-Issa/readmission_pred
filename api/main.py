"""
main.py
-------
FastAPI REST API for 30-day hospital readmission prediction.
Loads the trained XGBoost model and exposes a /predict endpoint.

Usage:
    uvicorn api.main:app --reload

Endpoints:
    GET  /          → health check
    GET  /info      → model info
    POST /predict   → readmission risk prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import os

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------

MODEL_PATH = "models/best_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run src/train_advanced.py first to generate the model."
    )

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="30-Day Hospital Readmission Predictor",
    description=(
        "Predicts the probability that a patient will be readmitted "
        "to the hospital within 30 days of discharge. "
        "Built on Synthea-generated FHIR R4 data using XGBoost."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request schema
# Pydantic validates all inputs automatically
# ---------------------------------------------------------------------------

class PatientData(BaseModel):
    age: float = Field(..., ge=18, le=120, description="Patient age in years (18-120)")
    gender_male: int = Field(..., ge=0, le=1, description="Gender: 1=male, 0=female")
    length_of_stay_days: float = Field(..., ge=0, le=30, description="Length of stay in days (0-30)")
    prior_admissions: int = Field(..., ge=0, description="Number of prior inpatient admissions")
    num_conditions: int = Field(..., ge=0, description="Number of active conditions at admission")
    num_medications: int = Field(..., ge=0, description="Number of active medications at admission")
    num_procedures: int = Field(..., ge=0, description="Number of procedures during encounter")
    has_prior_ed_visit: int = Field(..., ge=0, le=1, description="ED visit in prior 6 months: 1=yes, 0=no")
    is_inpatient: int = Field(..., ge=0, le=1, description="Encounter type: 1=inpatient, 0=emergency")

    @field_validator("age")
    @classmethod
    def age_must_be_adult(cls, v):
        if v < 18:
            raise ValueError("Age must be 18 or older (pediatric patients not supported)")
        return round(v, 1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 64.7,
                    "gender_male": 1,
                    "length_of_stay_days": 5.54,
                    "prior_admissions": 12,
                    "num_conditions": 25,
                    "num_medications": 50,
                    "num_procedures": 11,
                    "has_prior_ed_visit": 1,
                    "is_inpatient": 1
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    readmission_risk: float = Field(..., description="Predicted probability of 30-day readmission (0-1)")
    risk_label: str = Field(..., description="Risk category: Low / Medium / High")
    threshold: float = Field(..., description="Decision threshold used for label")
    model: str = Field(..., description="Model used for prediction")


# ---------------------------------------------------------------------------
# Risk labeling helper
# ---------------------------------------------------------------------------

def get_risk_label(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "message": "30-Day Readmission Predictor API is running",
        "docs": "/docs"
    }


@app.get("/info", summary="Model information")
def model_info():
    return {
        "model_type": type(model).__name__,
        "model_path": MODEL_PATH,
        "features": [
            "age",
            "gender_male",
            "length_of_stay_days",
            "prior_admissions",
            "num_conditions",
            "num_medications",
            "num_procedures",
            "has_prior_ed_visit",
            "is_inpatient",
        ],
        "target": "readmitted_30d",
        "training_data": "Synthea-generated FHIR R4 (10,000 patients, Massachusetts)",
        "roc_auc": 0.9053,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict readmission risk")
def predict(patient: PatientData):
    """
    Accepts patient data at time of discharge and returns a
    30-day readmission risk score between 0 and 1.

    Risk labels:
    - **Low Risk**: probability < 0.30
    - **Medium Risk**: probability 0.30 – 0.60
    - **High Risk**: probability > 0.60
    """
    try:
        # Build feature array in the exact order the model was trained on
        features = np.array([[
            patient.age,
            patient.gender_male,
            patient.length_of_stay_days,
            patient.prior_admissions,
            patient.num_conditions,
            patient.num_medications,
            patient.num_procedures,
            patient.has_prior_ed_visit,
            patient.is_inpatient,
        ]])

        probability = float(model.predict_proba(features)[0][1])
        probability = round(probability, 4)

        return PredictionResponse(
            readmission_risk=probability,
            risk_label=get_risk_label(probability),
            threshold=0.5,
            model=type(model).__name__,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")