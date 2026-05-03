"""
test_pipeline.py
----------------
Unit tests for the readmission predictor pipeline and API.

Tests:
    - FHIR parser (date parsing, readmission label logic)
    - Preprocessing (adult filter, LOS cap, encoding)
    - API endpoint (valid input, invalid input, response schema)

Usage:
    pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# FHIR Parser tests
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, "src")
from fhir_parser import parse_dt, get_age, is_inpatient


class TestParseDt:
    def test_full_iso_with_timezone(self):
        dt = parse_dt("1988-07-07T19:52:12-04:00")
        assert dt is not None
        assert dt.year == 1988
        assert dt.month == 7

    def test_date_only(self):
        dt = parse_dt("1979-07-26")
        assert dt is not None
        assert dt.year == 1979

    def test_none_input(self):
        assert parse_dt(None) is None

    def test_empty_string(self):
        assert parse_dt("") is None

    def test_returns_timezone_aware(self):
        dt = parse_dt("2020-01-01T00:00:00+00:00")
        assert dt.tzinfo is not None


class TestGetAge:
    def test_basic_age(self):
        birth = "1980-01-01"
        ref   = datetime(2020, 1, 1, tzinfo=timezone.utc)
        age   = get_age(birth, ref)
        assert 39.9 < age < 40.1

    def test_none_birth(self):
        ref = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert get_age(None, ref) is None

    def test_none_reference(self):
        assert get_age("1980-01-01", None) is None


class TestIsInpatient:
    def test_imp_is_inpatient(self):
        assert is_inpatient({"enc_class": "IMP"}) is True

    def test_emer_is_inpatient(self):
        assert is_inpatient({"enc_class": "EMER"}) is True

    def test_amb_is_not_inpatient(self):
        assert is_inpatient({"enc_class": "AMB"}) is False

    def test_unknown_is_not_inpatient(self):
        assert is_inpatient({"enc_class": "OTHER"}) is False


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:

    @pytest.fixture
    def sample_df(self):
        """Create a small sample dataframe mimicking features.csv"""
        return pd.DataFrame({
            "patient_id":          ["p1", "p2", "p3", "p4"],
            "encounter_id":        ["e1", "e2", "e3", "e4"],
            "age":                 [10.0, 25.0, 45.0, 70.0],
            "gender":              ["male", "female", "male", "female"],
            "length_of_stay_days": [1.0, 5.0, 3834.0, 2.0],
            "prior_admissions":    [0, 1, 2, 5],
            "num_conditions":      [0, 3, 8, 15],
            "num_medications":     [0, 5, 20, 50],
            "num_procedures":      [0, 2, 4, 6],
            "has_prior_ed_visit":  [0, 0, 1, 1],
            "enc_class":           ["EMER", "IMP", "EMER", "IMP"],
            "readmitted_30d":      [0, 0, 1, 1],
        })

    def test_adult_filter(self, sample_df):
        filtered = sample_df[sample_df["age"] >= 18]
        assert len(filtered) == 3
        assert 10.0 not in filtered["age"].values

    def test_los_cap(self, sample_df):
        sample_df["length_of_stay_days"] = sample_df["length_of_stay_days"].clip(upper=30)
        assert sample_df["length_of_stay_days"].max() == 30.0

    def test_gender_encoding(self, sample_df):
        sample_df["gender_male"] = (sample_df["gender"] == "male").astype(int)
        assert sample_df.loc[0, "gender_male"] == 1  # male
        assert sample_df.loc[1, "gender_male"] == 0  # female

    def test_enc_class_encoding(self, sample_df):
        sample_df["is_inpatient"] = (sample_df["enc_class"] == "IMP").astype(int)
        assert sample_df.loc[0, "is_inpatient"] == 0  # EMER
        assert sample_df.loc[1, "is_inpatient"] == 1  # IMP

    def test_no_missing_values(self, sample_df):
        assert sample_df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

import os
# Only run API tests if model exists
MODEL_EXISTS = os.path.exists("models/best_model.joblib")

@pytest.mark.skipif(not MODEL_EXISTS, reason="Model file not found — run train_advanced.py first")
class TestAPI:

    @pytest.fixture
    def client(self):
        from api.main import app
        return TestClient(app)

    @pytest.fixture
    def valid_patient(self):
        return {
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

    def test_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_model_info(self, client):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "roc_auc" in data
        assert data["roc_auc"] == 0.9053

    def test_predict_valid_input(self, client, valid_patient):
        response = client.post("/predict", json=valid_patient)
        assert response.status_code == 200
        data = response.json()
        assert "readmission_risk" in data
        assert "risk_label" in data
        assert 0.0 <= data["readmission_risk"] <= 1.0
        assert data["risk_label"] in ["Low Risk", "Medium Risk", "High Risk"]

    def test_predict_low_risk_patient(self, client):
        low_risk = {
            "age": 25.0,
            "gender_male": 0,
            "length_of_stay_days": 0.5,
            "prior_admissions": 0,
            "num_conditions": 1,
            "num_medications": 1,
            "num_procedures": 0,
            "has_prior_ed_visit": 0,
            "is_inpatient": 0
        }
        response = client.post("/predict", json=low_risk)
        assert response.status_code == 200
        data = response.json()
        assert data["readmission_risk"] < 0.5
        assert data["risk_label"] == "Low Risk"

    def test_predict_missing_field(self, client, valid_patient):
        del valid_patient["age"]
        response = client.post("/predict", json=valid_patient)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_age(self, client, valid_patient):
        valid_patient["age"] = 10  # under 18
        response = client.post("/predict", json=valid_patient)
        assert response.status_code == 422

    def test_predict_invalid_gender(self, client, valid_patient):
        valid_patient["gender_male"] = 5  # must be 0 or 1
        response = client.post("/predict", json=valid_patient)
        assert response.status_code == 422

    def test_response_model_field(self, client, valid_patient):
        response = client.post("/predict", json=valid_patient)
        data = response.json()
        assert data["model"] == "XGBClassifier"
        assert data["threshold"] == 0.5