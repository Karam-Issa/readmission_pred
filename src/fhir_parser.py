"""
fhir_parser.py
--------------
Parses Synthea-generated FHIR R4 JSON bundles into a flat feature matrix
suitable for ML training. Derives the 30-day readmission label from
Encounter dates.

Usage:
    python fhir_parser.py --input data/raw/fhir/ --output data/processed/features.csv

Features extracted per patient:
    - age (at time of index encounter)
    - gender
    - length_of_stay (days)
    - prior_inpatient_admissions (count before index encounter)
    - num_conditions (active conditions at time of encounter)
    - num_medications (active medication requests)
    - num_procedures (procedures during encounter)
    - has_emergency_visit (binary: any ED visit in prior 6 months)
    - readmitted_30d (TARGET LABEL)
"""

import json
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_dt(dt_str: str) -> datetime:
    """Parse ISO datetime string to timezone-aware datetime."""
    if not dt_str:
        return None
    # Handle various ISO formats Synthea produces
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(dt_str[:25], fmt[:len(dt_str[:25])])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def get_age(birth_date_str: str, reference_dt: datetime) -> float:
    """Calculate age in years at reference date."""
    bd = parse_dt(birth_date_str)
    if not bd or not reference_dt:
        return None
    if bd.tzinfo is None:
        bd = bd.replace(tzinfo=timezone.utc)
    return (reference_dt - bd).days / 365.25


# ---------------------------------------------------------------------------
# Resource extractors
# ---------------------------------------------------------------------------

def extract_patient(entries: list) -> dict:
    """Extract demographic features from Patient resource."""
    for entry in entries:
        r = entry["resource"]
        if r["resourceType"] == "Patient":
            return {
                "patient_id": r.get("id"),
                "gender": r.get("gender", "unknown"),
                "birth_date": r.get("birthDate"),
            }
    return {}


def extract_encounters(entries: list) -> list:
    """Extract all Encounter resources as a sorted list."""
    encounters = []
    for entry in entries:
        r = entry["resource"]
        if r["resourceType"] != "Encounter":
            continue

        period = r.get("period", {})
        start = parse_dt(period.get("start"))
        end = parse_dt(period.get("end"))
        if not start:
            continue

        enc_class = r.get("class", {}).get("code", "AMB")
        enc_type = r.get("type", [{}])[0].get("text", "") if r.get("type") else ""

        encounters.append({
            "encounter_id": r.get("id"),
            "enc_class": enc_class,
            "enc_type": enc_type,
            "start": start,
            "end": end,
            "los_hours": (end - start).total_seconds() / 3600 if end else 0,
        })

    return sorted(encounters, key=lambda x: x["start"])


def extract_conditions(entries: list) -> list:
    """Extract Condition resources."""
    conditions = []
    for entry in entries:
        r = entry["resource"]
        if r["resourceType"] != "Condition":
            continue
        onset = parse_dt(r.get("onsetDateTime") or r.get("recordedDate"))
        code = r.get("code", {}).get("coding", [{}])[0].get("code", "")
        text = r.get("code", {}).get("text", "")
        status = r.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "")
        conditions.append({
            "code": code,
            "text": text,
            "onset": onset,
            "clinical_status": status,
        })
    return conditions


def extract_medications(entries: list) -> list:
    """Extract MedicationRequest resources."""
    meds = []
    for entry in entries:
        r = entry["resource"]
        if r["resourceType"] != "MedicationRequest":
            continue
        authored = parse_dt(r.get("authoredOn"))
        meds.append({"authored": authored})
    return meds


def extract_procedures(entries: list) -> list:
    """Extract Procedure resources."""
    procs = []
    for entry in entries:
        r = entry["resource"]
        if r["resourceType"] != "Procedure":
            continue
        performed = parse_dt(
            r.get("performedPeriod", {}).get("start") or r.get("performedDateTime")
        )
        enc_ref = r.get("encounter", {}).get("reference", "")
        procs.append({"performed": performed, "encounter_ref": enc_ref})
    return procs


# ---------------------------------------------------------------------------
# Feature engineering per encounter
# ---------------------------------------------------------------------------

def is_inpatient(enc: dict) -> bool:
    """True if encounter is inpatient or emergency."""
    return enc["enc_class"] in ("IMP", "EMER", "inpatient", "emergency")


def build_patient_rows(bundle_path: str) -> list:
    """
    Parse one FHIR bundle and return a list of feature rows,
    one per inpatient/emergency encounter (index encounter).
    """
    with open(bundle_path, "r") as f:
        bundle = json.load(f)

    entries = bundle.get("entry", [])
    if not entries:
        return []

    patient = extract_patient(entries)
    if not patient.get("patient_id"):
        return []

    all_encounters = extract_encounters(entries)
    conditions = extract_conditions(entries)
    medications = extract_medications(entries)
    procedures = extract_procedures(entries)

    # Only build rows for inpatient/ED encounters (index encounters)
    inpatient_encs = [e for e in all_encounters if is_inpatient(e)]

    rows = []
    for i, enc in enumerate(inpatient_encs):
        enc_start = enc["start"]
        enc_end = enc["end"]

        # --- Age at admission ---
        age = get_age(patient["birth_date"], enc_start)
        if age is None or age < 0:
            continue

        # --- Length of stay in days ---
        los_days = enc["los_hours"] / 24 if enc["los_hours"] else 0

        # --- Prior inpatient admissions (before this encounter) ---
        prior_admissions = sum(
            1 for e in inpatient_encs[:i]  # encounters before current index
        )

        # --- Conditions active at time of encounter ---
        active_conditions = [
            c for c in conditions
            if c["onset"] and c["onset"] <= enc_start
            and c["clinical_status"] not in ("resolved", "inactive")
        ]
        num_conditions = len(active_conditions)

        # --- Medications at time of encounter ---
        active_meds = [
            m for m in medications
            if m["authored"] and m["authored"] <= enc_start
        ]
        num_medications = len(active_meds)

        # --- Procedures during this encounter ---
        enc_id_ref = f"urn:uuid:{enc['encounter_id']}"
        enc_procs = [
            p for p in procedures
            if p["encounter_ref"] == enc_id_ref
        ]
        num_procedures = len(enc_procs)

        # --- Emergency visit in prior 6 months ---
        six_months_ago = enc_start - timedelta(days=180)
        prior_ed = any(
            e for e in all_encounters
            if e["enc_class"] == "EMER"
            and six_months_ago <= e["start"] < enc_start
        )

        # --- 30-day readmission label ---
        # Look for any inpatient/ED encounter starting within 30 days after discharge
        readmitted = False
        if enc_end:
            window_end = enc_end + timedelta(days=30)
            for future_enc in inpatient_encs[i + 1:]:
                if enc_end < future_enc["start"] <= window_end:
                    readmitted = True
                    break

        rows.append({
            "patient_id": patient["patient_id"],
            "encounter_id": enc["encounter_id"],
            "age": round(age, 1),
            "gender": patient["gender"],
            "length_of_stay_days": round(los_days, 2),
            "prior_admissions": prior_admissions,
            "num_conditions": num_conditions,
            "num_medications": num_medications,
            "num_procedures": num_procedures,
            "has_prior_ed_visit": int(prior_ed),
            "enc_class": enc["enc_class"],
            "readmitted_30d": int(readmitted),
        })

    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_fhir_directory(input_dir: str, output_path: str):
    """Parse all FHIR JSON files in a directory into a single CSV."""
    fhir_files = list(Path(input_dir).glob("*.json"))
    print(f"Found {len(fhir_files)} FHIR patient files in {input_dir}")

    all_rows = []
    errors = 0

    for i, fpath in enumerate(fhir_files):
        try:
            rows = build_patient_rows(str(fpath))
            all_rows.extend(rows)
        except Exception as e:
            errors += 1
            if errors <= 5:  # only print first 5 errors
                print(f"  [WARN] Failed to parse {fpath.name}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(fhir_files)} files...")

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No inpatient encounters found. Check your FHIR files.")
        return

    print(f"\nDone. {len(df)} inpatient encounter rows from {len(fhir_files) - errors} patients.")
    print(f"Errors/skipped: {errors}")
    print(f"\nClass distribution:")
    print(df["readmitted_30d"].value_counts())
    print(f"Readmission rate: {df['readmitted_30d'].mean():.1%}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return df


# ---------------------------------------------------------------------------
# Quick test on a single file
# ---------------------------------------------------------------------------

def test_single_file(fhir_path: str):
    """Run parser on a single file and print results."""
    print(f"Parsing: {fhir_path}\n")
    rows = build_patient_rows(fhir_path)

    if not rows:
        print("No inpatient encounters found for this patient.")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nEncounters found: {len(rows)}")
    print(f"Readmitted: {df['readmitted_30d'].sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Synthea FHIR bundles into ML features")
    parser.add_argument("--input", default="data/raw/fhir/", help="Directory containing FHIR JSON files")
    parser.add_argument("--output", default="data/processed/features.csv", help="Output CSV path")
    parser.add_argument("--test", help="Test on a single FHIR JSON file")
    args = parser.parse_args()

    if args.test:
        import pandas as pd
        test_single_file(args.test)
    else:
        import pandas as pd
        parse_fhir_directory(args.input, args.output)