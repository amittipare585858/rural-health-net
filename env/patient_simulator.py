"""
Patient Simulator — generates realistic patient arrivals for RuralHealthNet-v0.
Uses condition-specific severity distributions and deterioration rates.
"""

from __future__ import annotations
import random
import uuid
from typing import List
from env.models import Patient, Severity, MedicineType

# ─── Condition Library ────────────────────────────────────────────────────────

CONDITION_PROFILES = [
    {
        "condition": "Chest Pain",
        "severity_weights": {"critical": 0.5, "moderate": 0.4, "mild": 0.1},
        "medicine": MedicineType.CARDIAC,
        "deterioration_base": 0.35,
    },
    {
        "condition": "High Fever",
        "severity_weights": {"critical": 0.1, "moderate": 0.5, "mild": 0.4},
        "medicine": MedicineType.ANTIBIOTICS,
        "deterioration_base": 0.15,
    },
    {
        "condition": "Severe Injury",
        "severity_weights": {"critical": 0.6, "moderate": 0.35, "mild": 0.05},
        "medicine": MedicineType.PAINKILLERS,
        "deterioration_base": 0.40,
    },
    {
        "condition": "Diabetic Crisis",
        "severity_weights": {"critical": 0.4, "moderate": 0.45, "mild": 0.15},
        "medicine": MedicineType.ANTIDIABETIC,
        "deterioration_base": 0.30,
    },
    {
        "condition": "Respiratory Distress",
        "severity_weights": {"critical": 0.45, "moderate": 0.40, "mild": 0.15},
        "medicine": MedicineType.ANTIBIOTICS,
        "deterioration_base": 0.25,
    },
    {
        "condition": "Minor Laceration",
        "severity_weights": {"critical": 0.0, "moderate": 0.2, "mild": 0.8},
        "medicine": MedicineType.NONE,
        "deterioration_base": 0.02,
    },
    {
        "condition": "Acute Abdominal Pain",
        "severity_weights": {"critical": 0.3, "moderate": 0.5, "mild": 0.2},
        "medicine": MedicineType.PAINKILLERS,
        "deterioration_base": 0.20,
    },
    {
        "condition": "Hypertensive Crisis",
        "severity_weights": {"critical": 0.4, "moderate": 0.45, "mild": 0.15},
        "medicine": MedicineType.CARDIAC,
        "deterioration_base": 0.28,
    },
]

SEVERITY_DETERIORATION_MULTIPLIER = {
    Severity.CRITICAL: 1.5,
    Severity.MODERATE: 1.0,
    Severity.MILD:     0.4,
}


def _weighted_severity(weights: dict) -> Severity:
    r = random.random()
    cumulative = 0.0
    for sev_str, prob in weights.items():
        cumulative += prob
        if r < cumulative:
            return Severity(sev_str)
    return Severity.MILD


def generate_patients(
    n: int,
    clinic_ids: List[int],
    rng: random.Random,
    epidemic_active: bool = False,
    epidemic_condition: str = "High Fever",
) -> List[Patient]:
    """
    Generate `n` new patients distributed across given clinic locations.
    If epidemic_active, majority of patients share the epidemic condition.
    """
    patients: List[Patient] = []
    for _ in range(n):
        if epidemic_active and rng.random() < 0.65:
            # During epidemic, skew toward the outbreak condition
            profile = next(
                p for p in CONDITION_PROFILES if p["condition"] == epidemic_condition
            )
        else:
            profile = rng.choice(CONDITION_PROFILES)

        severity = _weighted_severity(profile["severity_weights"])
        det_rate = (
            profile["deterioration_base"]
            * SEVERITY_DETERIORATION_MULTIPLIER[severity]
        )
        det_rate = max(0.0, min(1.0, det_rate + rng.uniform(-0.05, 0.05)))

        patients.append(
            Patient(
                patient_id=str(uuid.uuid4())[:8],
                severity=severity,
                condition=profile["condition"],
                wait_hours=round(rng.uniform(0.0, 2.0), 2),
                location_clinic=rng.choice(clinic_ids),
                medicine_needed=profile["medicine"],
                deterioration_rate=round(det_rate, 3),
                is_treated=False,
            )
        )
    return patients


def age_patients(patients: List[Patient], rng: random.Random) -> List[Patient]:
    """
    Simulate the passage of one time step: increase wait times,
    potentially worsen severity for untreated patients.
    Returns updated patient list (some may have deteriorated).
    """
    updated: List[Patient] = []
    for p in patients:
        if p.is_treated:
            updated.append(p)
            continue

        new_wait = round(p.wait_hours + 1.0, 2)

        # Severity deterioration
        new_severity = p.severity
        if p.severity == Severity.MILD and rng.random() < p.deterioration_rate:
            new_severity = Severity.MODERATE
        elif p.severity == Severity.MODERATE and rng.random() < p.deterioration_rate:
            new_severity = Severity.CRITICAL

        updated.append(p.model_copy(update={"wait_hours": new_wait, "severity": new_severity}))
    return updated
