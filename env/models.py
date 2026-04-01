"""
RuralHealthNet-v0 — Typed Models (OpenEnv Spec)
All state, action, observation, and result types defined with Pydantic.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum


# ─── Enumerations ────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "critical"    # life-threatening, must treat < 2 hours
    MODERATE = "moderate"    # serious, treat < 6 hours
    MILD     = "mild"        # non-urgent, treat < 24 hours


class TransportMode(str, Enum):
    AMBULANCE      = "ambulance"       # physical transport via ambulance
    TELEMEDICINE   = "telemedicine"    # remote video consult
    SELF_TRANSPORT = "self_transport"  # patient walks/drives
    DEFER          = "defer"           # defer to next cycle (risk of deterioration)


class MedicineType(str, Enum):
    ANTIBIOTICS   = "antibiotics"
    PAINKILLERS   = "painkillers"
    CARDIAC       = "cardiac"
    ANTIDIABETIC  = "antidiabetic"
    NONE          = "none"


# ─── Core Domain Objects ──────────────────────────────────────────────────────

class Patient(BaseModel):
    patient_id:       str
    severity:         Severity
    condition:        str
    wait_hours:       float = Field(ge=0.0, description="Hours patient has waited")
    location_clinic:  int   = Field(description="Nearest clinic ID")
    medicine_needed:  MedicineType = MedicineType.NONE
    deterioration_rate: float = Field(ge=0.0, le=1.0,
        description="Probability of worsening per step if untreated")
    is_treated:       bool  = False


class ClinicResources(BaseModel):
    clinic_id:           int
    name:                str
    doctors_available:   int   = Field(ge=0)
    ambulances_available:int   = Field(ge=0)
    capacity:            int   = Field(ge=1, description="Max simultaneous patients")
    current_load:        int   = Field(ge=0)
    medicine_stock:      Dict[MedicineType, int] = Field(default_factory=dict)
    is_online:           bool  = True


# ─── State / Observation ──────────────────────────────────────────────────────

class HealthNetState(BaseModel):
    """Full internal state of the environment."""
    step_number:          int
    time_hours:           float
    patients_waiting:     List[Patient]
    clinics:              List[ClinicResources]
    patients_treated:     int   = 0
    patients_deteriorated:int   = 0
    patients_lost:        int   = 0
    total_patients_seen:  int   = 0
    epidemic_active:      bool  = False
    cumulative_reward:    float = 0.0


class Observation(BaseModel):
    """Partial observation returned to the agent each step."""
    step_number:      int
    time_hours:       float
    patients_waiting: List[Patient]
    clinics:          List[ClinicResources]
    epidemic_active:  bool  = False
    info_hints:       Dict  = Field(default_factory=dict)


# ─── Action ───────────────────────────────────────────────────────────────────

class PatientAssignment(BaseModel):
    patient_id:  str
    clinic_id:   int
    transport:   TransportMode
    medicine:    MedicineType = MedicineType.NONE
    priority:    int          = Field(ge=1, le=5, description="1=highest urgency")


class Action(BaseModel):
    """Agent action: list of patient assignments for this step."""
    assignments: List[PatientAssignment] = Field(default_factory=list)


# ─── Step Result ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        Dict = Field(default_factory=dict)


# ─── Grader Result ────────────────────────────────────────────────────────────

class GraderResult(BaseModel):
    task_id:             str
    score:               float = Field(ge=0.0, le=1.0)
    treatment_rate:      float
    deterioration_rate:  float
    resource_efficiency: float
    breakdown:           Dict  = Field(default_factory=dict)
    passed:              bool
