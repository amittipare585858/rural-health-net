"""
RuralHealthNet-v0 — Heuristic Baseline Agent

Strategy:
  1. Sort patients by severity (critical first) then wait time
  2. Assign critical patients to clinics with doctors + ambulances available
  3. Prescribe the correct medicine if stock exists
  4. Use telemedicine for mild cases to save doctor slots
  5. Defer if no resources available
"""

from __future__ import annotations
from typing import List

from env.models import (
    Observation, Action, PatientAssignment,
    Patient, Severity, TransportMode, MedicineType,
    ClinicResources,
)


SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.MODERATE: 1, Severity.MILD: 2}


def _pick_best_clinic(patient: Patient, clinics: List[ClinicResources]) -> ClinicResources | None:
    """
    Choose the best available clinic for a patient.
    Prefer: patient's nearest clinic → least loaded online clinic.
    """
    online = [c for c in clinics if c.is_online and c.doctors_available > 0
              and c.current_load < c.capacity]
    if not online:
        return None

    # Prefer nearest clinic if available
    nearest = next((c for c in online if c.clinic_id == patient.location_clinic), None)
    if nearest:
        return nearest

    # Otherwise pick least loaded
    return min(online, key=lambda c: c.current_load / c.capacity)


def _choose_transport(patient: Patient, clinic: ClinicResources) -> TransportMode:
    """Use ambulance only for critical cases; telemedicine for mild."""
    if patient.severity == Severity.CRITICAL and clinic.ambulances_available > 0:
        return TransportMode.AMBULANCE
    if patient.severity == Severity.MILD:
        return TransportMode.TELEMEDICINE
    return TransportMode.SELF_TRANSPORT


def _choose_medicine(patient: Patient, clinic: ClinicResources) -> MedicineType:
    needed = patient.medicine_needed
    if needed == MedicineType.NONE:
        return MedicineType.NONE
    stock = clinic.medicine_stock.get(needed, 0)
    return needed if stock > 0 else MedicineType.NONE


class HeuristicAgent:
    """
    Rule-based heuristic agent for RuralHealthNet-v0.
    Used as a reproducible baseline.
    """

    def act(self, obs: Observation) -> Action:
        assignments: List[PatientAssignment] = []

        # Track resource state locally (to avoid double-assigning within a step)
        clinic_state = {
            c.clinic_id: {
                "doctors": c.doctors_available,
                "ambulances": c.ambulances_available,
                "load": c.current_load,
                "capacity": c.capacity,
                "is_online": c.is_online,
                "medicine": dict(c.medicine_stock),
            }
            for c in obs.clinics
        }

        # Sort patients: critical first, then by wait time descending
        sorted_patients = sorted(
            obs.patients_waiting,
            key=lambda p: (SEVERITY_ORDER[p.severity], -p.wait_hours),
        )

        for patient in sorted_patients:
            # Find best clinic using local state
            best_clinic_id = None
            for c in obs.clinics:
                cs = clinic_state[c.clinic_id]
                if (cs["is_online"]
                        and cs["doctors"] > 0
                        and cs["load"] < cs["capacity"]):
                    if best_clinic_id is None:
                        best_clinic_id = c.clinic_id
                    elif c.clinic_id == patient.location_clinic:
                        best_clinic_id = c.clinic_id
                        break

            if best_clinic_id is None:
                continue  # no resources; skip this patient

            cs = clinic_state[best_clinic_id]

            # Choose transport
            if patient.severity == Severity.CRITICAL and cs["ambulances"] > 0:
                transport = TransportMode.AMBULANCE
                cs["ambulances"] -= 1
            elif patient.severity == Severity.MILD:
                transport = TransportMode.TELEMEDICINE
            else:
                transport = TransportMode.SELF_TRANSPORT

            # Choose medicine
            needed = patient.medicine_needed
            if needed != MedicineType.NONE and cs["medicine"].get(needed, 0) > 0:
                medicine = needed
                cs["medicine"][needed] -= 1
            else:
                medicine = MedicineType.NONE

            # Update local state
            cs["doctors"] -= 1
            cs["load"]    += 1

            assignments.append(
                PatientAssignment(
                    patient_id=patient.patient_id,
                    clinic_id=best_clinic_id,
                    transport=transport,
                    medicine=medicine,
                    priority=SEVERITY_ORDER[patient.severity] + 1,
                )
            )

        return Action(assignments=assignments)
