"""
RuralHealthNet-v0 — Main Environment
Implements the full OpenEnv API: reset() / step() / state()
"""

from __future__ import annotations
import random
import copy
from typing import Dict, List, Optional

from env.models import (
    Action, Observation, StepResult, HealthNetState,
    ClinicResources, Patient, MedicineType, Severity,
    TransportMode, PatientAssignment,
)
from env.patient_simulator import generate_patients, age_patients


# ─── Reward Constants ─────────────────────────────────────────────────────────

REWARD_TREAT_MILD      =  0.20
REWARD_TREAT_MODERATE  =  0.45
REWARD_TREAT_CRITICAL  =  1.00
REWARD_DETERIORATION   = -0.30
REWARD_PATIENT_LOST    = -1.00
REWARD_WRONG_MEDICINE  = -0.15
REWARD_OVERLOAD_CLINIC = -0.10
REWARD_AMBULANCE_SAVED =  0.10   # bonus for correct ambulance use on critical
REWARD_RESOURCE_WASTE  = -0.05   # penalty for assigning ambulance to mild cases

TREAT_HOURS_LIMIT = {
    Severity.CRITICAL: 2.0,
    Severity.MODERATE: 6.0,
    Severity.MILD:     24.0,
}


class RuralHealthNetEnv:
    """
    OpenEnv-compatible environment simulating rural telemedicine triage.

    API:
        env.reset()       -> Observation
        env.step(action)  -> StepResult
        env.state()       -> HealthNetState

    Args:
        task_config (dict): Configuration dict from tasks/easy|medium|hard.py
        seed (int): Random seed for reproducibility
    """

    def __init__(self, task_config: dict, seed: int = 42):
        self.cfg   = task_config
        self.seed  = seed
        self._rng  = random.Random(seed)
        self._state: Optional[HealthNetState] = None

    # ─── OpenEnv API ─────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns first observation."""
        self._rng = random.Random(self.seed)

        clinics = self._build_clinics()
        clinic_ids = [c.clinic_id for c in clinics]

        initial_patients = generate_patients(
            n=self.cfg["initial_patients"],
            clinic_ids=clinic_ids,
            rng=self._rng,
        )

        self._state = HealthNetState(
            step_number=0,
            time_hours=0.0,
            patients_waiting=initial_patients,
            clinics=clinics,
            patients_treated=0,
            patients_deteriorated=0,
            patients_lost=0,
            total_patients_seen=len(initial_patients),
            epidemic_active=False,
            cumulative_reward=0.0,
        )

        return self._to_observation()

    def step(self, action: Action) -> StepResult:
        """
        Apply agent action, advance simulation by one time step.
        Returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        s = self._state
        step_reward = 0.0
        info: Dict = {
            "treated_this_step": 0,
            "deteriorated_this_step": 0,
            "lost_this_step": 0,
            "invalid_actions": 0,
        }

        # 1. Process agent assignments
        assigned_ids = set()
        patient_map  = {p.patient_id: p for p in s.patients_waiting}
        clinic_map   = {c.clinic_id: c for c in s.clinics}

        for assignment in action.assignments:
            pid = assignment.patient_id
            if pid not in patient_map or pid in assigned_ids:
                info["invalid_actions"] += 1
                continue

            patient = patient_map[pid]
            clinic  = clinic_map.get(assignment.clinic_id)

            if clinic is None or not clinic.is_online:
                info["invalid_actions"] += 1
                continue

            # Check clinic capacity
            if clinic.current_load >= clinic.capacity:
                step_reward += REWARD_OVERLOAD_CLINIC
                info["invalid_actions"] += 1
                continue

            # Check doctor availability
            if clinic.doctors_available <= 0:
                info["invalid_actions"] += 1
                continue

            # Validate transport
            transport_ok = True
            if assignment.transport == TransportMode.AMBULANCE:
                if clinic.ambulances_available <= 0:
                    transport_ok = False
                else:
                    clinic.ambulances_available -= 1
                    if patient.severity == Severity.CRITICAL:
                        step_reward += REWARD_AMBULANCE_SAVED
                    elif patient.severity == Severity.MILD:
                        step_reward += REWARD_RESOURCE_WASTE

            if not transport_ok:
                info["invalid_actions"] += 1
                continue

            # Medicine validation
            prescribed = assignment.medicine
            needed     = patient.medicine_needed
            if needed != MedicineType.NONE and prescribed == needed:
                med_stock = clinic.medicine_stock.get(needed, 0)
                if med_stock > 0:
                    clinic.medicine_stock[needed] = med_stock - 1
                else:
                    step_reward += REWARD_WRONG_MEDICINE
            elif needed != MedicineType.NONE and prescribed != needed and prescribed != MedicineType.NONE:
                step_reward += REWARD_WRONG_MEDICINE

            # Treat patient
            clinic.doctors_available -= 1
            clinic.current_load      += 1

            sev = patient.severity
            if sev == Severity.CRITICAL:
                step_reward += REWARD_TREAT_CRITICAL
            elif sev == Severity.MODERATE:
                step_reward += REWARD_TREAT_MODERATE
            else:
                step_reward += REWARD_TREAT_MILD

            # Time-sensitive bonus: treat within recommended window
            limit = TREAT_HOURS_LIMIT[sev]
            if patient.wait_hours <= limit:
                step_reward += 0.05 * (1 - patient.wait_hours / limit)

            patient_map[pid] = patient.model_copy(update={"is_treated": True})
            assigned_ids.add(pid)
            s.patients_treated += 1
            info["treated_this_step"] += 1

        # 2. Age remaining (untreated) patients — deterioration & losses
        untreated = [p for pid, p in patient_map.items() if not p.is_treated]
        aged      = age_patients(untreated, self._rng)
        lost_ids  = set()

        for orig, aged_p in zip(untreated, aged):
            if aged_p.severity != orig.severity:
                s.patients_deteriorated += 1
                info["deteriorated_this_step"] += 1
                step_reward += REWARD_DETERIORATION

            # Patient lost if critical and waited past 2x the critical time limit
            if (aged_p.severity == Severity.CRITICAL
                    and aged_p.wait_hours > TREAT_HOURS_LIMIT[Severity.CRITICAL] * 2
                    and aged_p.patient_id not in lost_ids):
                if self._rng.random() < 0.40:
                    s.patients_lost += 1
                    info["lost_this_step"] += 1
                    step_reward += REWARD_PATIENT_LOST
                    lost_ids.add(aged_p.patient_id)

        # 3. Restore doctor capacity partially (doctors rotate)
        for clinic in s.clinics:
            if clinic.is_online:
                clinic.doctors_available = min(
                    clinic.doctors_available + self.cfg["doctors_per_step_restore"],
                    self.cfg.get("max_doctors_per_clinic", 3),
                )
                clinic.current_load = max(0, clinic.current_load - 1)
                # Restore ambulances
                clinic.ambulances_available = min(
                    clinic.ambulances_available + 1,
                    self.cfg.get("max_ambulances_per_clinic", 2),
                )

        # 4. Advance time and arrive new patients
        s.step_number += 1
        s.time_hours  = round(s.time_hours + self.cfg["hours_per_step"], 2)

        # Epidemic trigger
        if not s.epidemic_active and self.cfg.get("epidemic_step") and \
                s.step_number == self.cfg["epidemic_step"]:
            s.epidemic_active = True

        # Clinic offline events
        for event in self.cfg.get("clinic_offline_events", []):
            if s.step_number == event["step"]:
                for c in s.clinics:
                    if c.clinic_id == event["clinic_id"]:
                        c.is_online = event["online"]

        # New arrivals
        clinic_ids = [c.clinic_id for c in s.clinics if c.is_online]
        new_patients = generate_patients(
            n=self._rng.randint(*self.cfg["arrivals_per_step_range"]),
            clinic_ids=clinic_ids,
            rng=self._rng,
            epidemic_active=s.epidemic_active,
            epidemic_condition=self.cfg.get("epidemic_condition", "High Fever"),
        )
        s.total_patients_seen += len(new_patients)

        # Combine: treated patients removed, aged untreated + new arrivals
        remaining = [p for p in aged if not p.is_treated and p.patient_id not in lost_ids]
        s.patients_waiting = remaining + new_patients

        s.cumulative_reward += step_reward

        # 5. Check termination
        done = s.step_number >= self.cfg["max_steps"]

        info.update({
            "step": s.step_number,
            "cumulative_reward": round(s.cumulative_reward, 4),
            "total_patients_seen": s.total_patients_seen,
            "patients_treated": s.patients_treated,
            "patients_lost": s.patients_lost,
            "epidemic_active": s.epidemic_active,
        })

        return StepResult(
            observation=self._to_observation(),
            reward=round(step_reward, 4),
            done=done,
            info=info,
        )

    def state(self) -> HealthNetState:
        """Return full internal state (for debugging, graders, logging)."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return copy.deepcopy(self._state)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _to_observation(self) -> Observation:
        s = self._state
        return Observation(
            step_number=s.step_number,
            time_hours=s.time_hours,
            patients_waiting=copy.deepcopy(s.patients_waiting),
            clinics=copy.deepcopy(s.clinics),
            epidemic_active=s.epidemic_active,
            info_hints={
                "max_steps": self.cfg["max_steps"],
                "patients_treated": s.patients_treated,
                "patients_lost": s.patients_lost,
            },
        )

    def _build_clinics(self) -> List[ClinicResources]:
        clinics = []
        for cfg_c in self.cfg["clinics"]:
            stock = {MedicineType(k): v for k, v in cfg_c.get("medicine_stock", {}).items()}
            clinics.append(
                ClinicResources(
                    clinic_id=cfg_c["id"],
                    name=cfg_c["name"],
                    doctors_available=cfg_c["doctors"],
                    ambulances_available=cfg_c["ambulances"],
                    capacity=cfg_c["capacity"],
                    current_load=0,
                    medicine_stock=stock,
                    is_online=True,
                )
            )
        return clinics
