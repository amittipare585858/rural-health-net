"""
Task: EASY — Single Clinic Triage
- 1 clinic, 2 doctors, 1 ambulance
- ~5 patients per step
- 30 steps (~30 simulation hours)
- No epidemic, no clinic failures
- Goal: treat as many patients as possible
"""

EASY_CONFIG = {
    "task_id": "easy",
    "name": "Single Clinic Triage",
    "description": (
        "Manage triage at a single rural clinic. "
        "Assign incoming patients to doctors with appropriate transport and medicine."
    ),
    "max_steps": 30,
    "hours_per_step": 1.0,
    "initial_patients": 5,
    "arrivals_per_step_range": [2, 5],
    "doctors_per_step_restore": 1,
    "max_doctors_per_clinic": 2,
    "max_ambulances_per_clinic": 1,
    "epidemic_step": None,
    "epidemic_condition": None,
    "clinic_offline_events": [],
    "clinics": [
        {
            "id": 0,
            "name": "Dhule Primary Health Centre",
            "doctors": 2,
            "ambulances": 1,
            "capacity": 5,
            "medicine_stock": {
                "antibiotics": 20,
                "painkillers": 20,
                "cardiac": 10,
                "antidiabetic": 10,
                "none": 999,
            },
        }
    ],
}
