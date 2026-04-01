"""
Task: MEDIUM — Multi-Clinic Resource Management
- 3 clinics, varying resources
- ~10 patients per step, spread across clinics
- 60 steps
- No epidemic, but medicine stock is limited
- Goal: balance load across clinics, conserve resources
"""

MEDIUM_CONFIG = {
    "task_id": "medium",
    "name": "Multi-Clinic Resource Management",
    "description": (
        "Coordinate triage across 3 rural clinics with unequal resources. "
        "Balance patient load, route ambulances efficiently, and conserve medicine."
    ),
    "max_steps": 60,
    "hours_per_step": 1.0,
    "initial_patients": 10,
    "arrivals_per_step_range": [5, 12],
    "doctors_per_step_restore": 1,
    "max_doctors_per_clinic": 3,
    "max_ambulances_per_clinic": 2,
    "epidemic_step": None,
    "epidemic_condition": None,
    "clinic_offline_events": [],
    "clinics": [
        {
            "id": 0,
            "name": "Nashik District Hospital",
            "doctors": 3,
            "ambulances": 2,
            "capacity": 8,
            "medicine_stock": {
                "antibiotics": 25,
                "painkillers": 25,
                "cardiac": 15,
                "antidiabetic": 15,
                "none": 999,
            },
        },
        {
            "id": 1,
            "name": "Nandurbar PHC",
            "doctors": 2,
            "ambulances": 1,
            "capacity": 5,
            "medicine_stock": {
                "antibiotics": 15,
                "painkillers": 10,
                "cardiac": 8,
                "antidiabetic": 8,
                "none": 999,
            },
        },
        {
            "id": 2,
            "name": "Dhule Sub-District Centre",
            "doctors": 1,
            "ambulances": 1,
            "capacity": 4,
            "medicine_stock": {
                "antibiotics": 10,
                "painkillers": 10,
                "cardiac": 5,
                "antidiabetic": 5,
                "none": 999,
            },
        },
    ],
}
