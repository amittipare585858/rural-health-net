"""
Task: HARD — Epidemic Surge Response
- 5 clinics across a region
- ~15–25 patients per step; epidemic surge at step 40
- One clinic goes offline at step 55 (simulates road/power failure)
- 100 steps total
- Limited medicine, ambulances, doctors
- Goal: treat critical patients, contain epidemic spread, survive the surge
"""

HARD_CONFIG = {
    "task_id": "hard",
    "name": "Epidemic Surge Response",
    "description": (
        "Manage a regional health network across 5 clinics during a disease outbreak. "
        "A High Fever epidemic strikes at step 40. One clinic loses power at step 55. "
        "Prioritize critical cases, ration scarce medicines, and prevent patient loss."
    ),
    "max_steps": 100,
    "hours_per_step": 1.0,
    "initial_patients": 15,
    "arrivals_per_step_range": [8, 20],
    "doctors_per_step_restore": 1,
    "max_doctors_per_clinic": 4,
    "max_ambulances_per_clinic": 3,

    # Epidemic configuration
    "epidemic_step": 40,
    "epidemic_condition": "High Fever",

    # Clinic failure events
    "clinic_offline_events": [
        {"step": 55, "clinic_id": 3, "online": False},   # clinic 3 goes offline
        {"step": 75, "clinic_id": 3, "online": True},    # clinic 3 comes back
    ],

    "clinics": [
        {
            "id": 0,
            "name": "Nashik District Hospital",
            "doctors": 4,
            "ambulances": 3,
            "capacity": 12,
            "medicine_stock": {
                "antibiotics": 40,
                "painkillers": 35,
                "cardiac": 20,
                "antidiabetic": 20,
                "none": 999,
            },
        },
        {
            "id": 1,
            "name": "Nandurbar PHC",
            "doctors": 2,
            "ambulances": 2,
            "capacity": 7,
            "medicine_stock": {
                "antibiotics": 20,
                "painkillers": 15,
                "cardiac": 10,
                "antidiabetic": 10,
                "none": 999,
            },
        },
        {
            "id": 2,
            "name": "Dhule Sub-District Centre",
            "doctors": 2,
            "ambulances": 1,
            "capacity": 6,
            "medicine_stock": {
                "antibiotics": 18,
                "painkillers": 12,
                "cardiac": 8,
                "antidiabetic": 8,
                "none": 999,
            },
        },
        {
            "id": 3,
            "name": "Jalgaon Rural Clinic",  # goes offline step 55–75
            "doctors": 2,
            "ambulances": 1,
            "capacity": 5,
            "medicine_stock": {
                "antibiotics": 15,
                "painkillers": 12,
                "cardiac": 6,
                "antidiabetic": 6,
                "none": 999,
            },
        },
        {
            "id": 4,
            "name": "Malegaon Community Health Post",
            "doctors": 1,
            "ambulances": 1,
            "capacity": 4,
            "medicine_stock": {
                "antibiotics": 12,
                "painkillers": 10,
                "cardiac": 5,
                "antidiabetic": 5,
                "none": 999,
            },
        },
    ],
}
