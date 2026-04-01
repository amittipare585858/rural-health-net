---
title: RuralHealthNet-v0
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - healthcare
  - reinforcement-learning
  - triage
  - telemedicine
license: mit
---

# 🏥 RuralHealthNet-v0

> **OpenEnv environment for rural telemedicine triage and resource allocation**  
> An AI agent coordinates doctors, medicines, and ambulances across rural health clinics —
> treating patients before they deteriorate, rationing scarce supplies, and surviving epidemic surges.

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://openenv.ai)
[![HF Spaces](https://img.shields.io/badge/🤗-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/amittipare585858/rural-health-net)
[![Python](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

---

## 🌍 Real-World Motivation

India has **6.4 lakh villages**, most of which rely on Primary Health Centres (PHCs) with
1–2 doctors, limited medicines, and a single ambulance. District health officers must
constantly decide: *which patient goes to which clinic*, *when to dispatch the ambulance*,
*which medicine to prescribe given scarce stock*, and *how to handle a sudden epidemic*.

**RuralHealthNet-v0** converts this coordination challenge into a structured
reinforcement learning environment where AI agents can learn effective triage policies.

---

## 📦 Environment Overview

| Property | Value |
|---|---|
| **Type** | Real-world simulation (not a game) |
| **API** | OpenEnv spec: `reset()` / `step()` / `state()` |
| **Tasks** | 3 (easy → medium → hard) |
| **Reward** | Partial progress signals per step |
| **Observation** | Typed Pydantic models |
| **Deployment** | Hugging Face Spaces + Docker |

---

## 🗂 Project Structure

```
rural-health-net/
├── env/
│   ├── __init__.py
│   ├── models.py            # All typed Pydantic models
│   ├── health_env.py        # Main OpenEnv environment (step/reset/state)
│   ├── patient_simulator.py # Realistic patient arrival generator
│   └── graders.py           # Task graders (scores 0.0–1.0)
├── tasks/
│   ├── __init__.py
│   ├── easy.py              # Task 1: Single Clinic Triage
│   ├── medium.py            # Task 2: Multi-Clinic Resource Management
│   └── hard.py              # Task 3: Epidemic Surge Response
├── agents/
│   ├── baseline_agent.py    # Heuristic baseline agent
│   └── run_baseline.py      # Reproducible scoring script
├── openenv.yaml             # OpenEnv specification
├── app.py                   # FastAPI server (Hugging Face Spaces)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔧 Setup

### Local Installation

```bash
git clone https://github.com/amittipare585858/rural-health-net
cd rural-health-net
pip install -r requirements.txt
```

### Run Baseline (Reproducible Scores)

```bash
# All tasks with seed=42
python agents/run_baseline.py

# Specific task
python agents/run_baseline.py --task easy --seed 42

# JSON output
python agents/run_baseline.py --json
```

### Start API Server

```bash
python app.py
# → http://localhost:7860
# → Swagger docs: http://localhost:7860/docs
```

### Docker

```bash
docker build -t rural-health-net .
docker run -p 7860:7860 rural-health-net
```

---

## 🧠 OpenEnv API

### `reset(task_id, seed) → Observation`

Resets the environment and returns the initial observation.

```python
from env import RuralHealthNetEnv
from tasks import TASK_CONFIGS

env = RuralHealthNetEnv(task_config=TASK_CONFIGS["easy"], seed=42)
obs = env.reset()
```

### `step(action) → StepResult`

Applies the agent's action and advances one simulation hour.

```python
from env.models import Action, PatientAssignment, TransportMode, MedicineType

action = Action(assignments=[
    PatientAssignment(
        patient_id="a1b2c3d4",
        clinic_id=0,
        transport=TransportMode.AMBULANCE,
        medicine=MedicineType.CARDIAC,
        priority=1,
    )
])
result = env.step(action)
print(result.reward, result.done)
```

### `state() → HealthNetState`

Returns the full internal environment state (useful for graders and debugging).

```python
state = env.state()
print(state.patients_treated, state.patients_lost)
```

---

## 👁 Observation Space

```
Observation
├── step_number       int          Current simulation step
├── time_hours        float        Elapsed simulation hours
├── epidemic_active   bool         Whether outbreak is ongoing
├── patients_waiting  List[Patient]
│   ├── patient_id        str
│   ├── severity          "critical" | "moderate" | "mild"
│   ├── condition         str (e.g. "Chest Pain", "High Fever")
│   ├── wait_hours        float     Hours waiting untreated
│   ├── location_clinic   int       Nearest clinic ID
│   ├── medicine_needed   MedicineType
│   └── deterioration_rate float   Probability of worsening per step
└── clinics           List[ClinicResources]
    ├── clinic_id             int
    ├── name                  str
    ├── doctors_available     int
    ├── ambulances_available  int
    ├── capacity              int
    ├── current_load          int
    ├── medicine_stock        Dict[MedicineType, int]
    └── is_online             bool
```

---

## ⚡ Action Space

```
Action
└── assignments  List[PatientAssignment]
    ├── patient_id   str           Must match patient in observation
    ├── clinic_id    int           Target clinic
    ├── transport    "ambulance" | "telemedicine" | "self_transport" | "defer"
    ├── medicine     "antibiotics" | "painkillers" | "cardiac" | "antidiabetic" | "none"
    └── priority     int (1=highest urgency)
```

---

## 💰 Reward Function

The reward is computed **per step** to provide dense, partial progress signals:

| Event | Reward |
|---|---|
| Treat critical patient | **+1.00** |
| Treat moderate patient | **+0.45** |
| Treat mild patient | **+0.20** |
| Timely treatment bonus | **+0.05** (scaled by urgency) |
| Correct ambulance use (critical) | **+0.10** |
| Patient deterioration | **−0.30** |
| Patient lost (untreated critical) | **−1.00** |
| Wrong medicine prescribed | **−0.15** |
| Clinic overload | **−0.10** |
| Ambulance wasted on mild case | **−0.05** |

---

## 🎯 Tasks

### Task 1: Easy — Single Clinic Triage
> `task_id="easy"` | 30 steps | 1 clinic | Pass: score ≥ 0.50

A single Primary Health Centre. Steady patient flow, 2–5 arrivals/step.  
Objective: Assign patients to the available doctor with correct medicine and transport.

### Task 2: Medium — Multi-Clinic Resource Management
> `task_id="medium"` | 60 steps | 3 clinics | Pass: score ≥ 0.55

Three clinics with unequal resources. ~10 patients/step spread across locations.  
Objective: Load-balance patients across clinics, conserve medicine stock, route ambulances efficiently.

### Task 3: Hard — Epidemic Surge Response
> `task_id="hard"` | 100 steps | 5 clinics | Pass: score ≥ 0.60

Five clinics with limited supplies. An epidemic strikes at **step 40**, flooding the system with fever cases.  
At **step 55**, one clinic loses power and goes offline. At **step 75** it comes back online.  
Objective: Triage under surge pressure, contain the epidemic, avoid patient loss.

---

## 🤖 Agent Graders

Each task has a dedicated grader returning a score from **0.0 to 1.0**:

| Grader | Weights |
|---|---|
| `grade_easy` | 70% treatment rate · −20% loss penalty · −10% deterioration |
| `grade_medium` | 55% treatment rate · +20% resource efficiency · −15% loss · −10% deterioration |
| `grade_hard` | 50% treatment rate · +20% resource efficiency · −20% loss · −10% deterioration · +15% epidemic bonus |

```python
from env.graders import grade
result = grade("easy", env.state())
print(result.score, result.passed, result.breakdown)
```

---

## 📊 Baseline Scores (seed=42)

| Task | Algorithm | Score | Treatment Rate | Status |
|---|---|---|---|---|
| easy | Heuristic (severity sort) | 0.1056 | 29.0% | ❌ floor |
| medium | Heuristic (severity sort) | 0.2628 | 34.4% | ❌ floor |
| hard | Heuristic (severity sort) | 0.0971 | 34.4% | ❌ floor |

> **These scores represent the lower bound.** A RL agent or better heuristic can significantly
> improve by: learning optimal load balancing, deferring mild patients during surges, and
> pre-positioning ambulances near critical patients.

---

## 🌐 HTTP API (Hugging Face Spaces)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Environment info |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset` | Reset environment |
| `POST` | `/step` | Apply action |
| `GET` | `/state/{session_id}` | Get full state |
| `POST` | `/grade` | Grade episode |

**Example:**
```bash
# Reset
curl -X POST https://huggingface.co/spaces/amittipare585858/rural-health-net/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42, "session_id": "demo"}'
```

---

## 🔬 Writing Your Own Agent

```python
import sys; sys.path.insert(0, ".")
from env import RuralHealthNetEnv
from env.models import Action, PatientAssignment, TransportMode, MedicineType
from env.graders import grade
from tasks import TASK_CONFIGS

env = RuralHealthNetEnv(TASK_CONFIGS["medium"], seed=42)
obs = env.reset()

while True:
    # ── Your agent logic here ──────────────────────────────────────
    assignments = []
    for patient in sorted(obs.patients_waiting, key=lambda p: p.wait_hours, reverse=True):
        clinic = obs.clinics[0]  # replace with your selection logic
        assignments.append(PatientAssignment(
            patient_id=patient.patient_id,
            clinic_id=clinic.clinic_id,
            transport=TransportMode.TELEMEDICINE,
            medicine=patient.medicine_needed,
            priority=1,
        ))
    # ──────────────────────────────────────────────────────────────
    result = env.step(Action(assignments=assignments))
    obs = result.observation
    if result.done:
        break

score = grade("medium", env.state())
print(f"Final score: {score.score:.4f} | Passed: {score.passed}")
```

---

## 📜 Conditions

8 real clinical conditions are simulated with severity distributions and medicine requirements:
`Chest Pain`, `High Fever`, `Severe Injury`, `Diabetic Crisis`,
`Respiratory Distress`, `Minor Laceration`, `Acute Abdominal Pain`, `Hypertensive Crisis`

---

## 🙏 Acknowledgements

Inspired by the real challenges faced by ASHA workers and district health officers across
rural Maharashtra and India at large. Built for the OpenEnv hackathon.

---

**Author:** Amit Tipare | NMIMS Global University, Dhule  
**GitHub:** [amittipare585858/rural-health-net](https://github.com/amittipare585858/rural-health-net)  
**HF Space:** [spaces/amittipare585858/rural-health-net](https://huggingface.co/spaces/amittipare585858/rural-health-net)
