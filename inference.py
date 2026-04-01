"""
RuralHealthNet-v0 — inference.py (Pre-Submission Baseline)
===========================================================
MANDATORY: This file is the official inference script for the hackathon.
Must be named `inference.py` and placed in the root directory.

Environment variables (set before running):
    API_BASE_URL   The LLM API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     The model identifier (e.g. gpt-4o-mini)
    HF_TOKEN       Your Hugging Face / API key

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=hf-your-token-or-openai-key

    python inference.py               # runs all 3 tasks
    python inference.py --task easy   # single task
    python inference.py --json        # JSON output

Infra compliance:
    - Runtime < 20 min on vcpu=2, memory=8gb
    - Max 15 LLM calls per task (capped internally)
    - Lightweight patient batching to stay within memory
"""

from __future__ import annotations
import os, sys, json, argparse, textwrap, time
sys.path.insert(0, ".")

from openai import OpenAI
from env.health_env import RuralHealthNetEnv
from env.models import Action, PatientAssignment, TransportMode, MedicineType
from env.graders import grade
from tasks import TASK_CONFIGS


# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# Infra compliance: cap LLM calls so runtime stays < 20 min
# Easy=30 steps, Medium=60 steps, Hard=100 steps — call LLM every N steps
LLM_CALL_EVERY = {
    "easy":   2,   # 15 LLM calls max
    "medium": 4,   # 15 LLM calls max
    "hard":   7,   # ~14 LLM calls max
}

MAX_PATIENTS_IN_PROMPT = 12   # truncate long queues to keep tokens low


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI health coordinator for rural telemedicine triage in India.
You receive a JSON observation with patients_waiting and clinics.

PRIORITY ORDER (most urgent first):
1. CRITICAL patients — use ambulance if available, must treat immediately
2. MODERATE patients — self_transport, treat within 6 hours
3. MILD patients     — use telemedicine to save doctor slots

RULES:
- Never assign to a clinic where doctors_available=0 or is_online=false
- Never exceed clinic capacity (capacity - current_load)
- Always prescribe the patient's medicine_needed field exactly
- Skip (defer) patients if no clinic can safely take them

Respond ONLY with valid JSON, no markdown fences:
{
  "assignments": [
    {
      "patient_id": "<string>",
      "clinic_id": <integer>,
      "transport": "<ambulance|telemedicine|self_transport|defer>",
      "medicine": "<antibiotics|painkillers|cardiac|antidiabetic|none>",
      "priority": <1-5>
    }
  ]
}
""").strip()


# ─── OpenAI Client (using API_BASE_URL + HF_TOKEN) ───────────────────────────

def get_client() -> OpenAI:
    """
    Initialize OpenAI client using the mandatory environment variables:
    API_BASE_URL, MODEL_NAME, HF_TOKEN.
    """
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set.\n"
            "Set it with: export HF_TOKEN=<your-api-key>\n"
            "Also set:    export API_BASE_URL=https://api.openai.com/v1\n"
            "             export MODEL_NAME=gpt-4o-mini"
        )
    return OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )


# ─── Action Parser ────────────────────────────────────────────────────────────

def _parse_action(data: dict) -> Action:
    assignments = []
    for item in data.get("assignments", []):
        try:
            t = item.get("transport", "self_transport")
            m = item.get("medicine",  "none")
            transport = TransportMode(t) if t in TransportMode._value2member_map_ else TransportMode.SELF_TRANSPORT
            medicine  = MedicineType(m)  if m in MedicineType._value2member_map_  else MedicineType.NONE
            assignments.append(PatientAssignment(
                patient_id=str(item["patient_id"]),
                clinic_id=int(item["clinic_id"]),
                transport=transport,
                medicine=medicine,
                priority=int(item.get("priority", 3)),
            ))
        except Exception:
            continue
    return Action(assignments=assignments)


def _obs_to_dict(obs, task_id: str) -> dict:
    """
    Serialize observation to a compact dict for the LLM.
    Truncates patient list to MAX_PATIENTS_IN_PROMPT to stay memory-safe.
    Prioritizes critical patients in the truncation window.
    """
    # Sort: critical first, then by wait_hours descending
    severity_order = {"critical": 0, "moderate": 1, "mild": 2}
    sorted_patients = sorted(
        obs.patients_waiting,
        key=lambda p: (severity_order.get(p.severity.value, 3), -p.wait_hours)
    )
    top_patients = sorted_patients[:MAX_PATIENTS_IN_PROMPT]

    return {
        "step":            obs.step_number,
        "time_hours":      obs.time_hours,
        "epidemic_active": obs.epidemic_active,
        "patients_shown":  len(top_patients),
        "patients_total":  len(obs.patients_waiting),
        "patients_waiting": [
            {
                "patient_id":       p.patient_id,
                "severity":         p.severity.value,
                "condition":        p.condition,
                "wait_hours":       p.wait_hours,
                "location_clinic":  p.location_clinic,
                "medicine_needed":  p.medicine_needed.value,
                "deterioration_rate": p.deterioration_rate,
            }
            for p in top_patients
        ],
        "clinics": [
            {
                "clinic_id":            c.clinic_id,
                "name":                 c.name,
                "doctors_available":    c.doctors_available,
                "ambulances_available": c.ambulances_available,
                "capacity":             c.capacity,
                "current_load":         c.current_load,
                "is_online":            c.is_online,
                "medicine_stock":       {k.value: v for k, v in c.medicine_stock.items()},
            }
            for c in obs.clinics
        ],
    }


# ─── Heuristic Fallback (runs on non-LLM steps) ──────────────────────────────

def _heuristic_action(obs) -> Action:
    """
    Fast rule-based fallback used on steps where LLM is not called.
    Severity-sorted triage with correct medicine/transport selection.
    """
    severity_order = {"critical": 0, "moderate": 1, "mild": 2}
    sorted_p = sorted(
        obs.patients_waiting,
        key=lambda p: (severity_order.get(p.severity.value, 3), -p.wait_hours)
    )
    clinic_state = {
        c.clinic_id: {
            "doctors": c.doctors_available,
            "ambulances": c.ambulances_available,
            "load": c.current_load,
            "capacity": c.capacity,
            "online": c.is_online,
            "stock": dict(c.medicine_stock),
        }
        for c in obs.clinics
    }
    assignments = []
    for p in sorted_p:
        best = None
        for c in obs.clinics:
            cs = clinic_state[c.clinic_id]
            if cs["online"] and cs["doctors"] > 0 and cs["load"] < cs["capacity"]:
                if best is None or c.clinic_id == p.location_clinic:
                    best = c.clinic_id
                    if c.clinic_id == p.location_clinic:
                        break
        if best is None:
            continue
        cs = clinic_state[best]
        if p.severity.value == "critical" and cs["ambulances"] > 0:
            transport = TransportMode.AMBULANCE
            cs["ambulances"] -= 1
        elif p.severity.value == "mild":
            transport = TransportMode.TELEMEDICINE
        else:
            transport = TransportMode.SELF_TRANSPORT

        needed = p.medicine_needed
        if needed != MedicineType.NONE and cs["stock"].get(needed, 0) > 0:
            medicine = needed
            cs["stock"][needed] -= 1
        else:
            medicine = MedicineType.NONE

        cs["doctors"] -= 1
        cs["load"]    += 1
        assignments.append(PatientAssignment(
            patient_id=p.patient_id,
            clinic_id=best,
            transport=transport,
            medicine=medicine,
            priority=severity_order.get(p.severity.value, 3) + 1,
        ))
    return Action(assignments=assignments)


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = 42, verbose: bool = False) -> dict:
    """
    Run one full episode using the OpenAI client.
    LLM is called every LLM_CALL_EVERY[task_id] steps to stay within 20min limit.
    Heuristic fallback used on intermediate steps.
    """
    client = get_client()
    config = TASK_CONFIGS[task_id]
    env    = RuralHealthNetEnv(task_config=config, seed=seed)
    call_every = LLM_CALL_EVERY[task_id]

    obs, done = env.reset(), False
    total_reward = 0.0
    llm_calls    = 0
    step_log     = []
    last_action  = Action(assignments=[])

    while not done:
        step = obs.step_number

        # Call LLM on scheduled steps; use heuristic on others
        if step % call_every == 0:
            obs_dict = _obs_to_dict(obs, task_id)
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=0.0,
                        seed=seed,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": json.dumps(obs_dict)},
                        ],
                        max_tokens=1024,
                    )
                    data        = json.loads(resp.choices[0].message.content)
                    last_action = _parse_action(data)
                    llm_calls  += 1
                    break
                except Exception as e:
                    if attempt == 2:
                        if verbose:
                            print(f"  [WARN] step {step}: LLM failed ({e}), using heuristic")
                        last_action = _heuristic_action(obs)
        else:
            last_action = _heuristic_action(obs)

        result       = env.step(last_action)
        obs, done    = result.observation, result.done
        total_reward += result.reward

        if verbose:
            step_log.append({
                "step":    step,
                "reward":  result.reward,
                "treated": result.info.get("treated_this_step", 0),
                "lost":    result.info.get("lost_this_step", 0),
                "llm":     step % call_every == 0,
            })

    fs = env.state()
    gr = grade(task_id, fs)

    out = {
        "task_id":             task_id,
        "model":               MODEL_NAME,
        "api_base_url":        API_BASE_URL,
        "seed":                seed,
        "total_steps":         fs.step_number,
        "llm_calls":           llm_calls,
        "cumulative_reward":   round(total_reward, 4),
        "grader_score":        gr.score,
        "passed":              gr.passed,
        "treatment_rate":      gr.treatment_rate,
        "deterioration_rate":  gr.deterioration_rate,
        "resource_efficiency": gr.resource_efficiency,
        "breakdown":           gr.breakdown,
    }
    if verbose:
        out["step_log"] = step_log
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RuralHealthNet-v0 inference.py — official baseline runner",
    )
    parser.add_argument("--task",    default="all", choices=["easy","medium","hard","all"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json",    action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    print(f"\nRuralHealthNet-v0 — Baseline Inference")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  HF_TOKEN     : {'✅ set' if HF_TOKEN else '❌ NOT SET — export HF_TOKEN=...'}")
    if not HF_TOKEN:
        sys.exit(1)

    tasks_to_run = ["easy","medium","hard"] if args.task == "all" else [args.task]
    results      = []
    t_start      = time.time()

    for task_id in tasks_to_run:
        print(f"\n{'='*58}")
        print(f"  Task: {task_id.upper()} | Model: {MODEL_NAME} | seed={args.seed}")
        print(f"{'='*58}")
        t0 = time.time()
        r  = run_episode(task_id, args.seed, args.verbose)
        elapsed = round(time.time() - t0, 1)
        results.append(r)
        if not args.json:
            print(f"  Score:               {r['grader_score']:.4f}")
            print(f"  Passed:              {'✅ YES' if r['passed'] else '❌ NO'}")
            print(f"  Cumulative Reward:   {r['cumulative_reward']:.2f}")
            print(f"  Treatment Rate:      {r['treatment_rate']*100:.1f}%")
            print(f"  Deterioration Rate:  {r['deterioration_rate']*100:.1f}%")
            print(f"  LLM Calls:           {r['llm_calls']}")
            print(f"  Time:                {elapsed}s")

    total_time = round(time.time() - t_start, 1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*58}\n  FINAL SUMMARY\n{'='*58}")
        for r in results:
            print(f"  {r['task_id'].upper():8s}  score={r['grader_score']:.4f}  {'✅ PASS' if r['passed'] else '❌ FAIL'}")
        avg = sum(r["grader_score"] for r in results) / len(results)
        print(f"\n  Avg Score  : {avg:.4f}")
        print(f"  Total Time : {total_time}s  (limit: 1200s / 20min)")
        print(f"  Model      : {MODEL_NAME}")
        print(f"  API Base   : {API_BASE_URL}")
        print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
