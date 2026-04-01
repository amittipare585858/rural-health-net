"""
RuralHealthNet-v0 — Baseline Inference Script (OpenAI API)
===========================================================
Uses the OpenAI API client to run an LLM agent against the environment.
Reads credentials from environment variables.

Required env var:
    OPENAI_API_KEY  — your OpenAI API key

Usage:
    python agents/run_baseline.py                        # all tasks, gpt-4o-mini
    python agents/run_baseline.py --task easy            # single task
    python agents/run_baseline.py --model gpt-4o         # different model
    python agents/run_baseline.py --json                 # JSON output
    python agents/run_baseline.py --heuristic            # no API key needed
"""

from __future__ import annotations
import os, sys, json, argparse, textwrap
sys.path.insert(0, ".")

from openai import OpenAI
from env.health_env import RuralHealthNetEnv
from env.models import Action, PatientAssignment, TransportMode, MedicineType
from env.graders import grade
from agents.baseline_agent import HeuristicAgent
from tasks import TASK_CONFIGS


SYSTEM_PROMPT = textwrap.dedent("""
You are an AI health coordinator managing rural telemedicine triage.
Each step you receive a JSON observation with patients_waiting and clinics.

Triage priorities (treat in this order):
1. CRITICAL patients — use ambulance if available, treat immediately
2. MODERATE patients — self_transport is fine
3. MILD patients    — use telemedicine to save doctor slots

Medicine: always prescribe the patient's medicine_needed field.
Never assign to a clinic where doctors_available=0, is_online=false,
or current_load >= capacity.

Respond ONLY with this exact JSON format, no markdown:
{
  "assignments": [
    {
      "patient_id": "<id>",
      "clinic_id": <int>,
      "transport": "<ambulance|telemedicine|self_transport|defer>",
      "medicine": "<antibiotics|painkillers|cardiac|antidiabetic|none>",
      "priority": <1-5>
    }
  ]
}
""").strip()


class LLMAgent:
    def __init__(self, model="gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set.\n"
                "Run: export OPENAI_API_KEY=sk-...\n"
                "Or use --heuristic flag for local testing."
            )
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def act(self, obs_dict: dict) -> Action:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    seed=42,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": json.dumps(obs_dict)},
                    ],
                    max_tokens=2048,
                )
                data = json.loads(resp.choices[0].message.content)
                return _parse_action(data)
            except Exception as e:
                if attempt == 2:
                    print(f"  [WARN] LLM failed: {e}")
                    return Action(assignments=[])
        return Action(assignments=[])


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


def _obs_to_dict(obs) -> dict:
    return {
        "step": obs.step_number,
        "time_hours": obs.time_hours,
        "epidemic_active": obs.epidemic_active,
        "patients_waiting": [
            {"patient_id": p.patient_id, "severity": p.severity.value,
             "condition": p.condition, "wait_hours": p.wait_hours,
             "location_clinic": p.location_clinic,
             "medicine_needed": p.medicine_needed.value,
             "deterioration_rate": p.deterioration_rate}
            for p in obs.patients_waiting
        ],
        "clinics": [
            {"clinic_id": c.clinic_id, "name": c.name,
             "doctors_available": c.doctors_available,
             "ambulances_available": c.ambulances_available,
             "capacity": c.capacity, "current_load": c.current_load,
             "is_online": c.is_online,
             "medicine_stock": {k.value: v for k, v in c.medicine_stock.items()}}
            for c in obs.clinics
        ],
    }


def run_episode(task_id, seed=42, model="gpt-4o-mini", use_heuristic=False, verbose=False):
    config = TASK_CONFIGS[task_id]
    env    = RuralHealthNetEnv(task_config=config, seed=seed)
    agent  = HeuristicAgent() if use_heuristic else LLMAgent(model=model)
    use_llm = not use_heuristic

    obs, done, total_reward, step_log = env.reset(), False, 0.0, []

    while not done:
        action  = agent.act(_obs_to_dict(obs)) if use_llm else agent.act(obs)
        result  = env.step(action)
        obs, done = result.observation, result.done
        total_reward += result.reward
        if verbose:
            step_log.append({"step": result.info.get("step"),
                             "reward": result.reward,
                             "treated": result.info.get("treated_this_step", 0),
                             "lost": result.info.get("lost_this_step", 0)})

    fs = env.state()
    gr = grade(task_id, fs)

    out = {
        "task_id": task_id, "seed": seed,
        "model": "heuristic" if use_heuristic else model,
        "total_steps": fs.step_number,
        "cumulative_reward": round(total_reward, 4),
        "grader_score": gr.score, "passed": gr.passed,
        "treatment_rate": gr.treatment_rate,
        "deterioration_rate": gr.deterioration_rate,
        "resource_efficiency": gr.resource_efficiency,
        "breakdown": gr.breakdown,
    }
    if verbose:
        out["step_log"] = step_log
    return out


def main():
    parser = argparse.ArgumentParser(description="RuralHealthNet-v0 Baseline Runner")
    parser.add_argument("--task",      default="all", choices=["easy","medium","hard","all"])
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--model",     default="gpt-4o-mini")
    parser.add_argument("--heuristic", action="store_true", help="Use built-in heuristic (no API key)")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--json",      action="store_true")
    args = parser.parse_args()

    tasks_to_run = ["easy","medium","hard"] if args.task=="all" else [args.task]
    results = []

    for task_id in tasks_to_run:
        label = "heuristic" if args.heuristic else args.model
        print(f"\n{'='*58}\n  Task: {task_id.upper()} | Agent: {label} | seed={args.seed}\n{'='*58}")
        r = run_episode(task_id, args.seed, args.model, args.heuristic, args.verbose)
        results.append(r)
        if not args.json:
            print(f"  Score:               {r['grader_score']:.4f}")
            print(f"  Passed:              {'✅ YES' if r['passed'] else '❌ NO'}")
            print(f"  Cumulative Reward:   {r['cumulative_reward']:.2f}")
            print(f"  Treatment Rate:      {r['treatment_rate']*100:.1f}%")
            print(f"  Deterioration Rate:  {r['deterioration_rate']*100:.1f}%")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*58}\n  BASELINE SUMMARY\n{'='*58}")
        for r in results:
            print(f"  {r['task_id'].upper():8s}  score={r['grader_score']:.4f}  {'✅ PASS' if r['passed'] else '❌ FAIL'}")
        avg = sum(r["grader_score"] for r in results) / len(results)
        print(f"\n  Avg Score: {avg:.4f} | Model: {results[0]['model']} | Seed: {results[0]['seed']}\n{'='*58}\n")


if __name__ == "__main__":
    main()
