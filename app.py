"""
RuralHealthNet-v0 — FastAPI App (Hugging Face Spaces)
Exposes the OpenEnv API over HTTP + openenv validate-compatible endpoints.
"""
from __future__ import annotations
import sys, yaml, copy
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env.health_env import RuralHealthNetEnv
from env.models import Action, HealthNetState, Observation, StepResult
from env.graders import grade
from tasks import TASK_CONFIGS


app = FastAPI(
    title="RuralHealthNet-v0",
    description=(
        "OpenEnv environment: rural telemedicine triage and resource allocation. "
        "AI agents coordinate doctors, medicines, and ambulances across rural clinics."
    ),
    version="1.0.0",
    tags_metadata=[
        {"name": "openenv", "description": "Core OpenEnv API"},
        {"name": "meta",    "description": "Metadata and validation endpoints"},
    ],
)

_sessions: dict[str, RuralHealthNetEnv] = {}


# ─── Request Models ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id:    str = "easy"
    seed:       int = 42
    session_id: str = "default"

class StepRequest(BaseModel):
    action:     Action
    session_id: str = "default"

class GradeRequest(BaseModel):
    task_id:    str
    session_id: str = "default"


# ─── Root / Meta ─────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {
        "name":        "RuralHealthNet-v0",
        "version":     "1.0.0",
        "openenv":     True,
        "description": "OpenEnv: Rural Telemedicine Triage & Resource Allocation",
        "tasks":       list(TASK_CONFIGS.keys()),
        "endpoints": {
            "reset":     "POST /reset",
            "step":      "POST /step",
            "state":     "GET  /state/{session_id}",
            "grade":     "POST /grade",
            "tasks":     "GET  /tasks",
            "spec":      "GET  /openenv.yaml",
            "validate":  "GET  /validate",
        },
        "docs": "/docs",
    }


@app.get("/openenv.yaml", tags=["meta"], response_class=PlainTextResponse)
def get_spec():
    """Return the raw openenv.yaml specification."""
    with open("openenv.yaml", "r") as f:
        return f.read()


@app.get("/validate", tags=["meta"])
def validate():
    """
    openenv validate compatibility endpoint.
    Checks that the spec, tasks, and API are all properly configured.
    """
    issues = []
    checks = {}

    # 1. YAML spec parseable
    try:
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)
        checks["openenv_yaml"] = "ok"
    except Exception as e:
        checks["openenv_yaml"] = f"ERROR: {e}"
        issues.append(str(e))

    # 2. All 3 tasks present
    tasks = [t["id"] for t in spec.get("tasks", [])]
    expected = {"easy", "medium", "hard"}
    if expected.issubset(set(tasks)):
        checks["tasks_3_minimum"] = f"ok ({tasks})"
    else:
        checks["tasks_3_minimum"] = f"MISSING: {expected - set(tasks)}"
        issues.append(f"Missing tasks: {expected - set(tasks)}")

    # 3. reset/step/state functional
    try:
        env = RuralHealthNetEnv(TASK_CONFIGS["easy"], seed=0)
        obs = env.reset()
        assert obs.step_number == 0
        checks["reset"] = "ok"

        result = env.step(Action(assignments=[]))
        assert hasattr(result, "reward") and hasattr(result, "done")
        checks["step"] = f"ok (reward={result.reward})"

        state = env.state()
        assert hasattr(state, "patients_treated")
        checks["state"] = "ok"
    except Exception as e:
        checks["reset/step/state"] = f"ERROR: {e}"
        issues.append(str(e))

    # 4. All graders functional
    try:
        from env.graders import grade as _grade
        for tid in ["easy", "medium", "hard"]:
            env2 = RuralHealthNetEnv(TASK_CONFIGS[tid], seed=0)
            env2.reset()
            env2.step(Action(assignments=[]))
            gr = _grade(tid, env2.state())
            assert 0.0 <= gr.score <= 1.0
        checks["graders"] = "ok"
    except Exception as e:
        checks["graders"] = f"ERROR: {e}"
        issues.append(str(e))

    # 5. Reward function present in spec
    if spec.get("reward"):
        checks["reward_spec"] = "ok"
    else:
        checks["reward_spec"] = "MISSING reward in openenv.yaml"
        issues.append("reward spec missing")

    return {
        "valid":   len(issues) == 0,
        "checks":  checks,
        "issues":  issues,
        "summary": f"{'✅ PASSED' if not issues else '❌ FAILED'} — {len(checks)} checks, {len(issues)} issues",
    }


# ─── OpenEnv Core API ────────────────────────────────────────────────────────

@app.get("/tasks", tags=["openenv"])
def list_tasks():
    return {
        tid: {
            "name":        cfg["name"],
            "description": cfg["description"],
            "max_steps":   cfg["max_steps"],
            "num_clinics": len(cfg["clinics"]),
        }
        for tid, cfg in TASK_CONFIGS.items()
    }


@app.post("/reset", tags=["openenv"])
def reset(req: ResetRequest):
    """Reset environment. Returns initial Observation."""
    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'. Choose: {list(TASK_CONFIGS)}")
    env = RuralHealthNetEnv(task_config=TASK_CONFIGS[req.task_id], seed=req.seed)
    obs = env.reset()
    _sessions[req.session_id] = env
    return {"session_id": req.session_id, "task_id": req.task_id,
            "seed": req.seed, "observation": obs.model_dump()}


@app.post("/step", tags=["openenv"])
def step(req: StepRequest):
    """Apply action, advance one step. Returns StepResult."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call /reset first.")
    return env.step(req.action).model_dump()


@app.get("/state/{session_id}", tags=["openenv"])
def get_state(session_id: str = "default"):
    """Return full internal environment state."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call /reset first.")
    return env.state().model_dump()


@app.post("/grade", tags=["openenv"])
def grade_episode(req: GradeRequest):
    """Grade the completed episode. Returns GraderResult with score 0.0-1.0."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call /reset first.")
    return grade(req.task_id, env.state()).model_dump()


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "environment": "RuralHealthNet-v0"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
