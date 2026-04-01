"""
RuralHealthNet-v0 — Agent Graders
Each grader evaluates a completed episode and returns a score 0.0–1.0
with detailed breakdown for analysis.
"""

from __future__ import annotations
from env.models import HealthNetState, GraderResult, Severity


# ─── Utility ─────────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


# ─── Easy Grader ──────────────────────────────────────────────────────────────

def grade_easy(state: HealthNetState) -> GraderResult:
    """
    Easy Task Grader — Single Clinic Triage
    Score = weighted treatment rate
    - Critical patients treated:  weight 3
    - Moderate patients treated:  weight 2
    - Mild patients treated:      weight 1
    Pass threshold: score >= 0.50
    """
    treated   = state.patients_treated
    total     = state.total_patients_seen
    lost      = state.patients_lost
    deterred  = state.patients_deteriorated

    # Simple treatment rate (normalized)
    treatment_rate    = _safe_div(treated, total)
    deterioration_pct = _safe_div(deterred, total)
    loss_penalty      = _safe_div(lost, total)

    score = (
        0.70 * treatment_rate
      - 0.20 * loss_penalty
      - 0.10 * deterioration_pct
    )
    score = max(0.0, min(1.0, score))

    return GraderResult(
        task_id="easy",
        score=round(score, 4),
        treatment_rate=round(treatment_rate, 4),
        deterioration_rate=round(deterioration_pct, 4),
        resource_efficiency=1.0,  # not evaluated at easy level
        breakdown={
            "patients_treated": treated,
            "total_patients":   total,
            "patients_lost":    lost,
            "deteriorated":     deterred,
            "treatment_rate":   round(treatment_rate, 3),
            "loss_penalty":     round(loss_penalty, 3),
        },
        passed=score >= 0.50,
    )


# ─── Medium Grader ────────────────────────────────────────────────────────────

def grade_medium(state: HealthNetState) -> GraderResult:
    """
    Medium Task Grader — Multi-Clinic Resource Management
    Score considers treatment rate + resource efficiency + critical outcomes.
    Pass threshold: score >= 0.55
    """
    treated   = state.patients_treated
    total     = state.total_patients_seen
    lost      = state.patients_lost
    deterred  = state.patients_deteriorated

    treatment_rate    = _safe_div(treated, total)
    deterioration_pct = _safe_div(deterred, total)
    loss_penalty      = _safe_div(lost, total)

    # Resource efficiency: how much medicine stock remains vs wasted assignments
    total_medicine = sum(
        sum(c.medicine_stock.values()) for c in state.clinics
    )
    max_possible_stock = 30 * len(state.clinics)  # baseline
    resource_efficiency = _safe_div(total_medicine, max_possible_stock)
    resource_efficiency = min(1.0, resource_efficiency)

    score = (
        0.55 * treatment_rate
      + 0.20 * resource_efficiency
      - 0.15 * loss_penalty
      - 0.10 * deterioration_pct
    )
    score = max(0.0, min(1.0, score))

    return GraderResult(
        task_id="medium",
        score=round(score, 4),
        treatment_rate=round(treatment_rate, 4),
        deterioration_rate=round(deterioration_pct, 4),
        resource_efficiency=round(resource_efficiency, 4),
        breakdown={
            "patients_treated":    treated,
            "total_patients":      total,
            "patients_lost":       lost,
            "deteriorated":        deterred,
            "treatment_rate":      round(treatment_rate, 3),
            "resource_efficiency": round(resource_efficiency, 3),
            "loss_penalty":        round(loss_penalty, 3),
        },
        passed=score >= 0.55,
    )


# ─── Hard Grader ──────────────────────────────────────────────────────────────

def grade_hard(state: HealthNetState) -> GraderResult:
    """
    Hard Task Grader — Epidemic Surge Response
    Score = treatment rate + resource efficiency + epidemic containment
    - Epidemic containment: how many patients treated during epidemic phase
    - Critical survival rate weighted heavily
    Pass threshold: score >= 0.60
    """
    treated   = state.patients_treated
    total     = state.total_patients_seen
    lost      = state.patients_lost
    deterred  = state.patients_deteriorated

    treatment_rate    = _safe_div(treated, total)
    deterioration_pct = _safe_div(deterred, total)
    loss_penalty      = _safe_div(lost, total) * 2.0  # double penalty at hard level

    # Resource efficiency
    total_medicine    = sum(sum(c.medicine_stock.values()) for c in state.clinics)
    max_stock         = 50 * len(state.clinics)
    resource_eff      = min(1.0, _safe_div(total_medicine, max_stock))

    # Epidemic handled: reward if epidemic was active but losses stayed low
    epidemic_bonus = 0.0
    if state.epidemic_active:
        # If epidemic happened and loss rate stayed below 10%, award bonus
        if loss_penalty < 0.10:
            epidemic_bonus = 0.15
        elif loss_penalty < 0.20:
            epidemic_bonus = 0.05

    score = (
        0.50 * treatment_rate
      + 0.20 * resource_eff
      - 0.20 * min(1.0, loss_penalty)
      - 0.10 * deterioration_pct
      + epidemic_bonus
    )
    score = max(0.0, min(1.0, score))

    return GraderResult(
        task_id="hard",
        score=round(score, 4),
        treatment_rate=round(treatment_rate, 4),
        deterioration_rate=round(deterioration_pct, 4),
        resource_efficiency=round(resource_eff, 4),
        breakdown={
            "patients_treated":    treated,
            "total_patients":      total,
            "patients_lost":       lost,
            "deteriorated":        deterred,
            "epidemic_active":     state.epidemic_active,
            "epidemic_bonus":      epidemic_bonus,
            "treatment_rate":      round(treatment_rate, 3),
            "resource_efficiency": round(resource_eff, 3),
            "loss_penalty":        round(min(1.0, loss_penalty), 3),
        },
        passed=score >= 0.60,
    )


# ─── Router ───────────────────────────────────────────────────────────────────

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

def grade(task_id: str, state: HealthNetState) -> GraderResult:
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(GRADERS)}")
    return GRADERS[task_id](state)
