"""M4a — calibration (ANALYSIS_PLAN.md s4): prior correction, temperature, EM;
and D13's bias-corrected temperature scaling (the plan's addendum, A.2)."""

from .stages import (  # noqa: F401
    BCTS,
    EM_MAX_ITER,
    EM_TOL,
    T_BOUNDS,
    EMResult,
    bcts_probs,
    coral_probs,
    em_prior,
    fit_bcts,
    fit_bias,
    fit_temperature,
    grade_prior,
    nll,
    predicted_confidence,
    prior_correct,
    stage01,
)
