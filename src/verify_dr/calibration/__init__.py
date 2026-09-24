"""M4a — calibration (ANALYSIS_PLAN.md s4): prior correction, temperature, EM."""

from .stages import (  # noqa: F401
    EM_MAX_ITER,
    EM_TOL,
    T_BOUNDS,
    EMResult,
    coral_probs,
    em_prior,
    fit_temperature,
    grade_prior,
    nll,
    predicted_confidence,
    prior_correct,
    stage01,
)
