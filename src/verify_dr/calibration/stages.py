"""M4a — calibration, exactly as ANALYSIS_PLAN.md section 4 pins it.

Applied to M1's *raw* cumulative logits z, in this order:

    stage 1   z / T                          T fitted on the calibration split
              -> CORAL class probabilities   P(0) = 1 - s0, P(k) = s(k-1) - s(k), P(4) = s3
    stage 0   p_src  ~  p * pi_src / u       u uniform: the sampler trained on it (s4.1)
    stage 2   p_tgt  ~  p_src * pi_tgt / pi_src
              pi_tgt estimated by EM from the target's UNLABELLED images (externals only)

Stage 0 is written before stage 1 in the plan because it is the correction the freeze
missed (D11); temperature is applied to the logits, so it acts first in the
computation. Stage 0 is parameter-free: pi_src is a count of the EyePACS training
rows, and u is uniform because `stratified_exposure` shows every grade equally often.

**Calibration changes probabilities, never the predicted grade.** y-hat is the CORAL
threshold count on the raw logits in every arm and at every stage (s3); the
functions here never return a grade, so nothing downstream can pick one up from
calibrated probabilities by mistake.

numpy only, on purpose: the same code must give the same numbers on Kaggle and in
the tests, and a bounded scalar search is short enough to write out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

NUM_CLASSES = 5
T_BOUNDS = (0.05, 20.0)      # plan s4.2, stage 1
EM_TOL = 1e-6                # plan s4.2, stage 2: max |delta pi|
EM_MAX_ITER = 1000
_GRID = 121                  # log-spaced starting grid for the temperature search


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def coral_probs(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Class probabilities [n, 5] from cumulative logits [n, 4] divided by T.

    The numpy twin of `models.grading.cumulative_to_probs`, including its clamp
    and renormalisation, so a probability computed here equals the one training
    evaluated at T = 1.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != NUM_CLASSES - 1:
        raise ValueError(f"expected cumulative logits of shape [n, 4], got {z.shape}")
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    s = _sigmoid(z / temperature)
    n = len(s)
    upper = np.concatenate([np.ones((n, 1)), s], axis=1)
    lower = np.concatenate([s, np.zeros((n, 1))], axis=1)
    p = np.clip(upper - lower, 0.0, None)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def prior_correct(p: np.ndarray, to_prior: Sequence[float],
                  from_prior: Optional[Sequence[float]] = None) -> np.ndarray:
    """p(y|x) under `from_prior` -> under `to_prior`: multiply by the ratio, renormalise.

    `from_prior=None` means uniform, which is stage 0.
    """
    p = np.asarray(p, dtype=np.float64)
    to_prior = np.asarray(to_prior, dtype=np.float64)
    from_prior = (np.full(p.shape[1], 1.0 / p.shape[1]) if from_prior is None
                  else np.asarray(from_prior, dtype=np.float64))
    if np.any(from_prior <= 0):
        raise ValueError("a prior being corrected away from cannot have a zero class")
    q = p * (to_prior / from_prior)
    return q / np.clip(q.sum(axis=1, keepdims=True), 1e-300, None)


def stage01(z: np.ndarray, temperature: float, pi_src: Sequence[float]) -> np.ndarray:
    """Stages 1 then 0: the calibrated source posterior p_src(y | x)."""
    return prior_correct(coral_probs(z, temperature), pi_src, None)


def nll(p: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true grades (clipped as `metrics` clips)."""
    y = np.asarray(y).astype(int)
    picked = np.asarray(p)[np.arange(len(y)), y]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def _golden(f: Callable[[float], float], lo: float, hi: float,
            tol: float = 1e-10, max_iter: int = 200) -> float:
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def fit_temperature(z: np.ndarray, y: np.ndarray, pi_src: Optional[Sequence[float]] = None,
                    bounds: Tuple[float, float] = T_BOUNDS) -> float:
    """T minimising the NLL of the true grades on [bounds] (plan s4.2, stage 1).

    With `pi_src` the NLL is that of stage 0+1 probabilities -- the frozen pipeline.
    Without it, of stage 1 alone -- D1's comparison variant, fitted separately so the
    effect of the prior correction is visible rather than folded into T.

    A log-spaced grid finds the basin, golden-section search refines it. The grid
    guards against a second, shallower minimum capturing a purely local search.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y).astype(int)

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        p = stage01(z, t, pi_src) if pi_src is not None else coral_probs(z, t)
        return nll(p, y)

    return _search_temperature(loss, bounds)


def _search_temperature(loss: Callable[[float], float], bounds: Tuple[float, float]) -> float:
    """Minimise `loss(log T)` on [bounds]: a log-spaced grid, then golden section."""
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    grid = np.linspace(lo, hi, _GRID)
    values = np.array([loss(g) for g in grid])
    i = int(np.argmin(values))
    left, right = grid[max(i - 1, 0)], grid[min(i + 1, _GRID - 1)]
    best = _golden(loss, left, right)
    if loss(best) > values[i]:           # never worse than the grid point
        best = grid[i]
    return float(math.exp(best))


# ------------------------------------------------------------------ D13


@dataclass(frozen=True)
class BCTS:
    """Bias-corrected temperature scaling: q(y|x) ~ p_T(y|x) * exp(bias_y), bias_0 = 0."""
    temperature: float
    bias: np.ndarray


def bcts_probs(z: np.ndarray, fit: BCTS) -> np.ndarray:
    p = coral_probs(z, fit.temperature)
    w = np.exp(np.asarray(fit.bias, dtype=np.float64) - np.max(fit.bias))
    q = p * w
    return q / np.clip(q.sum(axis=1, keepdims=True), 1e-300, None)


def _log_probs(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-300, None))


def _bias_nll(logp: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    s = logp + bias
    top = s.max(axis=1, keepdims=True)
    lse = (top + np.log(np.exp(s - top).sum(axis=1, keepdims=True)))[:, 0]
    return float((lse - s[np.arange(len(y)), y]).mean())


def fit_bias(p: np.ndarray, y: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """The per-grade log-biases (bias_0 = 0) minimising the NLL of q ~ p * exp(bias).

    The NLL is convex in the biases, so a descent method finds the global minimum.
    At it, the gradient -- the sum over images of q(c | x) minus the count of grade
    c -- is zero: the mean fitted posterior equals the observed grade mix, which is
    exactly EM's premise.

    Newton's step, damped Levenberg-Marquardt style when it does not descend: at an
    extreme temperature some grades' probabilities saturate at zero and the Hessian
    is near-singular, and an undamped step there goes nowhere useful.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    n, k = p.shape
    counts = np.bincount(y, minlength=k).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"every grade needs calibration images; counts {counts.tolist()}")
    logp = _log_probs(p)
    bias = np.zeros(k)
    current = _bias_nll(logp, y, bias)
    for _ in range(max_iter):
        s = logp + bias
        q = np.exp(s - s.max(axis=1, keepdims=True))
        q /= q.sum(axis=1, keepdims=True)
        grad = (q.sum(axis=0) - counts)[1:] / n
        if np.abs(grad).max() < 1e-12:
            break
        hess = (np.diag(q.sum(axis=0)) - q.T @ q)[1:, 1:] / n
        moved = False
        for damping in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0):
            try:
                step = np.linalg.solve(hess + damping * np.eye(k - 1), grad)
            except np.linalg.LinAlgError:
                continue                # exactly singular: the damped system is not
            t = 1.0
            while t > 1e-6:
                trial = bias.copy()
                trial[1:] -= t * step
                value = _bias_nll(logp, y, trial)
                if value < current:
                    bias, current, moved = trial, value, True
                    break
                t /= 2.0
            if moved:
                break
        if not moved:
            break                       # no descent in any direction tried: optimal
    return bias


def _bias_gradient(p: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    s = _log_probs(p) + bias
    q = np.exp(s - s.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    counts = np.bincount(np.asarray(y).astype(int), minlength=p.shape[1])
    return float(np.abs(q.sum(axis=0) - counts).max() / len(y))


def fit_bcts(z: np.ndarray, y: np.ndarray, bounds: Tuple[float, float] = T_BOUNDS) -> BCTS:
    """D13: T and the per-grade biases jointly minimising the NLL on calibration.

    T is found by the same bounded search as `fit_temperature`, on the profile NLL:
    at every candidate T the biases are solved from scratch, so the search is over
    one variable and every value it compares is a true minimum over the biases.
    Refuses to return a fit whose biases have not converged: EM's premise is only
    as good as the bias equations are solved.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y).astype(int)

    def profile(log_t: float) -> float:
        p = coral_probs(z, math.exp(log_t))
        return _bias_nll(_log_probs(p), y, fit_bias(p, y))

    t = _search_temperature(profile, bounds)
    p = coral_probs(z, t)
    bias = fit_bias(p, y)
    gap = _bias_gradient(p, y, bias)
    if gap > 1e-8:
        raise ValueError(f"BCTS biases did not converge at T={t:.4f} (mean moment gap "
                         f"{gap:.2e}); the fitted posteriors would not match the grade mix")
    return BCTS(t, bias)


@dataclass(frozen=True)
class EMResult:
    prior: np.ndarray
    iterations: int
    converged: bool


def em_prior(p_src: np.ndarray, pi_src: Sequence[float], tol: float = EM_TOL,
             max_iter: int = EM_MAX_ITER) -> EMResult:
    """Saerens, Latinne & Decaestecker (2002): the target's class prior, unlabelled.

    Start at pi_src; each step re-weights every posterior by pi / pi_src and takes
    the mean as the new prior. Stops when no class moves by `tol` or more, or after
    `max_iter` steps -- whichever comes first (plan s4.2, stage 2).
    """
    p_src = np.asarray(p_src, dtype=np.float64)
    pi_src = np.asarray(pi_src, dtype=np.float64)
    pi = pi_src.copy()
    for it in range(1, max_iter + 1):
        new = prior_correct(p_src, pi, pi_src).mean(axis=0)
        step = float(np.abs(new - pi).max())
        pi = new
        if step < tol:
            return EMResult(pi, it, True)
    return EMResult(pi, max_iter, False)


def predicted_confidence(p: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """p(y-hat | x): the probability of the grade actually predicted (plan s4.3)."""
    yhat = np.asarray(yhat).astype(int)
    return np.asarray(p)[np.arange(len(yhat)), yhat]


def grade_prior(grades: Sequence[int], k: int = NUM_CLASSES) -> np.ndarray:
    """A class distribution from a list of grades -- pi_src is exactly this."""
    counts = np.bincount(np.asarray(grades).astype(int), minlength=k).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("no grades to count")
    return counts / counts.sum()
