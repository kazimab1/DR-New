"""The four signals, the disagreement score and the combined policy (ANALYSIS_PLAN.md s5, s6.1).

All per image and per model: y-hat and the faithfulness deltas belong to one M1
instance, while the evidence grade e belongs to the image.

    d_evidence  |y-hat - e| where M3 can speak (y-hat <= 2), else 0      s5.1
    d_conf      1 - p_cal(y-hat | x), stages 0+1                        s5.2
    d_ood       z of the nearest-grade Mahalanobis distance             s5.3
    d_faith     1 iff the randomisation test determined "unfaithful"    s5.4

    disagreement = d_evidence + r * d_faith,  r in {0, 0.5, 1.5, 2.5}   s5.5
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .faithfulness import MIN_CONTROLS_OK

MAX_EXCLUDABLE_GRADE = 2          # frozen_config.reasoner.reachable_grades, D1
R_GRID = (0.0, 0.5, 1.5, 2.5)     # s5.5: the only four rankings that exist
CONF_PERCENTILE = 90.0            # s6.1: tau_conf
AUC_TIE = 1e-12                   # r's tie rule: equal AUCs go to the smaller r

ACTIONS = ("ACCEPT", "ADJACENT_GRADE_SET", "DEFER")   # level 0, 1, 2; REACQUIRE removed (D10)

# Faithfulness outcomes, per image and model.
FAITHFUL, UNFAITHFUL, UNDETERMINED, NO_LESION = 1, 0, -1, -2


def d_evidence(yhat: np.ndarray, e: np.ndarray, max_excludable: np.ndarray) -> np.ndarray:
    """|y-hat - e| where M3 can contradict y-hat; 0 above what it can reach.

    M3 abstains above grade 2 (abstain_upward), so the literal |y-hat - e| would count
    every M1 grade-3/4 call as a disagreement caused by M3's blind spot, not by the
    evidence (s5.1: the reading that does not favour H1).
    """
    yhat = np.asarray(yhat).astype(int)
    e = np.asarray(e).astype(int)
    return np.where(yhat <= np.asarray(max_excludable).astype(int), np.abs(yhat - e), 0)


def faith_outcome(e_orig: np.ndarray, e_lesion: np.ndarray, e_controls: np.ndarray,
                  min_ok: int = MIN_CONTROLS_OK) -> np.ndarray:
    """FAITHFUL / UNFAITHFUL / UNDETERMINED / NO_LESION for each row.

    Delta = E(original) - E(inpainted). Faithful iff the lesion Delta exceeds every
    successful control's (failed draws are NaN), with at least `min_ok` of them --
    the rule `faithfulness.faithful` states, applied to whole columns.
    """
    e_orig = np.asarray(e_orig, dtype=np.float64)
    e_lesion = np.asarray(e_lesion, dtype=np.float64)
    ctrl = np.asarray(e_controls, dtype=np.float64)
    has_lesion = np.isfinite(e_lesion)
    ok = np.isfinite(ctrl).sum(axis=1)
    delta_lesion = e_orig - e_lesion
    delta_ctrl = e_orig[:, None] - ctrl
    best_ctrl = np.where(np.isfinite(delta_ctrl), delta_ctrl, -np.inf).max(axis=1)
    out = np.full(len(e_orig), NO_LESION, dtype=np.int64)
    decided = has_lesion & (ok >= min_ok)
    out[has_lesion & ~decided] = UNDETERMINED
    out[decided] = np.where(delta_lesion[decided] > best_ctrl[decided], FAITHFUL, UNFAITHFUL)
    return out


def d_faith(outcome: np.ndarray) -> np.ndarray:
    """1 if determined unfaithful; 0 if faithful, no lesion, or undetermined (s5.4)."""
    return (np.asarray(outcome) == UNFAITHFUL).astype(np.int64)


def disagreement(d_ev: np.ndarray, d_fa: np.ndarray, r: float) -> np.ndarray:
    return np.asarray(d_ev, dtype=np.float64) + float(r) * np.asarray(d_fa, dtype=np.float64)


def combined_level(d_ev: np.ndarray, d_fa: np.ndarray, z_ood: np.ndarray, tau_ood: float,
                   e: np.ndarray, d_conf: np.ndarray, tau_conf: float) -> np.ndarray:
    """The docs/03 decision policy as pinned in s6.1: 2 DEFER, 1 ADJACENT, 0 ACCEPT.

    "unobservable non-empty" is true for every image (the same three findings are
    always unobservable), so it drops out.
    """
    d_ev = np.asarray(d_ev)
    defer = ((d_ev >= 2) | (np.asarray(d_fa) == 1) | (np.asarray(z_ood) >= tau_ood)
             | ((np.asarray(e) == 0) & (np.asarray(d_conf) >= tau_conf)))
    adjacent = ~defer & (d_ev == 1)
    return np.where(defer, 2, np.where(adjacent, 1, 0)).astype(np.int64)


def action_table(level: np.ndarray, yhat: np.ndarray, y: np.ndarray) -> Dict[str, dict]:
    """F5: share of cases and accuracy per action.

    For ADJACENT_GRADE_SET "accurate" means the true grade is in {y-hat - 1, y-hat,
    y-hat + 1}; for the other two it is the exact grade.
    """
    level, yhat, y = (np.asarray(a).astype(int) for a in (level, yhat, y))
    out = {}
    for lv, name in enumerate(ACTIONS):
        mask = level == lv
        if lv == 1:
            hit = np.abs(y - yhat) <= 1
        else:
            hit = y == yhat
        out[name] = {"n": int(mask.sum()), "share": float(mask.mean()) if len(mask) else float("nan"),
                     "accuracy": float(hit[mask].mean()) if mask.any() else float("nan")}
    return out
