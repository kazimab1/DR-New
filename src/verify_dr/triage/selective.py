"""The coverage-accuracy curve and everything read off it (ANALYSIS_PLAN.md s6.2, s6.3).

Every arm is a score; the most trusted case (lowest score) is accepted first. For
k = 1..n accepted cases, accuracy(k) is the exact-grade accuracy of y-hat among them
and coverage is k/n. **AUC is the mean of accuracy(k) over k.**

**Ties are resolved by expectation, computed exactly.** A block of tied cases is
accepted as if in uniformly random order, so after taking m of a block's n_B cases
the expected number correct is m * c_B / n_B. Replacing each case's correctness by
its block's mean and taking a running sum gives exactly that expectation at every k
-- no sampling, no tie-breaking by anything else. Breaking ties by confidence would
fold the baseline into the arm it is compared with (s6.2).

Scores enter as *dense integer ranks* (0 = most trusted). Two cases tie iff every key
of their score is equal, and a bootstrap resample keeps that structure by indexing
the ranks, so the AUC of a resample needs two bincounts rather than a sort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

COVERAGE_POINTS = (0.80, 0.90)       # frozen_config.triage.coverage_points


def dense_rank(*keys: np.ndarray) -> np.ndarray:
    """0 for the most trusted case; ties share a rank. The first key sorts first.

    A lexicographic score -- the combined arm's (level, d_conf) -- is passed as two
    keys instead of being folded into one float, where a tiny d_conf added to a
    level could round two different cases into a tie.
    """
    arrays = [np.asarray(k, dtype=np.float64) for k in keys]
    if not arrays:
        raise ValueError("no score")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("score keys differ in length")
    if any(np.isnan(a).any() for a in arrays):
        raise ValueError("a score is NaN; every case must be ranked")
    order = np.lexsort(arrays[::-1])
    stacked = np.stack([a[order] for a in arrays], axis=1)
    new_group = np.any(stacked[1:] != stacked[:-1], axis=1)
    group = np.concatenate([[0], np.cumsum(new_group)])
    rank = np.empty(n, dtype=np.int64)
    rank[order] = group
    return rank


def _harmonic(n: int) -> np.ndarray:
    """H[m] = 1 + 1/2 + ... + 1/m, with H[0] = 0."""
    return np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])


def auc_from_groups(counts: np.ndarray, hits: np.ndarray, harmonic: np.ndarray = None) -> float:
    """Coverage-accuracy AUC from per-rank-group counts and correct counts, in trust order.

    Within group g (cases S_{g-1}+1 .. S_g), E[correct among first k] is
    C_{g-1} + (k - S_{g-1}) * b_g with b_g = c_g / n_g, so the group's share of the sum
    of accuracy(k) = E[correct]/k has a closed form in harmonic numbers.
    """
    counts = np.asarray(counts, dtype=np.float64)
    hits = np.asarray(hits, dtype=np.float64)
    keep = counts > 0
    counts, hits = counts[keep], hits[keep]
    n = int(round(counts.sum()))
    if n == 0:
        return float("nan")
    if harmonic is None or len(harmonic) < n + 1:
        harmonic = _harmonic(n)
    ends = np.cumsum(counts).round().astype(np.int64)
    starts = ends - counts.round().astype(np.int64)
    before = np.concatenate([[0.0], np.cumsum(hits)[:-1]])
    rate = hits / counts
    a = before - starts * rate
    total = (counts * rate).sum() + (a * (harmonic[ends] - harmonic[starts])).sum()
    return float(total / n)


def auc(rank: np.ndarray, correct: np.ndarray) -> float:
    rank = np.asarray(rank, dtype=np.int64)
    size = int(rank.max()) + 1 if len(rank) else 0
    counts = np.bincount(rank, minlength=size)
    hits = np.bincount(rank, weights=np.asarray(correct, dtype=np.float64), minlength=size)
    return auc_from_groups(counts, hits)


def accuracy_curve(rank: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """E[accuracy(k)] for k = 1..n -- the curve itself, for figures and spot values."""
    rank = np.asarray(rank, dtype=np.int64)
    correct = np.asarray(correct, dtype=np.float64)
    counts = np.bincount(rank)
    block_mean = np.bincount(rank, weights=correct) / np.maximum(counts, 1)
    order = np.argsort(rank, kind="stable")
    expected = np.cumsum(block_mean[rank[order]])
    return expected / np.arange(1, len(rank) + 1)


def k_for(coverage: float, n: int) -> int:
    """The smallest number of accepted cases whose coverage is at least `coverage`."""
    return max(1, min(n, math.ceil(coverage * n - 1e-9)))


def acceptance_probability(rank: np.ndarray, k: int) -> np.ndarray:
    """P(case i is among the first k accepted), ties taken in random order."""
    rank = np.asarray(rank, dtype=np.int64)
    counts = np.bincount(rank).astype(np.float64)
    ends = np.cumsum(counts)
    starts = ends - counts
    per_group = np.clip((k - starts) / np.maximum(counts, 1), 0.0, 1.0)
    return per_group[rank]


def accuracy_at(rank: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    k = k_for(coverage, len(rank))
    p = acceptance_probability(rank, k)
    return float((p * np.asarray(correct, dtype=np.float64)).sum() / k)


@dataclass(frozen=True)
class Referral:
    """F3 at one coverage: referable-DR decisions among the ACCEPTED cases."""
    coverage: float
    sensitivity: float
    specificity: float
    referable_deferred: float        # share of truly referable cases sent to a human


def referral_at(rank: np.ndarray, yhat: np.ndarray, y: np.ndarray, coverage: float) -> Referral:
    """rDR sensitivity and specificity of y-hat >= 2 among accepted cases, as ratios of
    expected counts; plus how many truly referable cases were deferred (s6.3)."""
    k = k_for(coverage, len(rank))
    p = acceptance_probability(rank, k)
    pred = np.asarray(yhat) >= 2
    true = np.asarray(y) >= 2
    tp, fn = (p * (pred & true)).sum(), (p * (~pred & true)).sum()
    tn, fp = (p * (~pred & ~true)).sum(), (p * (pred & ~true)).sum()
    nan = float("nan")
    return Referral(
        coverage,
        float(tp / (tp + fn)) if tp + fn > 0 else nan,
        float(tn / (tn + fp)) if tn + fp > 0 else nan,
        float(((1.0 - p) * true).sum() / true.sum()) if true.any() else nan,
    )


def summary(rank: np.ndarray, correct: np.ndarray,
            coverages: Sequence[float] = COVERAGE_POINTS) -> Dict[str, float]:
    out = {"auc": auc(rank, correct)}
    for c in coverages:
        out[f"acc@{int(round(c * 100))}"] = accuracy_at(rank, correct, c)
    out["distinct_scores"] = int(np.asarray(rank).max()) + 1
    return out
