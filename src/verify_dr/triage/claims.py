"""When a hypothesis counts as supported -- ANALYSIS_PLAN.md s8, one rule for all.

For an effect measured on each of the three seeds:

    (a) direction   every seed's effect is > 0
    (b) precision   every seed's paired bootstrap 95% interval excludes 0
    (c) stability   the mean effect exceeds the seed-to-seed spread (max - min)

Supported only if all three hold; otherwise *not supported*, which the protocol
counts as falsified. Bootstrap: 2,000 resamples over images, seed 42, and ONE set of
resample indices per dataset, shared by every arm and model -- `resample_indices`
regenerates the identical sequence each time it is called, so every comparison is
paired without holding a 2,000 x n matrix in memory.
"""

from __future__ import annotations

from typing import Dict, Iterator, Sequence, Tuple

import numpy as np

RESAMPLES = 2000
SEED = 42


def resample_indices(n: int, count: int = RESAMPLES, seed: int = SEED) -> Iterator[np.ndarray]:
    """The dataset's bootstrap resamples, in the same order on every call."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield rng.integers(0, n, n)


def interval(values: Sequence[float]) -> Tuple[float, float]:
    """Percentile 95% interval over the finite resample values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def claim(effects: Sequence[float], intervals: Sequence[Tuple[float, float]]) -> Dict[str, object]:
    """The s8 rule for one hypothesis on one dataset: effects and intervals per seed."""
    e = np.asarray(effects, dtype=np.float64)
    if len(e) != len(intervals) or len(e) == 0:
        raise ValueError("one effect and one interval per seed")
    direction = bool(np.all(e > 0))
    precision = bool(all(lo > 0 or hi < 0 for lo, hi in intervals))
    mean, spread = float(e.mean()), float(e.max() - e.min())
    stability = bool(mean > spread)
    return {"effects": e.tolist(), "intervals": [list(map(float, iv)) for iv in intervals],
            "mean": mean, "spread": spread, "direction": direction, "precision": precision,
            "stability": stability, "supported": direction and precision and stability}


def on_every_dataset(verdicts: Dict[str, Dict[str, object]]) -> bool:
    """H1', H2 and H3 must hold on APTOS *and* Messidor-2."""
    return bool(verdicts) and all(v["supported"] for v in verdicts.values())
