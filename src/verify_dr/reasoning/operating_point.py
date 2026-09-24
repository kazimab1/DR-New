"""D12: M3's image-level operating point, fitted on calibration (ANALYSIS_PLAN.md addendum A.1).

The frozen rule calls a lesion type present when M2 finds any component of >= 4 px
at probability 0.5 -- an operating point chosen for pixel segmentation (C2), never for
deciding whether an image shows disease. At the rehearsal it found evidence in 75.9%
of grade-0 val images.

The amended rule keeps that condition and adds one: the type's total predicted area
(`area_<type>` in the pass) must reach a per-type minimum. The minimums are chosen on
the calibration split by M3's own QWK against the true grade. M1 is never consulted,
so the choice cannot be steered toward H1.

Everything works from the counts and areas the pass stored, so no image is read
again. M3's ladder itself is untouched: `ladder` is the vectorised form of
`rules.grade`'s count-only branch, and a test checks the two agree on every one of
the 16 presence patterns.
"""

from __future__ import annotations

import itertools
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from verify_dr.data.segmentation import LESION_NAMES
from verify_dr.evaluation.metrics import quadratic_weighted_kappa

AREA_GRID = (4, 16, 64, 256, 1024)      # px; all four at 4 is the frozen rule
FROZEN_AREA_MIN = {name: AREA_GRID[0] for name in LESION_NAMES}
QWK_TIE = 1e-12
RULE_NAMES = np.array(["R1", "R2", "R3", "R3*"])


def ladder(ma: np.ndarray, he: np.ndarray, ex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evidence grade and rule from presence flags -- `rules.grade` without quadrants.

        R3   MA and (haemorrhage or exudate)   -> 2
        R2   MA only                            -> 1
        R3*  haemorrhage or exudate, no MA      -> 2
        R1   nothing                            -> 0
    """
    ma, he, ex = (np.asarray(a, dtype=bool) for a in (ma, he, ex))
    larger = he | ex
    grade = np.where(ma & larger, 2, np.where(ma, 1, np.where(larger, 2, 0)))
    rule = np.where(ma & larger, 2, np.where(ma, 1, np.where(larger, 3, 0)))
    return grade.astype(np.int64), RULE_NAMES[rule]


def presence(images: pd.DataFrame, area_min: Mapping[str, int]) -> Dict[str, np.ndarray]:
    """Per type: at least one component of >= 4 px AND total area >= its minimum."""
    return {name: (images[f"n_{name}"].to_numpy() > 0)
            & (images[f"area_{name}"].to_numpy() >= int(area_min[name]))
            for name in LESION_NAMES}


def evidence(images: pd.DataFrame, area_min: Mapping[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """The evidence grade and rule under an operating point."""
    present = presence(images, area_min)
    return ladder(present["microaneurysm"], present["haemorrhage"],
                  present["hard_exudate"] | present["soft_exudate"])


def fit_area_minimums(images: pd.DataFrame, y: np.ndarray,
                      grid: Sequence[int] = AREA_GRID) -> dict:
    """The per-type minimums with the best M3 QWK against the true grade.

    QWKs within QWK_TIE are ties; a tie goes to the smallest sum of grid positions
    (closest to the frozen rule), then lexicographically in LESION_NAMES order.
    """
    y = np.asarray(y).astype(int)
    grid = tuple(int(g) for g in grid)
    table = {}
    for positions in itertools.product(range(len(grid)), repeat=len(LESION_NAMES)):
        e, _ = evidence(images, {name: grid[i] for name, i in zip(LESION_NAMES, positions)})
        table[positions] = quadratic_weighted_kappa(y, e)
    top = max(table.values())
    chosen = min((p for p, q in table.items() if q >= top - QWK_TIE),
                 key=lambda p: (sum(p), p))
    area_min = {name: grid[i] for name, i in zip(LESION_NAMES, chosen)}
    frozen_e, _ = evidence(images, FROZEN_AREA_MIN)
    return {"area_min": area_min, "grid": list(grid), "qwk": float(table[chosen]),
            "qwk_frozen": float(quadratic_weighted_kappa(y, frozen_e)),
            "combinations": len(table)}
