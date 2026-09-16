"""Structured facts from M2's predicted masks.

M3 reasons over counts and locations, not pixels, so this is the layer that turns
a [4, H, W] probability stack into something a rule can quote. Everything here is
deterministic.

The quadrant frame is the part that can be unavailable. ETDRS quadrants are
defined about the optic disc, split by the disc-to-fovea axis and its
perpendicular -- so they need M2b's geometry, and C1 failed its gate. When
geometry is absent or untrusted the quadrant counts are None rather than
guessed, and R4 (the 4-2-1 severe criterion) declines to fire. A wrong quadrant
assignment would not make R4 cautious, it would make it confidently wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from verify_dr.data.segmentation import LESION_NAMES

QUADRANTS = ("superotemporal", "inferotemporal", "superonasal", "inferonasal")

#: A lesion smaller than this is treated as noise rather than a finding. A
#: microaneurysm is 10-20 px across at 512 px, so ~4 px of area is well below a
#: real one and well above a single stray pixel.
MIN_LESION_PX = 4


@dataclass
class Facts:
    """What M3 is allowed to reason from."""

    counts: Dict[str, int]                     # connected components per lesion
    areas: Dict[str, int]                      # total predicted pixels per lesion
    quadrant_counts: Optional[Dict[str, Dict[str, int]]] = None
    geometry_available: bool = False
    geometry_reason: str = ""
    image_size: int = 0
    notes: List[str] = field(default_factory=list)

    def present(self, lesion: str) -> bool:
        return self.counts.get(lesion, 0) > 0


def quadrant_of(x: float, y: float, disc: Tuple[float, float],
                fovea: Tuple[float, float]) -> str:
    """Which ETDRS quadrant a point falls in.

    The frame is centred on the optic disc, with one axis along the disc-to-fovea
    direction. That makes the assignment independent of which eye it is: the
    temporal side is whichever side the fovea is on, so no laterality flag is
    needed and a mirrored image gives the mirrored -- correct -- answer.
    """
    ax, ay = fovea[0] - disc[0], fovea[1] - disc[1]
    norm = math.hypot(ax, ay)
    if norm == 0:
        raise ValueError("disc and fovea coincide; no axis can be defined")
    ax, ay = ax / norm, ay / norm

    px, py = x - disc[0], y - disc[1]
    along = px * ax + py * ay          # + toward the fovea, i.e. temporal
    side = "temporal" if along >= 0 else "nasal"

    # Superior/inferior is anatomically fixed -- up is up in either eye -- while
    # temporal/nasal follows the fovea. So the perpendicular must be pinned to
    # point DOWN the image, not merely rotated with the axis: (-ay, ax) flips
    # its vertical sense when the axis flips, which labelled a superior lesion
    # "infero" in a left eye and "supero" in a right one. The same laterality
    # trap C1 fell into, one layer up.
    perp_x, perp_y = -ay, ax
    if perp_y < 0 or (perp_y == 0 and perp_x < 0):
        perp_x, perp_y = ay, -ax

    across = px * perp_x + py * perp_y
    half = "infero" if across >= 0 else "supero"
    return f"{half}{side}"


def _components(mask: np.ndarray, min_px: int) -> Tuple[int, List[Tuple[float, float]]]:
    """Count of discrete lesions and their centroids."""
    import cv2

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0, []
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    centroids: List[Tuple[float, float]] = []
    for label in range(1, n):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < min_px:
            continue
        centroids.append((float(xs.mean()), float(ys.mean())))
    return len(centroids), centroids


def extract_facts(
    probs: np.ndarray,
    threshold: float = 0.5,
    geometry: Optional[Sequence[float]] = None,
    geometry_trusted: bool = False,
    geometry_reason: str = "",
    names: Sequence[str] = LESION_NAMES,
    min_lesion_px: int = MIN_LESION_PX,
) -> Facts:
    """Facts for one image from a [C, H, W] probability stack.

    `geometry` is (od_x, od_y, fovea_x, fovea_y) in pixels. `geometry_trusted`
    must be set deliberately by the caller: C1 failed its 0.5 DD gate, so
    quadrant counts are only computed when something upstream has decided the
    frame is good enough to quote.
    """
    probs = np.asarray(probs)
    if probs.ndim != 3:
        raise ValueError(f"expected [C, H, W], got {probs.shape}")
    pred = probs >= threshold
    size = int(probs.shape[-1])

    counts: Dict[str, int] = {}
    areas: Dict[str, int] = {}
    centroids: Dict[str, List[Tuple[float, float]]] = {}
    for i, name in enumerate(names):
        n, cents = _components(pred[i], min_lesion_px)
        counts[name] = n
        areas[name] = int(pred[i].sum())
        centroids[name] = cents

    notes: List[str] = []
    quadrant_counts: Optional[Dict[str, Dict[str, int]]] = None

    if geometry is not None and geometry_trusted:
        disc = (float(geometry[0]), float(geometry[1]))
        fovea = (float(geometry[2]), float(geometry[3]))
        try:
            quadrant_counts = {}
            for name in names:
                per = {q: 0 for q in QUADRANTS}
                for (x, y) in centroids[name]:
                    per[quadrant_of(x, y, disc, fovea)] += 1
                quadrant_counts[name] = per
        except ValueError as exc:
            quadrant_counts = None
            notes.append(f"quadrants unavailable: {exc}")
    elif geometry is None:
        notes.append("no geometry supplied; quadrant rules cannot be evaluated")
    else:
        notes.append(
            "geometry supplied but not trusted; quadrant rules decline to fire"
            + (f" ({geometry_reason})" if geometry_reason else ""))

    return Facts(
        counts=counts, areas=areas, quadrant_counts=quadrant_counts,
        geometry_available=quadrant_counts is not None,
        geometry_reason=geometry_reason, image_size=size, notes=notes,
    )
