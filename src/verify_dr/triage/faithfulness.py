"""M4b — counterfactual faithfulness, as pinned by ANALYSIS_PLAN.md section 5.4.

Does M1's grade depend on the lesions M2 found, or would removing *any* equal area
of retina move it as much? One inpainted copy with the lesions removed is compared
against K = 19 copies with the same shapes removed from random places. A network
that reacts to inpainting but not to *what* was inpainted makes the lesion removal
just one more draw, which then comes out on top with probability <= 1/20. That
replaces the unspecified margin in docs/03 with a test.

Everything here is deterministic given the image ID: the controls are seeded by it,
so the locked pass and any re-run draw exactly the same regions.

**Equal area is the frozen requirement** (`frozen_config.counterfactual.control:
random_regions_equal_area`). The shapes moved are the connected regions of the
*dilated union* of the lesions, not each channel's components separately: two
channels' components can overlap (a microaneurysm inside a haemorrhage), and
moving them independently would count the overlap twice. When nothing overlaps
the two readings are identical.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

CONTROLS = 19            # K; a lesion-blind network passes with p <= 1/(K + 1)
MIN_CONTROLS_OK = 15     # below this the test is undetermined, not failed
DILATE_PX = 3
INPAINT_RADIUS = 5       # frozen_config.counterfactual.inpaint.radius
MAX_ATTEMPTS = 200       # per region, per draw
FOV_THRESHOLD = 10       # a pixel is retina when its brightest channel exceeds this


def dilate(mask: np.ndarray, px: int = DILATE_PX) -> np.ndarray:
    """Grow a mask by `px` in every direction with a disc, not a square."""
    import cv2

    if px <= 0 or not mask.any():
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def field_of_view(rgb: np.ndarray, threshold: int = FOV_THRESHOLD) -> np.ndarray:
    """The retina: every pixel brighter than the padding around it."""
    return np.asarray(rgb).max(axis=2) > threshold


@dataclass(frozen=True)
class Region:
    """One connected shape, as offsets from its bounding box's top-left corner."""
    ys: np.ndarray
    xs: np.ndarray
    height: int
    width: int

    @property
    def area(self) -> int:
        return int(len(self.ys))


def regions(mask: np.ndarray) -> List[Region]:
    """Connected regions (8-connectivity) of a mask, largest first.

    Largest first because big shapes are the hard ones to place; trying them
    while the retina is still empty fails fewer draws for no change in meaning.
    """
    import cv2

    binary = np.asarray(mask).astype(np.uint8)
    if not binary.any():
        return []
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    out: List[Region] = []
    for label in range(1, n):
        ys, xs = np.nonzero(labels == label)
        y0, x0 = int(ys.min()), int(xs.min())
        out.append(Region(ys - y0, xs - x0, int(ys.max() - y0 + 1), int(xs.max() - x0 + 1)))
    out.sort(key=lambda r: (-r.area, r.height, r.width))
    return out


def control_seed(image_id: str, draw: int) -> List[int]:
    """Stable across processes, machines and Python's hash randomisation."""
    return [zlib.crc32(image_id.encode("utf-8")), int(draw)]


def place_control(shapes: Sequence[Region], lesions: np.ndarray, fov: np.ndarray,
                  image_id: str, draw: int,
                  max_attempts: int = MAX_ATTEMPTS) -> Optional[np.ndarray]:
    """One control mask: every shape moved, intact, to a random place.

    Each shape must land fully inside the field of view and overlap neither the
    real lesions nor a shape already placed in this draw. Returns None when some
    shape cannot be placed in `max_attempts` tries -- a failed draw, never a
    smaller control, because a smaller control is not equal area.
    """
    rng = np.random.default_rng(control_seed(image_id, draw))
    height, width = lesions.shape
    occupied = lesions.astype(bool).copy()
    control = np.zeros_like(occupied)
    for shape in shapes:
        if shape.height > height or shape.width > width:
            return None
        placed = False
        for _ in range(max_attempts):
            ty = int(rng.integers(0, height - shape.height + 1))
            tx = int(rng.integers(0, width - shape.width + 1))
            ys, xs = shape.ys + ty, shape.xs + tx
            if fov[ys, xs].all() and not occupied[ys, xs].any():
                occupied[ys, xs] = True
                control[ys, xs] = True
                placed = True
                break
        if not placed:
            return None
    return control


def inpaint(rgb: np.ndarray, mask: np.ndarray, radius: int = INPAINT_RADIUS) -> np.ndarray:
    """OpenCV Telea inpainting (frozen_config.counterfactual.inpaint)."""
    import cv2

    if not mask.any():
        return np.asarray(rgb).copy()
    return cv2.inpaint(np.ascontiguousarray(rgb), mask.astype(np.uint8) * 255,
                       radius, cv2.INPAINT_TELEA)


@dataclass
class Variants:
    """The lesion-removed copy and its controls for one image."""
    lesion: np.ndarray                        # [H, W, 3] uint8
    controls: List[Optional[np.ndarray]]      # K entries; None where the draw failed
    region_px: int                            # area removed, identical in every copy

    @property
    def controls_ok(self) -> int:
        return sum(c is not None for c in self.controls)


def make_variants(rgb: np.ndarray, lesion_mask: np.ndarray, image_id: str,
                  k: int = CONTROLS, dilate_px: int = DILATE_PX) -> Optional[Variants]:
    """Everything faithfulness needs for one image, or None if it has no lesions.

    `lesion_mask` is `lesion_region_mask(probs)`: the components M3 counted.
    """
    if not lesion_mask.any():
        return None
    removed = dilate(lesion_mask, dilate_px)
    fov = field_of_view(rgb)
    shapes = regions(removed)
    controls: List[Optional[np.ndarray]] = []
    for draw in range(k):
        where = place_control(shapes, removed, fov, image_id, draw)
        controls.append(None if where is None else inpaint(rgb, where))
    return Variants(lesion=inpaint(rgb, removed), controls=controls,
                    region_px=int(removed.sum()))


def expected_grade(cumulative_logits: np.ndarray) -> np.ndarray:
    """E[grade] = sum over k of P(grade > k), from raw CORAL logits.

    Raw, not calibrated: faithfulness asks what the network computes, and a
    temperature fitted later must not change which images count as faithful.
    """
    z = np.asarray(cumulative_logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-z))).sum(axis=-1)


def faithful(delta_lesion: float, delta_controls: Sequence[float],
             min_ok: int = MIN_CONTROLS_OK) -> Optional[bool]:
    """True/False, or None when too few controls succeeded to decide.

    Faithful iff removing the lesions moves the expected grade down by more than
    every successful control did. Failed draws arrive as NaN and are ignored.
    """
    ctrl = np.asarray(delta_controls, dtype=np.float64)
    ctrl = ctrl[np.isfinite(ctrl)]
    if len(ctrl) < min_ok:
        return None
    return bool(delta_lesion > ctrl.max())
