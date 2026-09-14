"""Metrics for M2b, the optic-disc / fovea regressor (experiment C1).

C1's gate is stated in disc diameters, not pixels, because that is the unit the
downstream reasoning actually cares about: quadrant assignment is defined relative
to the disc, and a fixed pixel tolerance would mean different things on a wide-field
image and a 30-degree one.

IDRiD publishes centre coordinates but no disc diameter, so it is derived from the
one measurement available on every image -- the disc-to-fovea distance.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# The fovea sits about 2.5 disc diameters temporal to the disc centre. This is the
# standard clinical relation and it is what makes a disc diameter recoverable here
# without an optic-disc mask. It is an approximation: it assumes a roughly normal
# posterior pole, and it inherits any error in the published centres. Stated as a
# named constant so the thesis can quote and defend the number rather than have it
# buried in an expression.
DISC_TO_FOVEA_IN_DIAMETERS = 2.5

#: C1 passes when mean landmark error is below this, in disc diameters.
C1_GATE_DD = 0.5


def disc_diameters(truth: np.ndarray) -> np.ndarray:
    """Per-image disc diameter, in the same units as `truth`.

    Derived per image rather than as one global constant so that magnification
    differences between cameras are absorbed: a wider field puts both landmarks
    closer together in image coordinates, and the disc shrinks with them.
    """
    od, fovea = truth[:, :2], truth[:, 2:]
    separation = np.linalg.norm(od - fovea, axis=1)
    return separation / DISC_TO_FOVEA_IN_DIAMETERS


def geometry_metrics(
    pred: np.ndarray, truth: np.ndarray, image_size: int = 512
) -> Dict[str, object]:
    """Landmark error for C1.

    `pred` and `truth` are [N, 4] as (od_x, od_y, fovea_x, fovea_y), normalised to
    [0, 1]. Errors are reported in pixels for intuition and in disc diameters for
    the gate.
    """
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)

    od_err = np.linalg.norm(pred[:, :2] - truth[:, :2], axis=1)
    fov_err = np.linalg.norm(pred[:, 2:] - truth[:, 2:], axis=1)

    dd = disc_diameters(truth)
    # A degenerate image -- both centres marked at the same point -- would divide by
    # zero and poison the mean. Drop it from the DD figures and say how many.
    usable = dd > 1e-6
    od_dd = od_err[usable] / dd[usable]
    fov_dd = fov_err[usable] / dd[usable]
    mean_dd = float(np.mean(np.concatenate([od_dd, fov_dd]))) if usable.any() else float("nan")

    out: Dict[str, object] = {
        "n": int(len(pred)),
        "od_error_px": float(od_err.mean() * image_size),
        "fovea_error_px": float(fov_err.mean() * image_size),
        "od_error_px_median": float(np.median(od_err) * image_size),
        "fovea_error_px_median": float(np.median(fov_err) * image_size),
        "od_error_dd": float(od_dd.mean()) if usable.any() else float("nan"),
        "fovea_error_dd": float(fov_dd.mean()) if usable.any() else float("nan"),
        # The C1 gate: mean over both landmarks.
        "mean_error_dd": mean_dd,
        "within_half_dd": float(np.mean(np.concatenate([od_dd, fov_dd]) < C1_GATE_DD))
        if usable.any() else float("nan"),
        "degenerate": int((~usable).sum()),
        "mean_disc_diameter_px": float(dd[usable].mean() * image_size) if usable.any() else float("nan"),
        "c1_pass": bool(mean_dd < C1_GATE_DD) if mean_dd == mean_dd else False,
    }
    return out


def baseline_prediction(truth: np.ndarray) -> np.ndarray:
    """Predict the training-set mean landmark for every image.

    The control C1 has to beat. Fundus framing is stereotyped -- the disc is
    usually near one side at mid-height -- so a constant predictor scores better
    than intuition suggests, and a model that fails to beat it has learned the
    average layout rather than where the landmarks are in *this* image.
    """
    return np.repeat(truth.mean(axis=0, keepdims=True), len(truth), axis=0)
