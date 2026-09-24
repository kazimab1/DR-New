"""M4c's d_ood, as ANALYSIS_PLAN.md s5.3 pins "z_score(embedding, train_manifold)".

    embedding   M1's 512-d `neck` output, the layer feeding every head
    manifold    5,000 of that model's own training images, drawn uniformly, seed 0;
                one Gaussian per grade with a single shared covariance, shrunk by
                Ledoit-Wolf (Lee et al., 2018)
    distance    the smallest Mahalanobis distance to any grade's mean
    z           standardised by the mean and SD of the same distance on the
                calibration split
    tau_ood     the 95th percentile of calibration z: 5% flagged in-domain by construction

Only ranks and a threshold are ever used downstream, so whether "distance" is the
quadratic form or its square root changes no outcome. It is the square root here,
the textbook definition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

REFERENCE_SIZE = 5000
REFERENCE_SEED = 0
OOD_PERCENTILE = 95.0
NUM_CLASSES = 5


def ledoit_wolf(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Shrunk covariance of rows that are ALREADY centred, and the shrinkage used.

    The Ledoit-Wolf (2004) estimator, written out as scikit-learn's `ledoit_wolf(x,
    assume_centered=True)` computes it (tests check the two agree), so the analysis
    does not depend on which scikit-learn a Kaggle image happens to ship.
    """
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    emp_cov = x.T @ x / n
    x2 = x ** 2
    trace_terms = x2.sum(axis=0) / n
    mu = trace_terms.sum() / p
    delta_ = (emp_cov ** 2).sum()
    beta_ = (x2.T @ x2).sum()
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * trace_terms.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else float(beta / delta)
    shrunk = (1.0 - shrinkage) * emp_cov
    shrunk[np.diag_indices(p)] += shrinkage * mu
    return shrunk, shrinkage


@dataclass(frozen=True)
class ClassGaussian:
    """Lee et al.'s generative classifier on the embedding: tied covariance."""
    means: np.ndarray            # [classes, d]
    precision: np.ndarray        # [d, d]
    shrinkage: float
    class_counts: np.ndarray     # [classes]


def fit_class_gaussian(emb: np.ndarray, y: np.ndarray, k: int = NUM_CLASSES) -> ClassGaussian:
    """Per-grade means; one covariance of every image around its own grade's mean."""
    emb = np.asarray(emb, dtype=np.float64)
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=k)
    if len(counts) > k or (counts == 0).any():
        raise ValueError(f"every grade needs reference images; counts per grade: {counts.tolist()}")
    means = np.stack([emb[y == c].mean(axis=0) for c in range(k)])
    cov, shrinkage = ledoit_wolf(emb - means[y])
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as err:
        raise ValueError("the shrunk covariance is singular; the reference sample is "
                         "degenerate") from err
    inv_chol = np.linalg.solve(chol, np.eye(len(cov)))
    precision = inv_chol.T @ inv_chol
    return ClassGaussian(means, (precision + precision.T) / 2.0, shrinkage, counts)


def min_mahalanobis(emb: np.ndarray, g: ClassGaussian) -> np.ndarray:
    """Distance from each embedding to the nearest grade mean, in the shared metric."""
    e = np.asarray(emb, dtype=np.float64)
    ep = e @ g.precision
    own = (ep * e).sum(axis=1)                                   # e' P e
    cross = ep @ g.means.T                                       # e' P mu_c
    centre = ((g.means @ g.precision) * g.means).sum(axis=1)     # mu_c' P mu_c
    q = own[:, None] - 2.0 * cross + centre[None, :]
    return np.sqrt(np.clip(q.min(axis=1), 0.0, None))


@dataclass(frozen=True)
class OODScale:
    mean: float
    sd: float
    tau: float


def fit_scale(distance_cal: np.ndarray, percentile: float = OOD_PERCENTILE) -> OODScale:
    """Standardise on calibration, then take the 95th percentile of calibration z."""
    d = np.asarray(distance_cal, dtype=np.float64)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if not sd > 0:
        raise ValueError("calibration distances have no spread")
    tau = float(np.percentile((d - mean) / sd, percentile))
    return OODScale(mean, sd, tau)


def z_score(distance: np.ndarray, scale: OODScale) -> np.ndarray:
    return (np.asarray(distance, dtype=np.float64) - scale.mean) / scale.sd


# ------------------------------------------------------------------ persistence


def gaussian_digest(g: ClassGaussian) -> str:
    """sha256 of the statistics themselves, not of a file holding them.

    An .npz is a zip stamped with the time it was written, so the same numbers saved
    twice hash differently. Hashing dtype, shape and bytes of each array instead
    means a re-run that reproduces the statistics reproduces the digest.
    """
    h = hashlib.sha256()
    for name, arr in (("means", g.means), ("precision", g.precision),
                      ("shrinkage", np.array(g.shrinkage, dtype=np.float64)),
                      ("class_counts", np.asarray(g.class_counts, dtype=np.int64))):
        arr = np.ascontiguousarray(arr)
        h.update(f"{name}|{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def save_gaussian(g: ClassGaussian, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:               # a handle: np.savez would append .npz
        np.savez(fh, means=g.means, precision=g.precision,
                 shrinkage=np.array(g.shrinkage, dtype=np.float64),
                 class_counts=np.asarray(g.class_counts, dtype=np.int64))
    return gaussian_digest(g)


def load_gaussian(path: Path, digest: str) -> ClassGaussian:
    """Refuses statistics whose digest is not the one the committed parameters recorded."""
    with np.load(Path(path)) as z:
        g = ClassGaussian(z["means"], z["precision"], float(z["shrinkage"]),
                          z["class_counts"])
    actual = gaussian_digest(g)
    if actual != digest:
        raise ValueError(f"{path}: digest {actual[:12]}... is not the committed "
                         f"{digest[:12]}... -- these are not the fitted statistics")
    return g
