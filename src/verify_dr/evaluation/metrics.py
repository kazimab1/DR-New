"""Metrics for the grading pathway.

Implemented directly rather than pulled from sklearn so the definitions are
explicit and auditable -- QWK in particular has several variants in circulation,
and the thesis has to state which one it reports.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

NUM_CLASSES = 5


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> float:
    """Cohen's kappa with quadratic weights: distant errors cost quadratically more.

    Chance agreement uses the outer product of the observed marginals, which is
    the standard DR-literature definition (and the Kaggle metric).
    """
    observed = confusion_matrix(y_true, y_pred, k).astype(float)
    total = observed.sum()
    if total == 0:
        return float("nan")

    indices = np.arange(k)
    weights = (indices[:, None] - indices[None, :]) ** 2 / (k - 1) ** 2
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total

    denominator = (weights * expected).sum()
    if denominator == 0:                    # every prediction and label identical
        return 1.0
    return float(1.0 - (weights * observed).sum() / denominator)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> float:
    """F1 averaged equally over classes. Unlike accuracy it does not let a model
    ignore grades 3 and 4 and still look good."""
    scores = []
    for c in range(k):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:               # class absent from both: skip, not 0
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> Dict[str, float]:
    """Recall per grade. Grade 1 is the one to watch: microaneurysms only, and a
    model that silently skips the class can still post a respectable QWK."""
    out = {}
    for c in range(k):
        support = float((y_true == c).sum())
        out[str(c)] = float(((y_pred == c) & (y_true == c)).sum() / support) if support else float("nan")
    return out


def per_class_precision(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> Dict[str, float]:
    """Precision per grade. Needed alongside recall: a model that predicts one grade
    for every image has perfect recall on that grade and is worthless."""
    out = {}
    for c in range(k):
        predicted = float((y_pred == c).sum())
        out[str(c)] = float(((y_pred == c) & (y_true == c)).sum() / predicted) if predicted else float("nan")
    return out


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> Dict[str, float]:
    """Harmonic mean of the two, which is what a per-grade verdict should be read on."""
    recall = per_class_recall(y_true, y_pred, k)
    precision = per_class_precision(y_true, y_pred, k)
    out = {}
    for c in map(str, range(k)):
        r, p = recall[c], precision[c]
        if r != r or p != p or r + p == 0:      # nan, or both zero
            out[c] = 0.0 if (r == r or p == p) else float("nan")
        else:
            out[c] = float(2 * r * p / (r + p))
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC, ties averaged. Equivalent to the Mann-Whitney statistic."""
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # average ranks within tied groups
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 15
) -> float:
    """ECE over equal-width confidence bins.

    `confidence` must be the probability of the **predicted** grade, not
    probs.max(). Under the ordinal head the predicted grade is the number of
    thresholds passed, which need not be the argmax of the class distribution --
    using probs.max() would measure the calibration of a prediction the model
    never made.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    if total == 0:
        return float("nan")
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        error += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def brier_score(probs: np.ndarray, y_true: np.ndarray, k: int = NUM_CLASSES) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    picked = probs[np.arange(len(y_true)), y_true.astype(int)]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def grading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    rdr_score: Optional[np.ndarray] = None,
    vtdr_score: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Everything docs/04_experiment_register.md asks of the grading pathway."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    probs = np.asarray(probs, dtype=float)

    confidence = probs[np.arange(len(y_pred)), y_pred]
    correct = (y_pred == y_true).astype(float)

    out: Dict[str, object] = {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1(y_true, y_pred),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "ece": expected_calibration_error(confidence, correct),
        "nll": negative_log_likelihood(probs, y_true),
        "brier": brier_score(probs, y_true),
        "per_class_recall": per_class_recall(y_true, y_pred),
        "per_class_precision": per_class_precision(y_true, y_pred),
        "per_class_f1": per_class_f1(y_true, y_pred),
        # How many distinct grades the model actually emits. 1 means it has
        # collapsed onto a single class, which makes every per-class recall
        # unreadable on its own.
        "distinct_predictions": int(len(np.unique(y_pred))),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "support": {str(c): int((y_true == c).sum()) for c in range(NUM_CLASSES)},
        # The ordinal grade is the threshold count, not the argmax. A low value
        # here is not a bug, but it does mean the two disagree often.
        "argmax_agreement": float((probs.argmax(axis=1) == y_pred).mean()),
    }
    if rdr_score is not None:
        out["auroc_rdr"] = auroc(np.asarray(rdr_score), y_true >= 2)
    if vtdr_score is not None:
        out["auroc_vtdr"] = auroc(np.asarray(vtdr_score), y_true >= 3)
    return out


def bootstrap_qwk(
    y_true: np.ndarray, y_pred: np.ndarray, resamples: int = 2000, seed: int = 42
) -> Dict[str, float]:
    """Percentile bootstrap interval over images (docs/02_research_protocol.md)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        values[i] = quadratic_weighted_kappa(y_true[idx], y_pred[idx])
    finite = values[np.isfinite(values)]
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "resamples": int(len(finite)),
    }
