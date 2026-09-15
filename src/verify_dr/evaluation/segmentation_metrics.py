"""Per-lesion Dice and IoU for M2a (experiments C2 and C3).

The reporting decision that matters here is **which images a channel's Dice is
averaged over**. Most fundus images carry no soft exudates, so averaging over
every image folds in a long run of empty-target/empty-prediction pairs that score
1.0 by convention. That inflates the headline number without the model having
segmented anything.

So `dice_present` -- the mean over images where the lesion is actually annotated --
is the figure to report, and `dice_all` is kept beside it to make the gap visible
rather than letting a reader assume the generous reading.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

LESION_NAMES = ("microaneurysm", "haemorrhage", "hard_exudate", "soft_exudate")


def _pairwise(pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-image, per-channel overlap terms. Shapes [N, C]."""
    axes = (2, 3)
    intersection = (pred * truth).sum(axes)
    pred_area = pred.sum(axes)
    truth_area = truth.sum(axes)
    union = pred_area + truth_area - intersection
    return {"intersection": intersection, "pred_area": pred_area,
            "truth_area": truth_area, "union": union}


def segmentation_metrics(
    probs: np.ndarray,
    truth: np.ndarray,
    threshold: float = 0.5,
    names: Sequence[str] = LESION_NAMES,
) -> Dict[str, object]:
    """Dice and IoU per lesion channel.

    `probs` and `truth` are [N, C, H, W]; `truth` is binary. Threshold 0.5 matches
    the structured-facts extraction in docs/03 section M2, so the number reported
    here is the one the reasoner will actually be working from.
    """
    pred = (np.asarray(probs) >= threshold).astype(np.float32)
    truth = (np.asarray(truth) > 0.5).astype(np.float32)
    terms = _pairwise(pred, truth)

    per_lesion: Dict[str, Dict[str, float]] = {}
    for c, name in enumerate(names):
        present = terms["truth_area"][:, c] > 0
        dice = (2 * terms["intersection"][:, c]) / np.maximum(
            terms["pred_area"][:, c] + terms["truth_area"][:, c], 1e-9)
        iou = terms["intersection"][:, c] / np.maximum(terms["union"][:, c], 1e-9)

        # An empty target with an empty prediction is a correct answer, and scores
        # 1 by convention; keep that for the all-image figure but never let it into
        # the present-only one.
        both_empty = (terms["truth_area"][:, c] == 0) & (terms["pred_area"][:, c] == 0)
        dice_all = np.where(both_empty, 1.0, dice)
        iou_all = np.where(both_empty, 1.0, iou)

        per_lesion[name] = {
            "dice_present": float(dice[present].mean()) if present.any() else float("nan"),
            "iou_present": float(iou[present].mean()) if present.any() else float("nan"),
            "dice_all": float(dice_all.mean()),
            "iou_all": float(iou_all.mean()),
            "images_with_lesion": int(present.sum()),
            "images_total": int(len(pred)),
            # Over-segmentation is the failure that matters for this project: a
            # model that finds lesions everywhere makes disagreement meaningless.
            "false_positive_images": int(((terms["truth_area"][:, c] == 0) &
                                          (terms["pred_area"][:, c] > 0)).sum()),
            "mean_pred_px": float(terms["pred_area"][:, c].mean()),
            "mean_truth_px": float(terms["truth_area"][:, c].mean()),
        }

    scored = [v["dice_present"] for v in per_lesion.values() if v["dice_present"] == v["dice_present"]]
    return {
        "per_lesion": per_lesion,
        "mean_dice_present": float(np.mean(scored)) if scored else float("nan"),
        "mean_iou_present": float(np.mean(
            [v["iou_present"] for v in per_lesion.values()
             if v["iou_present"] == v["iou_present"]])) if scored else float("nan"),
        "threshold": threshold,
        "n": int(len(pred)),
    }


def format_report(metrics: Dict[str, object]) -> str:
    """One block, aligned, for the training log and the register."""
    lines = [f"  {'lesion':<16}{'Dice':>8}{'IoU':>8}{'images':>9}{'FP imgs':>9}"
             f"{'pred px':>10}{'true px':>10}"]
    for name, v in metrics["per_lesion"].items():
        dice = "   n/a" if v["dice_present"] != v["dice_present"] else f"{v['dice_present']:.4f}"
        iou = "   n/a" if v["iou_present"] != v["iou_present"] else f"{v['iou_present']:.4f}"
        lines.append(
            f"  {name:<16}{dice:>8}{iou:>8}"
            f"{v['images_with_lesion']:>5}/{v['images_total']:<3}{v['false_positive_images']:>9}"
            f"{v['mean_pred_px']:>10.0f}{v['mean_truth_px']:>10.0f}")
    lines.append(f"  {'mean':<16}{metrics['mean_dice_present']:>8.4f}"
                 f"{metrics['mean_iou_present']:>8.4f}")
    lines.append("  Dice and IoU are over images where the lesion is annotated.")
    return "\n".join(lines)
