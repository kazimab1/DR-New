"""Loss for M2a (docs/03_model_architecture.md section M2a).

    L = 0.5 * Dice + 0.5 * BCE, per channel, averaged over channels.

Dice alone is unstable when a channel is empty, and empty channels are the norm
here rather than the exception -- many fundus images carry no soft exudates at
all. BCE keeps a gradient in that case; Dice supplies the overlap signal that BCE
alone loses under the extreme foreground/background imbalance of a lesion mask,
where a microaneurysm may be a few hundred pixels in a quarter-million.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over the (sample, channel) pairs that actually contain a lesion.

    Empty pairs are excluded rather than smoothed. Summing an empty target's Dice
    over batch and space makes the denominator the *accumulated* predicted
    probability, so at 512x512 even a confidently-empty prediction -- sigmoid(-10)
    per pixel -- piles up to ~95 against an eps of 1, scoring a correct answer at
    0.99 loss. No value of eps fixes that, because the leak scales with resolution
    and batch size while eps does not.

    Excluding them is also the right reading of the measure: Dice is an overlap
    ratio, and there is no overlap to score when nothing is there. BCE keeps the
    gradient on empty channels, which is the division of labour the 0.5/0.5 split
    exists for.
    """
    probs = torch.sigmoid(logits)
    dims = (2, 3)                                    # per sample, per channel
    intersection = (probs * targets).sum(dims)
    cardinality = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)

    present = targets.sum(dims) > 0
    if not bool(present.any()):
        # Nothing to overlap anywhere in this batch; BCE carries it alone.
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


class SegmentationLoss(nn.Module):
    """0.5 Dice + 0.5 BCE, with optional positive weighting.

    `pos_weight` counteracts the foreground/background imbalance inside BCE. It is
    off by default: the Dice term already supplies overlap pressure, and stacking
    both tends to produce a model that over-segments, which for this project is
    the worse failure -- M2's job is to provide evidence the reasoner can trust,
    and a segmenter that finds lesions everywhere makes disagreement meaningless.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5,
                 pos_weight: float | None = None) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        weight = (torch.full((targets.shape[1], 1, 1), self.pos_weight,
                             device=logits.device, dtype=logits.dtype)
                  if self.pos_weight else None)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
        dice = dice_loss(logits, targets)
        total = self.dice_weight * dice + self.bce_weight * bce
        return {"loss": total, "dice": dice.detach(), "bce": bce.detach()}
