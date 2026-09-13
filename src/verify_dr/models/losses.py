"""Losses for M1 (docs/03_model_architecture.md section M1).

    L_M1 = L_ordinal + 0.3 * BCE(rDR) + 0.3 * BCE(VTDR)

Focal weighting on the ordinal thresholds handles EyePACS's ~36:1 grade-0 to
grade-4 imbalance without discarding data. B3 compares plain CE and a non-focal
ordinal head against this.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def ordinal_targets(grades: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """levels[b, k] = 1 if grade_b > k. Shape [B, K-1]."""
    ks = torch.arange(num_classes - 1, device=grades.device)
    return (grades.unsqueeze(1) > ks.unsqueeze(0)).float()


def focal_bce(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 0.0
) -> torch.Tensor:
    """BCE with logits, optionally focal-weighted. gamma=0 is plain BCE."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma <= 0:
        return bce.mean()
    prob = torch.sigmoid(logits)
    p_t = targets * prob + (1 - targets) * (1 - prob)
    return ((1 - p_t).pow(gamma) * bce).mean()


class GradingLoss(nn.Module):
    """head: 'ordinal_focal' | 'ordinal' | 'softmax_ce'."""

    def __init__(
        self,
        head: str = "ordinal_focal",
        focal_gamma: float = 2.0,
        rdr_weight: float = 0.3,
        vtdr_weight: float = 0.3,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.head = head
        self.gamma = focal_gamma if head == "ordinal_focal" else 0.0
        self.rdr_weight = rdr_weight
        self.vtdr_weight = vtdr_weight
        self.num_classes = num_classes

    def forward(
        self, out: Dict[str, torch.Tensor], grades: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if self.head == "softmax_ce":
            main = F.cross_entropy(out["logits"], grades)
        else:
            main = focal_bce(
                out["logits"], ordinal_targets(grades, self.num_classes), self.gamma
            )

        rdr = focal_bce(out["rdr"], (grades >= 2).float())
        vtdr = focal_bce(out["vtdr"], (grades >= 3).float())
        total = main + self.rdr_weight * rdr + self.vtdr_weight * vtdr
        return {"loss": total, "main": main.detach(),
                "rdr": rdr.detach(), "vtdr": vtdr.detach()}
