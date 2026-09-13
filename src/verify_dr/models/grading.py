"""M1 — the image-level grading pathway.

    512x512x3 -> backbone -> GAP -> [B, D]
              -> (optional eye-pair fusion) -> [B, 2D]
              -> dropout -> FC 512 -> ReLU -> [B, 512]
              -> ordinal head (4 thresholds) + rDR head + VTDR head

Specified in docs/03_model_architecture.md section M1. This module must never
import or share weights with the evidence pathway (M2): their independence is the
thesis, and a shared encoder would make them fail together.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES = ("efficientnet_b0", "resnet50")


# ------------------------------------------------------------------- backbone


def build_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Return a feature extractor producing pooled [B, D] and the width D."""
    import torchvision.models as tvm

    if name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        encoder = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        return encoder, 1280

    if name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))
        return encoder, 2048

    raise ValueError(f"unknown backbone {name!r}; expected one of {BACKBONES}")


# --------------------------------------------------------------- ordinal head


class OrdinalHead(nn.Module):
    """CORAL-style ordinal head: unit k predicts P(grade > k).

    One shared weight vector and K-1 biases, with the biases forced to be
    non-increasing by construction:

        b_0 = beta,  b_k = b_{k-1} - softplus(delta_k)  >  strictly decreasing

    Monotonicity therefore holds for *any* parameter values, so the model can
    never emit P(grade>2) > P(grade>1). An independent K-1-logit head would
    produce that regularly, and the class probabilities derived from it would go
    negative.
    """

    def __init__(self, in_features: int, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.weight = nn.Linear(in_features, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))
        # softplus(0.5413) ~= 1.0, so thresholds start about one logit apart.
        self.deltas = nn.Parameter(torch.full((self.num_thresholds - 1,), 0.5413))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.deltas)
        offsets = torch.cat([gaps.new_zeros(1), torch.cumsum(gaps, dim=0)])
        return self.beta - offsets                      # [K-1], decreasing

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Logits for P(grade > k), shape [B, K-1]."""
        return self.weight(features) + self.thresholds()


def cumulative_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Class probabilities from P(grade > k) logits.

        P(0) = 1 - P(>0)
        P(k) = P(>k-1) - P(>k)
        P(K-1) = P(>K-2)

    The clamp is belt and braces: the head's monotonicity already guarantees
    non-negative differences, but a clamp keeps this usable if a future head
    does not.
    """
    cum = torch.sigmoid(logits)                          # [B, K-1]
    upper = torch.cat([cum.new_ones(cum.size(0), 1), cum], dim=1)
    lower = torch.cat([cum, cum.new_zeros(cum.size(0), 1)], dim=1)
    probs = (upper - lower).clamp_min(0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def cumulative_to_grade(logits: torch.Tensor) -> torch.Tensor:
    """Predicted grade = number of thresholds passed."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


# -------------------------------------------------------------------- model


class GradingModel(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        eye_pair_fusion: bool = False,
        num_classes: int = 5,
        head: str = "ordinal_focal",
    ) -> None:
        super().__init__()
        self.encoder, width = build_backbone(backbone, pretrained)
        self.eye_pair_fusion = eye_pair_fusion
        self.head = head
        self.num_classes = num_classes

        neck_in = width * (2 if eye_pair_fusion else 1)
        self.neck = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(neck_in, 512), nn.ReLU(inplace=True)
        )

        if head == "softmax_ce":
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.classifier = OrdinalHead(512, num_classes)

        # rDR = grade >= 2 (referable), VTDR = grade >= 3 (vision-threatening).
        # Redundant with the ordinal head by construction, but training them
        # explicitly sharpens the two boundaries screening actually acts on.
        self.rdr = nn.Linear(512, 1)
        self.vtdr = nn.Linear(512, 1)

    def forward(
        self, image: torch.Tensor, fellow: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        features = self.encoder(image)
        if self.eye_pair_fusion:
            # A patient with one usable eye duplicates it, so the fused vector is
            # always the same width.
            other = self.encoder(fellow) if fellow is not None else features
            features = torch.cat([features, other], dim=1)

        hidden = self.neck(features)
        out = {
            "logits": self.classifier(hidden),
            "rdr": self.rdr(hidden).squeeze(1),
            "vtdr": self.vtdr(hidden).squeeze(1),
        }
        if self.head == "softmax_ce":
            out["probs"] = F.softmax(out["logits"], dim=1)
            out["grade"] = out["logits"].argmax(dim=1)
        else:
            out["probs"] = cumulative_to_probs(out["logits"])
            out["grade"] = cumulative_to_grade(out["logits"])
        return out
