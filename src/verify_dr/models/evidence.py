"""M2 — the evidence pathway.

    512x512x3
      -> ResNet18 encoder (ImageNet init), skips at 256, 128, 64, 32
      -> M2a: UNet decoder -> 1x1 conv -> 4 channels -> sigmoid   [B, 4, 512, 512]
      -> M2b: GAP -> FC 128 -> FC 4 -> sigmoid                    [B, 4] in [0,1]

Specified in docs/03_model_architecture.md section M2.

**This module must never import from, or share weights with, `grading.py`.** The
independence of M1 and M2 is the thesis (CLAUDE.md rule 4): a shared encoder would
make the two pathways fail together and disagreement would carry no information.

M2a and M2b *do* share their encoder with each other, which the spec calls for and
rule 4 permits -- the rule is about M1 versus M2, not about the two M2 heads.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LESION_CHANNELS = ("microaneurysm", "haemorrhage", "hard_exudate", "soft_exudate")
# Mask directory names as build_cache.py writes them, in channel order.
MASK_DIRS = ("MA", "HE", "EX", "SE")


# ------------------------------------------------------------------- encoder


class EvidenceEncoder(nn.Module):
    """ResNet18 trunk that also hands back the skips a UNet decoder needs.

    At 512 px input the stages land at 256, 128, 64, 32 and a 16 px bottleneck,
    which is exactly the ladder the spec asks for.
    """

    #: (channels, spatial size at 512 px input) for each skip, shallow to deep
    SKIPS: Tuple[Tuple[int, int], ...] = ((64, 256), (64, 128), (128, 64), (256, 32))
    BOTTLENECK = 512

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm

        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet18(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2  -> 256
        self.pool = net.maxpool                                    # /4  -> 128
        self.layer1 = net.layer1                                   #     -> 128
        self.layer2 = net.layer2                                   # /8  -> 64
        self.layer3 = net.layer3                                   # /16 -> 32
        self.layer4 = net.layer4                                   # /32 -> 16

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s0 = self.stem(x)                 # [B, 64, 256, 256]
        s1 = self.layer1(self.pool(s0))   # [B, 64, 128, 128]
        s2 = self.layer2(s1)              # [B, 128, 64, 64]
        s3 = self.layer3(s2)              # [B, 256, 32, 32]
        bottleneck = self.layer4(s3)      # [B, 512, 16, 16]
        return bottleneck, [s0, s1, s2, s3]


# --------------------------------------------------------------- M2b geometry


class GeometryHead(nn.Module):
    """GAP -> FC 128 -> FC 4, squashed to [0, 1].

    The sigmoid is deliberate. Centres are normalised image coordinates, so they
    cannot leave [0, 1]; letting a linear head predict outside it would waste
    capacity learning a constraint the parameterisation can simply enforce -- the
    same reasoning as the ordinal head's monotonicity in M1.
    """

    def __init__(self, in_features: int = EvidenceEncoder.BOTTLENECK,
                 hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(bottleneck, 1)
        return torch.sigmoid(self.net(pooled))      # [B, 4] = od_x, od_y, fx, fy


class GeometryModel(nn.Module):
    """M2b on its own, for C1. Emits [od_x, od_y, fovea_x, fovea_y] in [0, 1]."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = EvidenceEncoder(pretrained)
        self.head = GeometryHead(dropout=dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encoder(image)
        return self.head(bottleneck)


# ---------------------------------------------------------- M2a segmentation


class DecoderBlock(nn.Module):
    """Upsample, concatenate the skip, then two 3x3 convs.

    Bilinear upsampling rather than a transposed convolution: transposed convs
    leave checkerboard artefacts, and a microaneurysm is 10-20 px, so an artefact
    at that scale is indistinguishable from the thing being segmented.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LesionSegmenter(nn.Module):
    """M2a: encoder + UNet decoder + 1x1 conv to four lesion channels.

    Returns raw logits. The loss applies its own sigmoid, and keeping logits lets
    `0.5*Dice + 0.5*BCE` use the numerically stable BCE-with-logits.
    """

    def __init__(self, pretrained: bool = True, num_channels: int = 4,
                 decoder_width: Tuple[int, ...] = (256, 128, 64, 32, 16)) -> None:
        super().__init__()
        self.encoder = EvidenceEncoder(pretrained)
        skips = [c for c, _ in EvidenceEncoder.SKIPS][::-1]     # 256, 128, 64, 64
        widths = decoder_width

        self.up3 = DecoderBlock(EvidenceEncoder.BOTTLENECK, skips[0], widths[0])  # 16->32
        self.up2 = DecoderBlock(widths[0], skips[1], widths[1])                   # 32->64
        self.up1 = DecoderBlock(widths[1], skips[2], widths[2])                   # 64->128
        self.up0 = DecoderBlock(widths[2], skips[3], widths[3])                   # 128->256
        self.up_full = DecoderBlock(widths[3], 0, widths[4])                      # 256->512
        self.classifier = nn.Conv2d(widths[4], num_channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        bottleneck, (s0, s1, s2, s3) = self.encoder(image)
        x = self.up3(bottleneck, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        x = self.up_full(x, None)
        return self.classifier(x)                   # [B, 4, 512, 512] logits


class EvidenceModel(nn.Module):
    """Both heads on one encoder, as the spec describes M2 at inference.

    Training is split across two scripts because the supervision is disjoint --
    DDR has masks and no centres, IDRiD Part C has centres and no masks -- so
    `train_geometry.py` fits the encoder plus M2b, and `train_evidence.py` then
    loads that encoder and fits M2a on top of it.
    """

    def __init__(self, pretrained: bool = True, num_channels: int = 4) -> None:
        super().__init__()
        self.segmenter = LesionSegmenter(pretrained, num_channels)
        self.geometry = GeometryHead()

    @property
    def encoder(self) -> EvidenceEncoder:
        return self.segmenter.encoder

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        bottleneck, (s0, s1, s2, s3) = self.encoder(image)
        x = self.segmenter.up3(bottleneck, s3)
        x = self.segmenter.up2(x, s2)
        x = self.segmenter.up1(x, s1)
        x = self.segmenter.up0(x, s0)
        x = self.segmenter.up_full(x, None)
        return {
            "lesion_logits": self.segmenter.classifier(x),
            "geometry": self.geometry(bottleneck),
        }
