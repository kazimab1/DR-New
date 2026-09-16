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
# Names mirror build_cache.py's LESION_CHANNELS; only the count matters here,
# but keeping one vocabulary stops the two files drifting apart.
MASK_DIRS = ("microaneurysm", "haemorrhage", "hard_exudate", "soft_exudate")


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


class GeometryHeatmapModel(nn.Module):
    """M2b as heatmap localisation rather than coordinate regression.

    Why this exists: C1's first run failed its 0.5 DD gate at 0.686, and the
    per-image dump showed the cause is not imprecision. 10 of 83 validation
    images placed the optic disc on the *wrong side of the fovea* -- median error
    3.875 DD against 0.333 DD elsewhere -- and those eight worst images carried
    47% of the total error.

    That is a representational limit, not a capacity one. The disc sits nasal to
    the macula, so its x position is **bimodal**: left for one eye, right for the
    other. A regressed coordinate must emit one number, so it has to commit to a
    mode and is simply wrong when it commits to the wrong one. A heatmap can hold
    both peaks and let argmax pick the stronger.

    **argmax, not soft-argmax.** A global soft-argmax over a two-peaked heatmap
    returns the weighted mean -- a point between the two discs, worse than either.
    So the peak is taken by argmax and refined by a soft-argmax over a small
    window around it, which buys sub-pixel accuracy without averaging across
    modes.

    The encoder is the same EvidenceEncoder under the same attribute name, so a
    checkpoint from this model still loads into C2 via --encoder-from.
    """

    #: Heatmap side at 512 px input. /4 keeps a microaneurysm-scale grid while
    #: staying cheap; the disc is ~75 px, so 128 is ample for it.
    STRIDE = 4

    def __init__(self, pretrained: bool = True, sigma: float = 2.0,
                 refine_window: int = 5) -> None:
        super().__init__()
        self.encoder = EvidenceEncoder(pretrained)
        self.sigma = sigma
        self.refine_window = refine_window
        skips = EvidenceEncoder.SKIPS
        self.up3 = DecoderBlock(EvidenceEncoder.BOTTLENECK, skips[3][0], 256)  # 16->32
        self.up2 = DecoderBlock(256, skips[2][0], 128)                          # 32->64
        self.up1 = DecoderBlock(128, skips[1][0], 64)                           # 64->128
        # 2 channels: optic disc, fovea. Order matches the target vector's
        # (od_x, od_y, fovea_x, fovea_y) pairing.
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Heatmap logits, [B, 2, H/4, W/4]."""
        bottleneck, skips = self.encoder(image)
        x = self.up3(bottleneck, skips[3])
        x = self.up2(x, skips[2])
        x = self.up1(x, skips[1])
        return self.out(x)

    def coordinates(self, logits: torch.Tensor) -> torch.Tensor:
        """Peak per channel as [od_x, od_y, fovea_x, fovea_y] in [0, 1].

        Taken by argmax so a second mode cannot drag the estimate toward it, then
        refined by a soft-argmax inside `refine_window` around that peak.
        """
        b, c, h, w = logits.shape
        flat = logits.reshape(b, c, -1)
        peak = flat.argmax(dim=-1)
        py, px = peak // w, peak % w

        half = self.refine_window // 2
        prob = torch.softmax(flat, dim=-1).reshape(b, c, h, w)
        ys = torch.arange(h, device=logits.device, dtype=logits.dtype)
        xs = torch.arange(w, device=logits.device, dtype=logits.dtype)

        out = torch.zeros(b, c * 2, device=logits.device, dtype=logits.dtype)
        for i in range(b):
            for j in range(c):
                y0, y1 = int(max(0, py[i, j] - half)), int(min(h, py[i, j] + half + 1))
                x0, x1 = int(max(0, px[i, j] - half)), int(min(w, px[i, j] + half + 1))
                patch = prob[i, j, y0:y1, x0:x1]
                total = patch.sum()
                if total <= 0:                      # degenerate window: keep the peak
                    cy, cx = py[i, j].to(logits.dtype), px[i, j].to(logits.dtype)
                else:
                    cy = (patch.sum(dim=1) * ys[y0:y1]).sum() / total
                    cx = (patch.sum(dim=0) * xs[x0:x1]).sum() / total
                out[i, j * 2] = cx / (w - 1)
                out[i, j * 2 + 1] = cy / (h - 1)
        return out

    def target_heatmaps(self, target: torch.Tensor, size: int) -> torch.Tensor:
        """Gaussian targets at heatmap resolution from normalised coordinates.

        `target` is [B, 4] as (od_x, od_y, fovea_x, fovea_y) in [0, 1].
        """
        b = target.shape[0]
        grid = torch.arange(size, device=target.device, dtype=target.dtype)
        maps = torch.zeros(b, 2, size, size, device=target.device, dtype=target.dtype)
        for j in range(2):
            cx = target[:, j * 2] * (size - 1)
            cy = target[:, j * 2 + 1] * (size - 1)
            dx = grid.view(1, 1, size) - cx.view(b, 1, 1)
            dy = grid.view(1, size, 1) - cy.view(b, 1, 1)
            maps[:, j] = torch.exp(-(dx ** 2 + dy ** 2) / (2 * self.sigma ** 2))
        return maps


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
