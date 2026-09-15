"""Lesion-mask dataset for M2a (experiments C2-C4).

build_cache.py writes masks as

    <cache-root>/<dataset>/masks/<CHANNEL>/<stem>.png

one directory per lesion channel, flat, keyed by the image's stem. A channel with
no file for an image means that lesion is absent from it -- DDR and IDRiD only
ship a mask where the lesion occurs -- so an absent file becomes an all-zero
target rather than a skipped sample.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Channel order, fixed. docs/03_model_architecture.md section M2a.
#:
#: These are the directory names **build_cache.py actually writes** -- its
#: LESION_CHANNELS -- not the MA/HE/EX/SE that DDR uses in its own raw tree.
#: build_cache normalises both DDR's abbreviations and IDRiD's "1. Microaneurysms"
#: to one vocabulary, and the cache is what this code reads. An earlier version
#: used the DDR spelling here, so mask_path looked in masks/MA/ while every mask
#: sat in masks/microaneurysm/, and C2 reported "no masks on disk" against a cache
#: that had all of them. tests/test_evidence.py pins these against build_cache.
MASK_DIRS = ("microaneurysm", "haemorrhage", "hard_exudate", "soft_exudate")
LESION_NAMES = MASK_DIRS

#: Log abbreviations only. Distinct by construction -- "haemorrhage" and
#: "hard_exudate" both truncate to "HA", which hid a collapsed channel once.
SHORT_LABELS = ("MA", "HE", "EX", "SE")


def mask_path(image_path: Path, channel: str) -> Path:
    """Where build_cache.py put this image's mask for one channel."""
    image_path = Path(image_path)
    # .../<dataset>/images/<maybe/nested>/<stem>.jpg -> .../<dataset>/masks/<CH>/<stem>.png
    parts = image_path.parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i + 1] == "images":
            dataset_dir = Path(*parts[: i + 1])
            return dataset_dir / "masks" / channel / f"{image_path.stem}.png"
    return image_path.parent.parent / "masks" / channel / f"{image_path.stem}.png"


def available_channels(image_path: Path) -> List[str]:
    return [c for c in MASK_DIRS if mask_path(image_path, c).exists()]


def load_segmentation_manifest(
    path: Path,
    cache_root=None,
    datasets: Optional[Sequence[str]] = None,
    require_any_mask: bool = True,
) -> pd.DataFrame:
    """Manifest rows that actually have lesion masks on disk.

    `require_any_mask` drops images with no mask in any channel. Such an image is
    either genuinely lesion-free or simply not part of the annotated subset, and
    nothing in the file distinguishes the two -- training on it as all-negative
    would teach the model that unannotated means healthy.
    """
    from verify_dr.data.dataset import repath_to_cache

    frame = pd.read_csv(path)
    if datasets:
        wanted = {d.lower() for d in datasets}
        frame = frame[frame["dataset"].str.lower().isin(wanted)]
        if frame.empty:
            raise ValueError(f"{path}: no rows for dataset(s) {sorted(wanted)}")
    if cache_root is not None:
        frame = repath_to_cache(frame, cache_root)

    frame = frame.reset_index(drop=True)
    counts = frame["image_path"].map(lambda p: len(available_channels(p)))
    if require_any_mask:
        before = len(frame)
        frame = frame[counts > 0].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"{path}: no image has a mask in any of {MASK_DIRS}. Check that the "
                "cache root holding the masks is attached -- IDRiD's ship in a "
                "separate dataset from the main build."
            )
        if len(frame) != before:
            print(f"  {Path(path).name}: {len(frame)} of {before} images carry masks")
    return frame


class SegmentationDataset(Dataset):
    """Image plus a [4, H, W] mask stack in MASK_DIRS order.

    Augmentation is applied to image and mask **together**. A flip applied to one
    and not the other trains the model against mirrored targets and looks exactly
    like a model that failed to converge, so the transforms here are deliberately
    limited to ones that can be applied identically to both, and
    `tests/test_evidence.py` pins the synchronisation.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: int = 512,
        train: bool = False,
        seed: int = 42,
        hflip: float = 0.5,
        vflip: float = 0.0,
        jitter: float = 0.2,
    ) -> None:
        self.frame = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.train = train
        self.hflip = hflip if train else 0.0
        self.vflip = vflip if train else 0.0
        self.jitter = jitter if train else 0.0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.frame)

    def _load_masks(self, image_path: Path) -> np.ndarray:
        from PIL import Image

        stack = np.zeros((len(MASK_DIRS), self.image_size, self.image_size), dtype=np.float32)
        for i, channel in enumerate(MASK_DIRS):
            path = mask_path(image_path, channel)
            if not path.exists():
                continue                       # absent means the lesion is absent
            with Image.open(path) as handle:
                mask = handle.convert("L")
                if mask.size != (self.image_size, self.image_size):
                    # NEAREST, never bilinear: interpolating a binary mask invents
                    # partial-membership pixels along every lesion boundary, and a
                    # microaneurysm is only a few pixels across to begin with.
                    mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
                stack[i] = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        return stack

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        from PIL import Image
        from torchvision import transforms as T

        path = Path(self.frame.at[index, "image_path"])
        with Image.open(path) as handle:
            img = handle.convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        masks = self._load_masks(path)

        if self.hflip and self._rng.random() < self.hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            masks = masks[:, :, ::-1].copy()
        if self.vflip and self._rng.random() < self.vflip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            masks = masks[:, ::-1, :].copy()

        tensor = T.functional.to_tensor(img)
        if self.jitter:
            # Photometric only, so the mask is untouched by construction.
            tensor = T.ColorJitter(brightness=self.jitter, contrast=self.jitter)(tensor)
        tensor = T.functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

        return {
            "image": tensor,
            "mask": torch.from_numpy(masks),
            "index": torch.tensor(index, dtype=torch.long),
        }


def channel_presence(frame: pd.DataFrame) -> Dict[str, int]:
    """How many images carry each lesion. Soft exudates are routinely rare, and a
    channel that appears in a handful of images cannot support a Dice figure."""
    out = {name: 0 for name in LESION_NAMES}
    for path in frame["image_path"]:
        for channel, name in zip(MASK_DIRS, LESION_NAMES):
            if mask_path(path, channel).exists():
                out[name] += 1
    return out
