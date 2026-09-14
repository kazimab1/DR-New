"""Dataset for M2b, the optic-disc / fovea regressor (experiment C1).

Reads the IDRiD manifest written by prepare_manifest.py, whose coordinate columns
are already re-projected into cache pixel space:

    od_x, od_y, od_in_frame, fovea_x, fovea_y, fovea_in_frame

and serves them normalised to [0, 1], which is what the head predicts.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

COORD_COLUMNS = ("od_x", "od_y", "fovea_x", "fovea_y")


def load_geometry_manifest(
    path: Path, cache_root=None, require_in_frame: bool = True
) -> pd.DataFrame:
    """Rows that carry both centres, with paths repointed at the current cache."""
    from verify_dr.data.dataset import repath_to_cache

    frame = pd.read_csv(path)
    missing = [c for c in COORD_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: missing coordinate columns {missing}. Re-run prepare_manifest.py "
            "with --coords and --coords-source-dir for IDRiD Part C."
        )

    before = len(frame)
    frame = frame.dropna(subset=list(COORD_COLUMNS))
    if require_in_frame:
        for flag in ("od_in_frame", "fovea_in_frame"):
            if flag in frame.columns:
                frame = frame[frame[flag].astype(int) == 1]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path}: no rows have both centres in frame")
    if len(frame) != before:
        print(f"  {Path(path).name}: kept {len(frame)} of {before} rows with usable centres")

    if cache_root is not None:
        frame = repath_to_cache(frame, cache_root)
    absent = [p for p in frame["image_path"].head(20) if not Path(p).exists()]
    if absent:
        raise FileNotFoundError(
            f"{path}: image paths do not exist, e.g. {absent[:3]}. Pass --cache-root."
        )
    return frame


class GeometryDataset(Dataset):
    """One row per image. Targets are [od_x, od_y, fovea_x, fovea_y] in [0, 1].

    Augmentation is deliberately limited to transforms whose effect on the
    coordinates is exact and obvious: a horizontal flip, and photometric jitter
    that does not move anything. Rotation and scaling are omitted on purpose --
    they would roughly double the effective dataset again, but a sign or centre-of-
    rotation error in the coordinate transform corrupts the targets silently and
    would look exactly like a model that simply failed to learn. With 516 images
    the flip alone doubles the set, and `tests/test_evidence.py` pins its
    correctness.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: int = 512,
        cache_size: int = 512,
        train: bool = False,
        seed: int = 42,
        hflip: float = 0.5,
        jitter: float = 0.2,
    ) -> None:
        self.frame = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.cache_size = cache_size
        self.train = train
        self.hflip = hflip if train else 0.0
        self.jitter = jitter if train else 0.0
        self._rng = random.Random(seed)

        # Normalised once here rather than per __getitem__.
        self.targets = (
            self.frame[list(COORD_COLUMNS)].to_numpy(dtype=np.float32) / float(cache_size)
        ).clip(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        from PIL import Image
        from torchvision import transforms as T

        path = self.frame.at[index, "image_path"]
        with Image.open(path) as handle:
            img = handle.convert("RGB")
        target = self.targets[index].copy()

        if self.hflip and self._rng.random() < self.hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Only x moves, and it reflects about the frame centre. In normalised
            # coordinates that is exactly 1 - x, for both landmarks.
            target[0] = 1.0 - target[0]
            target[2] = 1.0 - target[2]

        if self.image_size != img.size[0]:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        tensor = T.functional.to_tensor(img)
        if self.jitter:
            tensor = T.ColorJitter(brightness=self.jitter, contrast=self.jitter)(tensor)
        tensor = T.functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

        return {
            "image": tensor,
            "target": torch.from_numpy(target),
            "index": torch.tensor(index, dtype=torch.long),
        }


def patient_split(
    frame: pd.DataFrame, val_frac: float = 0.2, seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Split by patient_id, never by row (CLAUDE.md rule 2).

    IDRiD images are one per eye with no repeated subjects, so this is usually a
    plain shuffle -- but it is written this way so the guarantee does not depend on
    that staying true for whatever mirror is attached.
    """
    patients = sorted(frame["patient_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(patients)
    cut = max(1, int(round(len(patients) * val_frac)))
    val_ids = set(patients[:cut])

    val = frame[frame["patient_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    train = frame[~frame["patient_id"].astype(str).isin(val_ids)].reset_index(drop=True)

    overlap = set(train["patient_id"].astype(str)) & set(val["patient_id"].astype(str))
    if overlap:
        raise AssertionError(f"patient split leaked: {sorted(overlap)[:5]}")
    return {"train": train, "val": val}
