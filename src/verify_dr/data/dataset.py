"""Manifest-backed dataset, transforms and samplers for the grading pathway.

Reads the manifests written by prepare_manifest.py / build_variants.py and serves
images out of the 512 px cache. Nothing here ever reads a raw dataset.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAMPLERS = ("natural", "stratified_exposure", "class_balanced")


def build_transforms(image_size: int, train: bool):
    """Augmentation per docs/03_model_architecture.md section M1.

    No colour-channel shuffling: lesion colour is diagnostic. Hue is left alone
    for the same reason -- a haemorrhage and an exudate differ largely by colour.
    """
    from torchvision import transforms as T

    if not train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, scale=(0.9, 1.1), interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class GradingDataset(Dataset):
    """One row per image. Optionally serves the fellow eye alongside it."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: int = 512,
        train: bool = False,
        eye_pair_fusion: bool = False,
        seed: int = 42,
    ) -> None:
        self.frame = manifest.reset_index(drop=True)
        self.transform = build_transforms(image_size, train)
        self.eye_pair_fusion = eye_pair_fusion
        self.train = train
        self._rng = random.Random(seed)

        # patient -> row indices, for the fellow-eye lookup
        self.by_patient: Dict[str, List[int]] = defaultdict(list)
        for i, pid in enumerate(self.frame["patient_id"]):
            self.by_patient[pid].append(i)

        self.grades = self.frame["grade"].to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def _load(self, index: int) -> torch.Tensor:
        from PIL import Image

        path = self.frame.at[index, "image_path"]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))

    def _fellow_index(self, index: int) -> int:
        """The other eye of the same patient, or this image again.

        Duplicating is deliberate: a patient with one usable eye must still
        produce a fused vector of the same width (M1, eye-pair fusion).
        """
        siblings = [i for i in self.by_patient[self.frame.at[index, "patient_id"]] if i != index]
        if not siblings:
            return index
        return self._rng.choice(siblings) if self.train else siblings[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        grade = int(self.grades[index])
        item = {
            "image": self._load(index),
            "grade": torch.tensor(grade, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.eye_pair_fusion:
            item["fellow"] = self._load(self._fellow_index(index))
        return item


def repath_to_cache(frame: pd.DataFrame, cache_root: Path) -> pd.DataFrame:
    """Point image_path at `cache_root`, keeping the <dataset>/images/<file> tail.

    Manifests store absolute paths, and the cache is mounted somewhere different
    in every Kaggle session -- a Phase 2 manifest names the path the cache had
    during Phase 2. build_cache.py always writes <root>/<dataset>/images/<file>,
    so the last three components identify the image and only the root moves.
    """
    cache_root = Path(cache_root)

    def rewrite(value: str) -> str:
        parts = Path(value).parts
        return str(cache_root.joinpath(*parts[-3:])) if len(parts) >= 3 else value

    frame = frame.copy()
    frame["image_path"] = frame["image_path"].map(rewrite)
    return frame


def load_manifest(path: Path, split: Optional[str] = None,
                  cache_root: Optional[Path] = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if split:
        if "split" not in frame.columns:
            raise ValueError(f"{path} has no 'split' column -- use a build_variants.py output")
        frame = frame[frame["split"] == split]
    frame = frame[frame["grade"].between(0, 4)]
    if frame.empty:
        raise ValueError(f"{path}: no rows left for split={split!r}")
    if cache_root is not None:
        frame = repath_to_cache(frame, cache_root)
    missing = [p for p in frame["image_path"].head(20) if not Path(p).exists()]
    if missing:
        hint = ("Is the cache mounted at the same location it was built at? "
                "Pass --cache-root to repath the manifest."
                if cache_root is None else
                f"Checked against cache_root={cache_root}. Is that the right root?")
        raise FileNotFoundError(f"{path}: image paths do not exist, e.g. {missing[:3]}. {hint}")
    return frame.reset_index(drop=True)


def make_sampler(
    grades: np.ndarray, strategy: str, epoch_samples: Optional[int] = None, seed: int = 42
) -> Optional[WeightedRandomSampler]:
    """Sampler over training rows. None means plain shuffling.

    stratified_exposure draws a fixed number of samples per epoch with every
    grade equally likely, so rare grades are seen often without discarding the
    common ones. class_balanced weights by inverse frequency over the natural
    epoch length. Neither changes the data -- only how often each row is drawn.
    """
    if strategy == "natural":
        return None
    if strategy not in SAMPLERS:
        raise ValueError(f"unknown sampler {strategy!r}; expected one of {SAMPLERS}")

    counts = np.bincount(grades, minlength=5).astype(float)
    counts[counts == 0] = np.inf                     # never draw an absent grade
    weights = (1.0 / counts)[grades]

    num_samples = len(grades)
    if strategy == "stratified_exposure":
        num_samples = epoch_samples or len(grades)

    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


def grade_counts(frame: pd.DataFrame) -> Dict[int, int]:
    return {int(g): int(n) for g, n in sorted(frame["grade"].value_counts().items())}
