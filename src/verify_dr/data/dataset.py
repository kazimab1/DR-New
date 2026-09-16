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


CACHE_CHANNELS = ("images", "masks")


def cache_tail(value: str, dataset: Optional[str] = None) -> Optional[Path]:
    """The <dataset>/images/<rel> portion of a cached image path.

    build_cache.py writes `image_out / rel`, where rel is the image's full path
    relative to its source root -- so nesting is preserved verbatim. EyePACS
    arrives as

        <root>/eyepacs/images/EYEPACS_original.../train/2/16_left.jpg

    five levels below the dataset directory, while APTOS and IDRiD sit flat at
    one. Anchoring on the <dataset>/images boundary handles both; assuming a
    fixed three-component tail silently truncates EyePACS to train/2/16_left.jpg
    and every path then misses.
    """
    parts = Path(value).parts
    for i in range(len(parts) - 2, -1, -1):        # last match wins
        if parts[i + 1] not in CACHE_CHANNELS:
            continue
        if dataset and parts[i].lower() != str(dataset).lower():
            continue
        return Path(*parts[i:])
    return None


def _mask_channels(root: Path, dataset: str) -> int:
    """How many mask channel directories this root has for `dataset`."""
    masks = root / dataset / "masks"
    try:
        return sum(1 for d in masks.iterdir() if d.is_dir())
    except OSError:
        return 0


def _best_root(roots, tail, dataset: str) -> Path:
    """The root holding the most complete copy of `dataset`.

    Candidates must actually contain the file; among those, more mask channels
    wins. Order breaks a tie, so a single-root cache behaves exactly as before.
    """
    holders = [r for r in roots if (r / tail).exists()]
    if not holders:
        return roots[0]
    return max(holders, key=lambda r: _mask_channels(r, dataset))


def repath_to_cache(frame: pd.DataFrame, cache_root) -> pd.DataFrame:
    """Point image_path at `cache_root`, keeping the <dataset>/images/... tail.

    Manifests store absolute paths, and the cache is mounted somewhere different
    in every Kaggle session -- a Phase 2 manifest names the path the cache had
    during Phase 2.

    `cache_root` may be several roots. The cache is legitimately split across
    published datasets: a full build plus a later top-up (IDRiD's masks ship as
    their own dataset). Each dataset resolves to the root holding the **most
    complete** copy, so a split cache works without merging it on disk.

    "Most complete" means most mask channels, not merely first-with-the-image.
    IDRiD appears in both roots -- the Phase 1 build has Part A's images with no
    masks, the top-up has them with masks -- so resolving on image existence
    alone picked the mask-less copy, every row was dropped for carrying no mask,
    and C3 reported IDRiD absent from a cache that had it. Images are identical
    between the two, so preferring the mask-bearing root costs a caller that
    only wants images nothing at all.
    """
    roots = [Path(r) for r in (cache_root if isinstance(cache_root, (list, tuple))
                               else [cache_root])]
    datasets = frame["dataset"] if "dataset" in frame.columns else None

    # Resolved once per dataset, not once per row: there are at most a handful of
    # datasets and a stat() per row over 88k EyePACS images is pure waste.
    chosen: Dict[str, Path] = {}

    def rewrite(value: str, dataset: Optional[str]) -> str:
        tail = cache_tail(value, dataset)
        if tail is None:
            return value
        key = tail.parts[0]
        if key not in chosen:
            chosen[key] = _best_root(roots, tail, key)
        return str(chosen[key] / tail)

    frame = frame.copy()
    if datasets is None:
        frame["image_path"] = frame["image_path"].map(lambda v: rewrite(v, None))
    else:
        frame["image_path"] = [
            rewrite(v, d) for v, d in zip(frame["image_path"], datasets)
        ]
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
                f"Checked against cache_root={cache_root}. Is that the right root? "
                "If the cache is split across several datasets, pass each root.")
        raise FileNotFoundError(f"{path}: image paths do not exist, e.g. {missing[:3]}. {hint}")
    return frame.reset_index(drop=True)


def make_sampler(
    grades: np.ndarray, strategy: str, epoch_samples: Optional[int] = None, seed: int = 42
) -> Optional[WeightedRandomSampler]:
    """Sampler over training rows. None means plain shuffling.

    Both weighted strategies draw with probability inversely proportional to
    grade frequency, so every grade is equally likely per draw. They differ only
    in how many draws make an epoch:

        class_balanced       always len(grades) draws
        stratified_exposure  epoch_samples draws, defaulting to len(grades)

    **So with epoch_samples unset the two are the same sampler**, and given the
    same seed they emit the same index sequence. That is not a bug, but it is a
    trap: B4 was originally run as a three-way sweep and its third arm turned out
    to be a duplicate of the second. Pass --epoch-samples to make
    stratified_exposure genuinely different, or expect identical results. The
    warning below fires whenever the two coincide.

    Neither strategy changes the data -- only how often each row is drawn.
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
        if not epoch_samples:
            print("  note: stratified_exposure without --epoch-samples is identical to "
                  "class_balanced;\n        pass --epoch-samples to make them differ.")

    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


def grade_counts(frame: pd.DataFrame) -> Dict[int, int]:
    return {int(g): int(n) for g, n in sorted(frame["grade"].value_counts().items())}
