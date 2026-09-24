"""Reading what `scripts/predict.py` wrote, and joining labels to it -- in one place.

A pass directory holds `images.csv` (one row per image), `m1.csv` (one row per image
per model), `emb/<model>.npy` (rows in `images.csv` order) and `run.json`. Everything
downstream reads it through `load_pass`, which aligns every model's rows to the image
order by ID rather than trusting file order, and refuses a directory whose parts
disagree.

Labels are joined by `labels_for`, the only function here that reads a grade.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MODEL_PATTERN = re.compile(r"^H1_(eyepacs_full|eyepacs_ddr_full)_s(\d+)$")
PRIMARY_VARIANT = "eyepacs_full"          # plan s3: decides H1, H1', H2
REPLICATION_VARIANT = "eyepacs_ddr_full"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_model(name: str) -> tuple:
    """'H1_eyepacs_ddr_full_s43' -> ('eyepacs_ddr_full', 43)."""
    m = MODEL_PATTERN.match(name)
    if not m:
        raise ValueError(f"not a Phase 6a run name: {name!r}")
    return m.group(1), int(m.group(2))


def image_id(frame: pd.DataFrame) -> pd.Series:
    """The pass's image key: dataset-qualified file stem (predict.image_ids)."""
    stems = frame["image_path"].map(lambda p: Path(str(p)).stem)
    return frame["dataset"].astype(str) + "::" + stems


@dataclass
class Pass:
    root: Path
    images: pd.DataFrame
    m1: pd.DataFrame
    run: dict

    @property
    def ids(self) -> np.ndarray:
        return self.images["image_id"].to_numpy()

    @property
    def models(self) -> List[str]:
        return list(dict.fromkeys(self.m1["model"]))

    @property
    def has_evidence(self) -> bool:
        return "evidence_grade" in self.images.columns

    def model_rows(self, name: str) -> pd.DataFrame:
        rows = self.m1[self.m1["model"] == name].set_index("image_id")
        if len(rows) != len(self.images) or not rows.index.is_unique:
            raise ValueError(f"{self.root}: {name} has {len(rows)} rows for "
                             f"{len(self.images)} images")
        return rows.loc[self.ids]

    def embeddings(self, name: str) -> np.ndarray:
        emb = np.load(self.root / "emb" / f"{name}.npy")
        if emb.shape[0] != len(self.images):
            raise ValueError(f"{self.root}: {name} embeddings have {emb.shape[0]} rows "
                             f"for {len(self.images)} images")
        return emb.astype(np.float64)

    def checkpoint_sha256(self, name: str) -> str:
        return self.run["grading_checkpoints"][name]["sha256"]

    def fingerprint(self) -> Dict[str, str]:
        """sha256 of every file the analysis reads from this directory."""
        files = [self.root / "images.csv", self.root / "m1.csv", self.root / "run.json"]
        files += sorted((self.root / "emb").glob("*.npy"))
        return {str(f.relative_to(self.root)): sha256_file(f) for f in files}


def load_pass(root: Path) -> Pass:
    root = Path(root)
    for need in ("images.csv", "m1.csv", "run.json"):
        if not (root / need).exists():
            raise FileNotFoundError(f"{root / need} missing -- is this pass complete?")
    run = json.loads((root / "run.json").read_text())
    if run.get("labels_written") is not False:
        raise ValueError(f"{root}: run.json does not certify a label-free pass")
    images = pd.read_csv(root / "images.csv")
    m1 = pd.read_csv(root / "m1.csv")
    if not images["image_id"].is_unique:
        raise ValueError(f"{root}: duplicate image IDs")
    if set(m1["image_id"]) != set(images["image_id"]):
        raise ValueError(f"{root}: m1.csv and images.csv cover different images")
    missing = [m for m in dict.fromkeys(m1["model"]) if m not in run["grading_checkpoints"]]
    if missing:
        raise ValueError(f"{root}: models {missing} have no checkpoint record in run.json")
    return Pass(root, images, m1, run)


def labels_for(ids: np.ndarray, manifest: Path, split: Optional[str] = None,
               column: str = "grade") -> np.ndarray:
    """The true grade of every image in `ids`, from a manifest. Every ID must resolve.

    The only label reader in the Phase 7 code. Callers decide *whether* labels may be
    read (fit_params: internal splits; analyse: internal, or --unblind).
    """
    frame = pd.read_csv(manifest)
    if split is not None:
        frame = frame[frame["split"].astype(str) == split]
    keyed = pd.Series(frame[column].to_numpy(), index=image_id(frame).to_numpy())
    if not keyed.index.is_unique:
        raise ValueError(f"{manifest}: an image appears twice")
    absent = [i for i in ids if i not in keyed.index]
    if absent:
        raise ValueError(f"{manifest}{'' if split is None else f' (split={split})'}: "
                         f"{len(absent)} image(s) have no label, e.g. {absent[:3]}")
    return keyed.loc[list(ids)].to_numpy().astype(int)
