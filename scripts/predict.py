#!/usr/bin/env python
"""The single pass: one label-free prediction table per split (ANALYSIS_PLAN.md s9).

Every analysis in Phase 6b and Phase 7 reads what this writes, and nothing reads the
images again. So this script records the raw material, not conclusions: M1's
cumulative logits rather than calibrated probabilities, M2's lesion counts and M3's
verdict, and M1's expected grade on the lesion-inpainted copy and on each of the
19 equal-area controls. Calibration, the gate and every score are computed
downstream from these columns and the frozen fitted parameters -- so a calibration
bug found later costs a re-computation, never a second look at a locked image.

    # calibration or val: everything
    python scripts/predict.py --manifest eyepacs_full.csv --split calibration \\
        --grading-checkpoints results/H1_*/best.pt \\
        --evidence-checkpoint results/C2_lesions/best.pt \\
        --cache-root <roots> --out-dir predictions/calibration

    # the OOD reference (plan s5.3): 5,000 training images, embeddings only
    python scripts/predict.py --manifest eyepacs_full.csv --split train \\
        --sample 5000 --sample-seed 0 --embeddings-only \\
        --grading-checkpoints results/H1_eyepacs_full_s4*/best.pt \\
        --cache-root <roots> --out-dir predictions/reference_eyepacs_full

**It writes no label.** The manifest's label columns are dropped the moment it is
read, before any row is selected or any image opened; `tests/test_predict.py`
checks the source statically and the outputs end to end. For internal splits the
labels are joined later, from the manifest, by the code that needs them.

**Locked data needs --locked.** The EyePACS test split, APTOS and Messidor-2 are
refused without it, before any image is read, so rehearsing on internal data
cannot touch them by accident.

Work is written in shards and a shard is only marked done once all its files are.
A killed session loses at most one shard: re-run the same command to resume.
Exit codes: 0 done, 3 stopped at --max-minutes (re-run to resume), 2 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_dr.data.dataset import IMAGENET_MEAN, IMAGENET_STD, repath_to_cache  # noqa: E402
from verify_dr.data.segmentation import LESION_NAMES  # noqa: E402
from verify_dr.models.evidence import LesionSegmenter  # noqa: E402
from verify_dr.models.grading import GradingModel, cumulative_to_grade  # noqa: E402
from verify_dr.reasoning import extract_facts, grade, lesion_region_mask  # noqa: E402
from verify_dr.reasoning.facts import MIN_LESION_PX  # noqa: E402
from verify_dr.triage import faithfulness as F  # noqa: E402

#: Every name a label goes by in this project's manifests, and the pattern that
#: catches the ones not yet invented. This is the only place a label is named.
LABEL_COLUMNS = ("grade", "true_grade", "label", "level", "diagnosis")
LABEL_PATTERN = re.compile(r"(^|_)(grade|label|level|diagnosis)($|_)", re.IGNORECASE)

#: frozen_config.evidence_model -- pinned here, not exposed as flags.
EVIDENCE_THRESHOLD = 0.5
GEOMETRY_REASON = "C1 failed its 0.5 DD gate (D1): count-only"

LOCKED_DATASETS = {"aptos", "messidor2", "messidor-2", "messidor"}
LOCKED_SPLITS = {"test"}

EXIT_DONE, EXIT_REFUSED, EXIT_PARTIAL = 0, 2, 3


# ------------------------------------------------------------------ manifest


def read_unlabelled(path: Path) -> pd.DataFrame:
    """The manifest, minus every label column. The only reader in this script."""
    frame = pd.read_csv(path)
    labels = [c for c in frame.columns if c in LABEL_COLUMNS or LABEL_PATTERN.search(c)]
    return frame.drop(columns=labels)


def locked_reasons(frame: pd.DataFrame, manifest: Path) -> List[str]:
    reasons = []
    if "split" in frame.columns and set(frame["split"].astype(str)) & LOCKED_SPLITS:
        reasons.append("rows from a test split")
    if "dataset" in frame.columns:
        hit = sorted({d for d in frame["dataset"].astype(str) if d.lower() in LOCKED_DATASETS})
        if hit:
            reasons.append(f"locked dataset(s) {hit}")
    if "locked" in frame.columns and frame["locked"].astype(str).str.lower().isin(
            {"true", "1"}).any():
        reasons.append("rows marked locked")
    if "external" in manifest.name.lower():
        reasons.append(f"an external manifest ({manifest.name})")
    return reasons


def image_ids(frame: pd.DataFrame) -> List[str]:
    """Dataset-qualified stems: EyePACS and DDR can share a file name."""
    datasets = frame["dataset"].astype(str) if "dataset" in frame.columns else ""
    stems = frame["image_path"].map(lambda p: Path(str(p)).stem)
    return [f"{d}::{s}" for d, s in zip(datasets, stems)]


# ------------------------------------------------------------------ images


def load_rgb(path: str, size: int) -> np.ndarray:
    """[H, W, 3] uint8, resized exactly as the eval transforms do.

    Both pathways' eval transforms resize with PIL bilinear and then normalise with
    ImageNet statistics; on a 512 px cache image at size 512 the resize is a copy.
    Normalisation happens on the device (`normalise`), so workers ship uint8 --
    a quarter of the memory for 21 copies of every lesion image.
    """
    from PIL import Image

    with Image.open(path) as handle:
        img = handle.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)        # a writable copy; PIL's view is read-only


def to_chw(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()


def normalise(batch: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 3, H, W] -> exactly ToTensor() then Normalize(ImageNet)."""
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch.float().div_(255.0) - mean) / std


class Originals(Dataset):
    def __init__(self, paths: Sequence[str], size: int) -> None:
        self.paths, self.size = list(paths), size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return to_chw(load_rgb(self.paths[i], self.size)), i


class Counterfactuals(Dataset):
    """The lesion-inpainted copy and its K controls, built in the worker."""

    def __init__(self, entries: Sequence[tuple], size: int, k: int) -> None:
        self.entries, self.size, self.k = list(entries), size, k

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        path, image_id, packed, shape, row = self.entries[i]
        rgb = load_rgb(path, self.size)
        mask = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape).astype(bool)
        v = F.make_variants(rgb, mask, image_id, k=self.k)
        blank = np.zeros_like(rgb)
        stack = [v.lesion] + [c if c is not None else blank for c in v.controls]
        ok = torch.tensor([c is not None for c in v.controls], dtype=torch.bool)
        return torch.stack([to_chw(x) for x in stack]), ok, v.region_px, row


# ------------------------------------------------------------------ models


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Grader:
    """One M1 instance on one device, with its penultimate layer tapped."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = state.get("config", {})
        if cfg.get("eye_pair_fusion"):
            raise ValueError(f"{checkpoint}: a fusion model needs the fellow eye; "
                             "the frozen recipe has none (B5)")
        self.name = checkpoint.parent.name
        self.path = checkpoint
        self.config = cfg
        self.image_size = int(cfg.get("image_size", 512))
        self.device = device
        self.model = GradingModel(backbone=cfg.get("backbone", "efficientnet_b0"),
                                  pretrained=False, dropout=cfg.get("dropout", 0.3),
                                  eye_pair_fusion=False,
                                  head=cfg.get("head", "ordinal_focal"))
        self.model.load_state_dict(state["model"])
        self.model.to(device).eval()
        self._tap: Dict[str, torch.Tensor] = {}
        self.model.neck.register_forward_hook(
            lambda _m, _i, out: self._tap.__setitem__("embedding", out))
        self.best_epoch = state.get("epoch")

    def launch(self, x: torch.Tensor, amp: bool):
        """Queue the forward pass without waiting for it (see `run_graders`)."""
        with torch.autocast(device_type=x.device.type, dtype=torch.float16,
                            enabled=amp and x.device.type == "cuda"):
            out = self.model(x)
        return out["logits"], out["rdr"], out["vtdr"], self._tap["embedding"]


@torch.no_grad()
def run_graders(graders: Sequence[Grader], batch_uint8: torch.Tensor, amp: bool,
                keep_embeddings: bool) -> Dict[str, Dict[str, np.ndarray]]:
    """All M1 instances on one batch.

    The batch is copied and normalised once per device, and every model's work is
    queued before any result is fetched, so two GPUs run their three models each
    concurrently instead of taking turns.
    """
    on_device = {}
    for dev in {g.device for g in graders}:
        on_device[dev] = normalise(batch_uint8.to(dev, non_blocking=True))
    pending = [(g, g.launch(on_device[g.device], amp)) for g in graders]
    results = {}
    for g, (logits, rdr, vtdr, emb) in pending:
        z = logits.float()
        results[g.name] = {
            "z": z.cpu().numpy(),
            "yhat": cumulative_to_grade(z).cpu().numpy(),
            "rdr": rdr.float().cpu().numpy(),
            "vtdr": vtdr.float().cpu().numpy(),
            "emb": emb.half().cpu().numpy() if keep_embeddings else None,
        }
    return results


def load_segmenter(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = LesionSegmenter(pretrained=False)
    model.load_state_dict(state["model"])
    size = state.get("config", {}).get("image_size")
    return model.to(device).eval(), (int(size) if size else None)


# ------------------------------------------------------------------ one shard


def shard_paths(out: Path, i: int) -> Dict[str, Path]:
    d = out / "shards"
    return {"images": d / f"images_{i:04d}.csv", "m1": d / f"m1_{i:04d}.csv",
            "emb": d / f"emb_{i:04d}.npz", "done": d / f"done_{i:04d}"}


@torch.no_grad()
def process_shard(frame: pd.DataFrame, ids: List[str], graders: List[Grader], segmenter,
                  size: int, args, devices: List[torch.device]) -> tuple:
    amp = not args.no_amp
    pin = devices[0].type == "cuda"
    n, k = len(frame), args.controls
    per_model = {g.name: {"z": np.zeros((n, 4), np.float32), "yhat": np.zeros(n, np.int64),
                          "rdr": np.zeros(n, np.float32), "vtdr": np.zeros(n, np.float32),
                          "emb": np.zeros((n, 512), np.float16),
                          "e_lesion": np.full(n, np.nan), "e_ctrl": np.full((n, k), np.nan)}
                 for g in graders}
    images = pd.DataFrame({"image_id": ids,
                           "dataset": frame.get("dataset", pd.Series([""] * n)).values,
                           "image_path": frame["image_path"].values})
    if "split" in frame.columns:
        images["split"] = frame["split"].values

    # ---- pass A: originals through every M1, and M2 unless embeddings only ----
    loader = DataLoader(Originals(frame["image_path"].tolist(), size),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=pin)
    records: List[dict] = [None] * n
    lesion_entries = []
    for batch, rows in loader:
        rows = rows.numpy()
        results = run_graders(graders, batch, amp, keep_embeddings=True)
        for name, r in results.items():
            m = per_model[name]
            m["z"][rows], m["yhat"][rows] = r["z"], r["yhat"]
            m["rdr"][rows], m["vtdr"][rows], m["emb"][rows] = r["rdr"], r["vtdr"], r["emb"]
        if segmenter is None:
            continue
        # M2 in full precision, as C4 ran it: a 0.5 threshold on half-precision
        # probabilities could move a borderline pixel and with it a count.
        x = normalise(batch.to(devices[0], non_blocking=True))
        probs = torch.sigmoid(segmenter(x).float()).cpu().numpy()
        for j, row in enumerate(rows):
            facts = extract_facts(probs[j], threshold=EVIDENCE_THRESHOLD, geometry=None,
                                  geometry_trusted=False, geometry_reason=GEOMETRY_REASON,
                                  min_lesion_px=MIN_LESION_PX)
            verdict = grade(facts)
            region = lesion_region_mask(probs[j], EVIDENCE_THRESHOLD, MIN_LESION_PX)
            rec = {"evidence_grade": verdict.evidence_grade,
                   "max_excludable_grade": verdict.max_excludable_grade,
                   "rule": verdict.rules_fired[0].split(":")[0],
                   "lesion_px": int(region.sum())}
            rec.update({f"n_{name}": facts.counts[name] for name in LESION_NAMES})
            rec.update({f"area_{name}": facts.areas[name] for name in LESION_NAMES})
            records[row] = rec
            if region.any():
                lesion_entries.append((frame["image_path"].iat[row], ids[row],
                                       np.packbits(region.ravel()), region.shape, int(row)))

    if segmenter is not None:
        images = pd.concat([images, pd.DataFrame(records)], axis=1)
        images["region_px"] = 0
        images["controls_ok"] = 0
        images["faith_status"] = "none"

        # ---- pass C: the lesion-inpainted copy and K controls ---------------
        loader = DataLoader(Counterfactuals(lesion_entries, size, k),
                            batch_size=args.variant_batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)
        for stacks, ok, region_px, rows in loader:
            b = stacks.shape[0]
            flat = stacks.view(b * (k + 1), *stacks.shape[2:])
            results = run_graders(graders, flat, amp, keep_embeddings=False)
            ok_np, rows_np = ok.numpy(), rows.numpy()
            for name, r in results.items():
                e = F.expected_grade(r["z"]).reshape(b, k + 1)
                ctrl = np.where(ok_np, e[:, 1:], np.nan)
                per_model[name]["e_lesion"][rows_np] = e[:, 0]
                per_model[name]["e_ctrl"][rows_np] = ctrl
            n_ok = ok_np.sum(axis=1)
            images.loc[rows_np, "region_px"] = region_px.numpy()
            images.loc[rows_np, "controls_ok"] = n_ok
            images.loc[rows_np, "faith_status"] = np.where(
                n_ok >= F.MIN_CONTROLS_OK, "determined", "undetermined")

    # ---- long table: one row per image per model ----------------------------
    blocks = []
    for g in graders:
        m = per_model[g.name]
        block = pd.DataFrame({"image_id": ids, "model": g.name})
        for j in range(4):
            block[f"z{j}"] = m["z"][:, j]
        block["yhat"] = m["yhat"]
        block["rdr_logit"], block["vtdr_logit"] = m["rdr"], m["vtdr"]
        block["e_orig"] = F.expected_grade(m["z"])
        if segmenter is not None:
            block["e_lesion"] = m["e_lesion"]
            for j in range(k):
                block[f"e_c{j:02d}"] = m["e_ctrl"][:, j]
        blocks.append(block)
    embeddings = {g.name: per_model[g.name]["emb"] for g in graders}
    return images, pd.concat(blocks, ignore_index=True), embeddings


def write_shard(out: Path, i: int, images: pd.DataFrame, m1: pd.DataFrame,
                embeddings: Dict[str, np.ndarray]) -> None:
    """Every file first, the done marker last -- a half-written shard is redone."""
    paths = shard_paths(out, i)
    paths["images"].parent.mkdir(parents=True, exist_ok=True)
    def save_npz(path: Path) -> None:
        with open(path, "wb") as fh:            # a handle: np.savez would append .npz
            np.savez(fh, **embeddings)

    for key, writer in (("images", lambda p: images.to_csv(p, index=False)),
                        ("m1", lambda p: m1.to_csv(p, index=False)),
                        ("emb", save_npz)):
        tmp = paths[key].with_name(paths[key].name + ".tmp")
        writer(tmp)
        tmp.replace(paths[key])
    paths["done"].write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"))


def merge(out: Path, n_shards: int, names: List[str]) -> Dict[str, int]:
    images = pd.concat([pd.read_csv(shard_paths(out, i)["images"])
                        for i in range(n_shards)], ignore_index=True)
    m1 = pd.concat([pd.read_csv(shard_paths(out, i)["m1"])
                    for i in range(n_shards)], ignore_index=True)
    images.to_csv(out / "images.csv", index=False)
    m1.to_csv(out / "m1.csv", index=False)
    (out / "emb").mkdir(exist_ok=True)
    for name in names:
        parts = [np.load(shard_paths(out, i)["emb"])[name] for i in range(n_shards)]
        np.save(out / "emb" / f"{name}.npy", np.concatenate(parts).astype(np.float16))
    return {"images": len(images), "rows_m1": len(m1)}


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                               "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------ main


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--split", default=None, help="Rows whose split column equals this.")
    p.add_argument("--grading-checkpoints", required=True, type=Path, nargs="+")
    p.add_argument("--evidence-checkpoint", type=Path, default=None)
    p.add_argument("--embeddings-only", action="store_true",
                   help="M1 outputs and embeddings only: no M2, M3 or faithfulness. "
                        "For the OOD reference sample (plan s5.3).")
    p.add_argument("--sample", type=int, default=0,
                   help="Draw this many rows uniformly after the split filter.")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--cache-root", type=Path, nargs="+", default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--locked", action="store_true",
                   help="Required to read the EyePACS test split, APTOS or Messidor-2. "
                        "Pass it only in the Phase 6b notebook.")
    p.add_argument("--controls", type=int, default=F.CONTROLS)
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--variant-batch", type=int, default=2,
                   help="Lesion images per batch in the faithfulness pass; each "
                        "brings 1 + controls copies.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="Stop after the shard that crosses this. 0 = no limit.")
    p.add_argument("--limit", type=int, default=0, help="First N rows (smoke tests).")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default=None, help="e.g. cpu. Default: every CUDA GPU.")
    args = p.parse_args(argv)
    started = time.time()

    if args.controls != F.CONTROLS:
        print(f"note: --controls {args.controls} differs from the plan's K = {F.CONTROLS}; "
              "record it as a deviation if this run is used for anything")
    if not args.embeddings_only and args.evidence_checkpoint is None:
        print("error: --evidence-checkpoint is required unless --embeddings-only",
              file=sys.stderr)
        return EXIT_REFUSED

    # ---- the manifest: labels go first, then the locked-data guard ---------
    frame = read_unlabelled(args.manifest)
    if args.split is not None:
        if "split" not in frame.columns:
            print(f"error: {args.manifest} has no split column", file=sys.stderr)
            return EXIT_REFUSED
        frame = frame[frame["split"].astype(str) == args.split]
    if frame.empty:
        print(f"error: no rows in {args.manifest} for split={args.split!r}", file=sys.stderr)
        return EXIT_REFUSED
    reasons = locked_reasons(frame, args.manifest)
    if reasons and not args.locked:
        print("REFUSED: this selection contains locked data -- " + "; ".join(reasons) + ".\n"
              "  Locked sets are read once, in Phase 6b, after the analysis code and\n"
              "  fitted parameters are committed (ANALYSIS_PLAN.md s10). Nothing was read.",
              file=sys.stderr)
        return EXIT_REFUSED
    if args.sample:
        frame = frame.sample(n=min(args.sample, len(frame)), random_state=args.sample_seed)
    frame = frame.sort_values("image_path", kind="stable")
    if args.limit:
        frame = frame.head(args.limit)
    if args.cache_root:
        frame = repath_to_cache(frame, args.cache_root)
    frame = frame.reset_index(drop=True)
    missing = [q for q in frame["image_path"].head(20) if not Path(str(q)).exists()]
    if missing:
        print(f"error: image paths do not exist, e.g. {missing[:3]}. Pass --cache-root "
              "with every root the cache is split across.", file=sys.stderr)
        return 1
    ids = image_ids(frame)
    if len(set(ids)) != len(ids):
        print("error: duplicate image IDs -- the manifest repeats an image", file=sys.stderr)
        return 1

    # ---- models ----------------------------------------------------------------
    if args.device:
        devices = [torch.device(args.device)]
    elif torch.cuda.is_available():
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]
    graders = [Grader(c, devices[i % len(devices)])
               for i, c in enumerate(args.grading_checkpoints)]
    names = [g.name for g in graders]
    if len(set(names)) != len(names):
        print(f"error: two checkpoints share a run name: {names}", file=sys.stderr)
        return 1
    sizes = {g.image_size for g in graders}
    if len(sizes) != 1:
        print(f"error: M1 instances disagree on image size: {sizes}", file=sys.stderr)
        return 1
    size = sizes.pop()
    segmenter = None
    if not args.embeddings_only:
        segmenter, seg_size = load_segmenter(args.evidence_checkpoint, devices[0])
        if seg_size is not None and seg_size != size:
            print(f"error: M2 was trained at {seg_size} px and M1 at {size} px; one image "
                  "tensor must serve both", file=sys.stderr)
            return 1

    # ---- run record, written before any work so a partial run says what it is ----
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fingerprints = {g.name: {"path": str(g.path), "sha256": sha256(g.path),
                             "best_epoch": g.best_epoch, "config": g.config}
                    for g in graders}
    run = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(args.manifest), "split": args.split, "rows": len(frame),
        "sample": args.sample, "sample_seed": args.sample_seed,
        "embeddings_only": args.embeddings_only, "locked": bool(reasons),
        "labels_written": False,
        "grading_checkpoints": fingerprints,
        "evidence_checkpoint": None if segmenter is None else {
            "path": str(args.evidence_checkpoint),
            "sha256": sha256(args.evidence_checkpoint)},
        "constants": {"evidence_threshold": EVIDENCE_THRESHOLD,
                      "min_lesion_px": MIN_LESION_PX, "controls": args.controls,
                      "min_controls_ok": F.MIN_CONTROLS_OK, "dilate_px": F.DILATE_PX,
                      "inpaint_radius": F.INPAINT_RADIUS, "max_attempts": F.MAX_ATTEMPTS,
                      "fov_threshold": F.FOV_THRESHOLD, "image_size": size},
        "amp": not args.no_amp, "devices": [str(d) for d in devices],
        "torch": torch.__version__, "code_commit": git_commit(),
        "shard_size": args.shard_size,
    }
    previous = out / "run.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        for key in ("manifest", "split", "rows", "sample", "sample_seed", "embeddings_only"):
            if old.get(key) != run[key]:
                print(f"error: {out} holds a different run ({key}: {old.get(key)!r} "
                      f"vs {run[key]!r}). Use a new --out-dir.", file=sys.stderr)
                return 1
        if {k: v["sha256"] for k, v in old["grading_checkpoints"].items()} != \
                {k: v["sha256"] for k, v in fingerprints.items()}:
            print("error: the checkpoints differ from the ones this directory was "
                  "started with. Use a new --out-dir.", file=sys.stderr)
            return 1
    previous.write_text(json.dumps(run, indent=2, default=str))

    n_shards = (len(frame) + args.shard_size - 1) // args.shard_size
    print(f"{len(frame)} images, {n_shards} shard(s), {len(graders)} M1 instance(s)"
          f"{'' if segmenter is None else ', M2 + M3 + faithfulness'}, on {run['devices']}",
          flush=True)
    for i in range(n_shards):
        if shard_paths(out, i)["done"].exists():
            print(f"  shard {i + 1}/{n_shards}: already done", flush=True)
            continue
        lo, hi = i * args.shard_size, min(len(frame), (i + 1) * args.shard_size)
        t0 = time.time()
        images, m1, emb = process_shard(frame.iloc[lo:hi].reset_index(drop=True),
                                        ids[lo:hi], graders, segmenter, size, args, devices)
        write_shard(out, i, images, m1, emb)
        extra = ""
        if segmenter is not None:
            with_lesions = int((images["faith_status"] != "none").sum())
            extra = f", {with_lesions} with lesions"
        print(f"  shard {i + 1}/{n_shards}: {hi - lo} images in {time.time() - t0:.0f}s"
              f"{extra}", flush=True)
        if args.max_minutes and (time.time() - started) / 60 > args.max_minutes and \
                i + 1 < n_shards:
            print(f"\nstopped after shard {i + 1}/{n_shards} at --max-minutes "
                  f"{args.max_minutes}. Re-run the same command to resume.", flush=True)
            return EXIT_PARTIAL

    counts = merge(out, n_shards, names)
    run.update({"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "minutes": round((time.time() - started) / 60, 1), **counts})
    previous.write_text(json.dumps(run, indent=2, default=str))
    print(f"wrote {out}: {counts['images']} images x {len(names)} model(s)", flush=True)
    return EXIT_DONE


if __name__ == "__main__":
    sys.exit(main())
