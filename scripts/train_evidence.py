#!/usr/bin/env python3
"""Train M2a, the four-channel lesion segmenter (Phase 4, experiments C2-C3).

    512x512x3 -> ResNet18 encoder -> UNet decoder -> 4 channels
    L = 0.5*Dice + 0.5*BCE

Channels are microaneurysm, haemorrhage, hard exudate and soft exudate -- the only
lesion types with pixel annotations on a Kaggle-available dataset. Venous beading,
IRMA and neovascularisation are not labelled anywhere reachable, and M3 declares
them unobservable rather than pretending (docs/03_model_architecture.md § M2a).

    # C2: train on everything with masks
    python scripts/train_evidence.py --manifest manifests/idrid_manifest.csv \\
        --experiment C2_lesions --encoder-from results/stage_c/C1_geometry/best.pt

    # C3: cross-domain, train on one source and test on the other
    python scripts/train_evidence.py --manifest manifests/combined.csv \\
        --experiment C3_ddr_to_idrid --train-datasets ddr --val-datasets idrid

The encoder is initialised from C1's checkpoint when `--encoder-from` is given.
docs/03 has M2a and M2b sharing an encoder, but their supervision is disjoint --
DDR has masks and no centres, IDRiD Part C has centres and no masks -- so the two
are fitted in sequence rather than jointly, geometry first.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from verify_dr.data.segmentation import (  # noqa: E402
    LESION_NAMES, SHORT_LABELS, SegmentationDataset, channel_presence,
    load_segmentation_manifest,
)
from verify_dr.evaluation.segmentation_metrics import (  # noqa: E402
    format_report, segmentation_metrics,
)
from verify_dr.models.evidence import LesionSegmenter  # noqa: E402
from verify_dr.models.seg_losses import SegmentationLoss  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool) -> Dict[str, object]:
    model.eval()
    probs, truths, losses = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(image)
            losses.append(float(criterion(logits, mask)["loss"].detach()))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        truths.append(mask.float().cpu().numpy())

    metrics = segmentation_metrics(np.concatenate(probs), np.concatenate(truths))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_epoch(model, loader, criterion, optimiser, scaler, device, amp: bool) -> Dict[str, float]:
    model.train()
    totals, seen = {"loss": 0.0, "dice": 0.0, "bce": 0.0}, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            parts = criterion(model(image), mask)
        scaler.scale(parts["loss"]).backward()
        scaler.step(optimiser)
        scaler.update()

        n = image.size(0)
        seen += n
        for key in totals:
            totals[key] += float(parts[key].detach()) * n
    return {k: v / max(1, seen) for k, v in totals.items()}


ARCHITECTURE_KEYS = ("image_size",)
TRAJECTORY_KEYS = ("lr", "weight_decay", "epochs", "batch_size", "manifest", "seed",
                   "train_datasets", "val_datasets", "dice_weight", "pos_weight")


def check_resume_config(stored: Dict[str, object], current: Dict[str, object]) -> None:
    fatal = [(k, stored.get(k), current[k]) for k in ARCHITECTURE_KEYS
             if stored.get(k) != current[k]]
    if fatal:
        lines = "\n".join(f"    {k}: checkpoint={o!r}  now={n!r}" for k, o, n in fatal)
        raise SystemExit(
            f"error: --resume against a checkpoint built differently:\n{lines}\n"
            "  Use a new --experiment id, or drop --resume to start clean.")
    for k in TRAJECTORY_KEYS:
        if stored.get(k) != current[k]:
            print(f"  note: {k} changed on resume ({stored.get(k)!r} -> {current[k]!r}); "
                  "record this as a deviation")


@dataclass
class Config:
    manifest: str
    experiment: str
    image_size: int
    batch_size: int
    epochs: int
    warmup_epochs: int
    lr: float
    weight_decay: float
    dice_weight: float
    pos_weight: Optional[float]
    train_datasets: Optional[List[str]]
    val_datasets: Optional[List[str]]
    val_frac: float
    pretrained: bool
    encoder_from: Optional[str]
    freeze_encoder: bool
    patience: int
    seed: int
    amp: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the M2a lesion segmenter (C2, C3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path, nargs="+",
                   help="One or more manifests. Several are concatenated, which is "
                        "how DDR-seg and IDRiD-seg are combined for C2.")
    p.add_argument("--experiment", required=True)
    p.add_argument("--results-dir", type=Path, default=Path("results/stage_c"))
    p.add_argument("--cache-root", type=Path, nargs="+", default=None)
    p.add_argument("--train-datasets", nargs="+", default=None,
                   help="Restrict training to these datasets (C3 cross-domain).")
    p.add_argument("--val-datasets", nargs="+", default=None,
                   help="Validate on these instead of a held-out split (C3).")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--pos-weight", type=float, default=None,
                   help="Positive weighting inside BCE. Off by default: with Dice "
                        "already pushing overlap, both together over-segment.")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--encoder-from", type=Path, default=None,
                   help="C1 checkpoint to initialise the shared encoder from.")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", type=Path, default=None, metavar="CHECKPOINT",
                   help="Evaluate this checkpoint on --val-datasets and exit, "
                        "training nothing. The cross-domain question 'does a "
                        "DDR-trained segmenter work on IDRiD' needs no second "
                        "training run -- the DDR model already exists.")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def load_encoder(model: LesionSegmenter, checkpoint: Path, device) -> None:
    """Initialise the encoder from C1's checkpoint.

    The geometry model and the segmenter hold the same ResNet18 under the same
    attribute name, so the encoder weights transfer by prefix. Anything that does
    not match is reported rather than dropped quietly -- a silent no-op here would
    look exactly like an encoder that simply did not help.
    """
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    weights = state.get("model", state)
    encoder_weights = {k[len("encoder."):]: v for k, v in weights.items()
                       if k.startswith("encoder.")}
    if not encoder_weights:
        raise SystemExit(
            f"error: {checkpoint} has no 'encoder.' parameters. Is it a "
            "train_geometry.py checkpoint?")
    missing, unexpected = model.encoder.load_state_dict(encoder_weights, strict=False)
    print(f"  encoder initialised from {checkpoint}")
    print(f"    loaded {len(encoder_weights) - len(unexpected)} tensors"
          + (f", {len(missing)} left at ImageNet init" if missing else "")
          + (f", {len(unexpected)} unused" if unexpected else ""))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = args.results_dir / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"

    import pandas as pd
    try:
        frames = [load_segmentation_manifest(m, args.cache_root) for m in args.manifest]
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("image_path")
    if args.limit:
        frame = frame.head(args.limit)

    lower = frame["dataset"].str.lower()
    if args.eval_only:
        # No training set is needed or wanted: the checkpoint fixes what was
        # trained on, and naming it again here could only contradict it.
        if not args.val_datasets:
            print("error: --eval-only requires --val-datasets", file=sys.stderr)
            return 1
        val_frame = frame[lower.isin([d.lower() for d in args.val_datasets])]
        train_frame = val_frame.head(0)
        if val_frame.empty:
            print(f"error: no rows for {args.val_datasets}. Available: "
                  f"{sorted(frame['dataset'].unique())}", file=sys.stderr)
            return 1
    elif args.train_datasets or args.val_datasets:
        # C3: the split is by source, not by row, so the two sets are disjoint by
        # construction and "cross-domain" means what it says.
        train_frame = frame[lower.isin([d.lower() for d in (args.train_datasets or [])])]
        val_frame = frame[lower.isin([d.lower() for d in (args.val_datasets or [])])]
        if train_frame.empty or val_frame.empty:
            print(f"error: cross-domain split left {len(train_frame)} train / "
                  f"{len(val_frame)} val rows. Available datasets: "
                  f"{sorted(frame['dataset'].unique())}", file=sys.stderr)
            return 1
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(frame))
        cut = max(1, int(round(len(frame) * args.val_frac)))
        val_frame = frame.iloc[order[:cut]]
        train_frame = frame.iloc[order[cut:]]

    train_frame = train_frame.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)

    print(f"{args.experiment}: {len(train_frame)} train / {len(val_frame)} val   device={device}")
    print(f"  train sources: {dict(train_frame['dataset'].value_counts())}")
    print(f"  val sources:   {dict(val_frame['dataset'].value_counts())}")
    presence = channel_presence(train_frame)
    print(f"  training images carrying each lesion: {presence}")
    thin = [n for n, c in presence.items() if c < 10]
    if thin:
        print(f"  WARNING: {thin} appear in under 10 training images. A Dice figure")
        print("           for those channels will not mean much; report it with the count.")

    train_set = SegmentationDataset(train_frame, args.image_size, train=True, seed=args.seed)
    val_set = SegmentationDataset(val_frame, args.image_size, train=False, seed=args.seed)
    # None under --eval-only: the training frame is deliberately empty there, and
    # a shuffling DataLoader over zero rows raises before the eval branch is reached.
    train_loader = None if args.eval_only else DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")

    model = LesionSegmenter(pretrained=args.pretrained).to(device)
    if args.encoder_from:
        load_encoder(model, args.encoder_from, device)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("  encoder frozen; only the decoder trains")

    criterion = SegmentationLoss(dice_weight=args.dice_weight,
                                 bce_weight=1.0 - args.dice_weight,
                                 pos_weight=args.pos_weight)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda e: lr_lambda(e, args.warmup_epochs, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = Config(
        manifest=str([str(m) for m in args.manifest]), experiment=args.experiment,
        image_size=args.image_size, batch_size=args.batch_size, epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, lr=args.lr, weight_decay=args.weight_decay,
        dice_weight=args.dice_weight, pos_weight=args.pos_weight,
        train_datasets=args.train_datasets, val_datasets=args.val_datasets,
        val_frac=args.val_frac, pretrained=args.pretrained,
        encoder_from=str(args.encoder_from) if args.encoder_from else None,
        freeze_encoder=args.freeze_encoder, patience=args.patience,
        seed=args.seed, amp=args.amp)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    started = time.time()
    if args.eval_only:
        state = torch.load(args.eval_only, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        trained_on = state.get("config", {}).get("manifest", "unrecorded")
        print(f"  evaluating {args.eval_only.name} on "
              f"{sorted(val_frame['dataset'].unique())}: {len(val_set)} images")
        print(f"  that checkpoint was trained from: {trained_on}")
        result = evaluate(model, val_loader, criterion, device, args.amp)
        print()
        print(format_report(result))

        summary = {
            "experiment": args.experiment,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "eval_only": True,
            "checkpoint": str(args.eval_only),
            "checkpoint_trained_on": trained_on,
            "best_mean_dice_present": result["mean_dice_present"],
            "best_val": result,
            "train_images": 0,
            "val_images": len(val_set),
            "val_datasets": sorted(val_frame["dataset"].unique()),
            "minutes": round((time.time() - started) / 60, 1),
            "config": asdict(config),
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
        print(f"\n  wrote {out_dir}")
        return 0

    start_epoch, best_dice, best_epoch, history = 0, -math.inf, -1, []
    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        check_resume_config(state.get("config", {}), asdict(config))
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_dice, best_epoch, history = state["best_dice"], state["best_epoch"], state["history"]
        print(f"  resumed from epoch {state['epoch']} (best mean Dice {best_dice:.4f})")

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        stats = train_one_epoch(model, train_loader, criterion, optimiser, scaler,
                                device, args.amp)
        val = evaluate(model, val_loader, criterion, device, args.amp)
        scheduler.step()

        history.append({"epoch": epoch, "train": stats, "val": val,
                        "seconds": round(time.time() - epoch_start, 1)})
        per = val["per_lesion"]
        # SHORT_LABELS, not the first two letters of LESION_NAMES: haemorrhage and
        # hard_exudate both truncate to "HA", so two different channels would print
        # under one label and a collapsed channel could hide behind a healthy one.
        print(f"  epoch {epoch:3d}  train {stats['loss']:.4f}  val {val['loss']:.4f}  "
              f"Dice {val['mean_dice_present']:.4f}  ["
              + " ".join(f"{abbr} {per[n]['dice_present']:.3f}"
                         if per[n]['dice_present'] == per[n]['dice_present'] else f"{abbr} --"
                         for abbr, n in zip(SHORT_LABELS, LESION_NAMES))
              + f"]  ({history[-1]['seconds']:.0f}s)", flush=True)

        if val["mean_dice_present"] > best_dice:
            best_dice, best_epoch = val["mean_dice_present"], epoch
            torch.save({"model": model.state_dict(), "config": asdict(config),
                        "epoch": epoch, "val": val}, out_dir / "best.pt")

        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_dice": best_dice, "best_epoch": best_epoch,
                    "history": history, "config": asdict(config)}, checkpoint_path)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if epoch - best_epoch >= args.patience:
            print(f"  early stop: no improvement in {args.patience} epochs")
            break

    best = next((h["val"] for h in history if h["epoch"] == best_epoch), None)
    summary = {
        "experiment": args.experiment,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epochs_run": len(history), "best_epoch": best_epoch,
        "best_mean_dice_present": best_dice, "best_val": best,
        "minutes": round((time.time() - started) / 60, 1),
        "config": asdict(config),
        "train_images": len(train_set), "val_images": len(val_set),
        "train_channel_presence": presence,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  best epoch {best_epoch}: mean Dice {best_dice:.4f}")
    if best:
        print(format_report(best))
    print(f"  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
