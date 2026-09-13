#!/usr/bin/env python3
"""Train the M1 grading pathway (Phase 3, experiments B1-B7).

Reads a manifest produced by build_variants.py, trains on its `train` rows and
selects on its `val` rows. Test rows are never read here -- selection happens on
validation only (docs/02_research_protocol.md Rule 2).

Built for Kaggle's ~12 hour session cap: a checkpoint is written every epoch and
--resume continues from it, so a killed session costs one epoch rather than the
whole run.

Examples
--------
    # B1: what resolution do we need?
    python scripts/train_grading.py --manifest manifests/eyepacs_balanced_1000.csv \\
        --experiment B1_res512 --image-size 512 --epochs 10

    # B5: does eye-pair fusion help?
    python scripts/train_grading.py --manifest manifests/eyepacs_full.csv \\
        --experiment B5_fusion --eye-pair-fusion --seed 42
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
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from verify_dr.data.dataset import (  # noqa: E402
    SAMPLERS, GradingDataset, grade_counts, load_manifest, make_sampler,
)
from verify_dr.evaluation.metrics import grading_metrics  # noqa: E402
from verify_dr.models.grading import BACKBONES, GradingModel  # noqa: E402
from verify_dr.models.losses import GradingLoss  # noqa: E402

HEADS = ("ordinal_focal", "ordinal", "softmax_ce")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(epoch: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay, evaluated per epoch."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool) -> Dict[str, object]:
    model.eval()
    grades, preds, probs, rdr, vtdr, losses = [], [], [], [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        fellow = batch.get("fellow")
        fellow = fellow.to(device, non_blocking=True) if fellow is not None else None
        target = batch["grade"].to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(image, fellow)
            losses.append(float(criterion(out, target)["loss"].detach()))

        grades.append(target.cpu().numpy())
        preds.append(out["grade"].cpu().numpy())
        probs.append(out["probs"].float().cpu().numpy())
        rdr.append(torch.sigmoid(out["rdr"].float()).cpu().numpy())
        vtdr.append(torch.sigmoid(out["vtdr"].float()).cpu().numpy())

    metrics = grading_metrics(
        np.concatenate(grades), np.concatenate(preds), np.concatenate(probs),
        np.concatenate(rdr), np.concatenate(vtdr),
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_epoch(model, loader, criterion, optimiser, scaler, device, amp: bool) -> Dict[str, float]:
    model.train()
    totals, seen = {"loss": 0.0, "main": 0.0, "rdr": 0.0, "vtdr": 0.0}, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        fellow = batch.get("fellow")
        fellow = fellow.to(device, non_blocking=True) if fellow is not None else None
        target = batch["grade"].to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            parts = criterion(model(image, fellow), target)
        scaler.scale(parts["loss"]).backward()
        scaler.step(optimiser)
        scaler.update()

        n = target.size(0)
        seen += n
        for key in totals:
            totals[key] += float(parts[key].detach()) * n
    return {k: v / max(1, seen) for k, v in totals.items()}


# Differences that change what the network *is*. Resuming across one of these
# silently produces a hybrid experiment: the backbone is fully convolutional and
# globally pooled, so a 768 px checkpoint loads into a 384 px run without error.
ARCHITECTURE_KEYS = ("backbone", "head", "image_size", "eye_pair_fusion")
# Differences that only change the trajectory from here on. Legal, but the thesis
# has to be able to say it happened (docs/02_research_protocol.md Rule 5).
TRAJECTORY_KEYS = ("lr", "weight_decay", "dropout", "focal_gamma", "sampler",
                   "epochs", "batch_size", "epoch_samples", "manifest", "seed")


def check_resume_config(stored: Dict[str, object], current: Dict[str, object]) -> None:
    """Raise on an architecture change, warn on a trajectory change."""
    fatal = [(k, stored.get(k), current[k]) for k in ARCHITECTURE_KEYS
             if stored.get(k) != current[k]]
    if fatal:
        lines = "\n".join(f"    {k}: checkpoint={old!r}  now={new!r}" for k, old, new in fatal)
        raise SystemExit(
            "error: --resume against a checkpoint built with a different architecture:\n"
            f"{lines}\n"
            "  The weights would load without complaint and the run would be a hybrid.\n"
            "  Use a new --experiment id, or drop --resume to start clean."
        )
    drifted = [(k, stored.get(k), current[k]) for k in TRAJECTORY_KEYS
               if stored.get(k) != current[k]]
    for k, old, new in drifted:
        print(f"  note: {k} changed on resume ({old!r} -> {new!r}); record this as a deviation")


@dataclass
class Config:
    manifest: str
    experiment: str
    backbone: str
    head: str
    sampler: str
    image_size: int
    batch_size: int
    epochs: int
    warmup_epochs: int
    lr: float
    weight_decay: float
    dropout: float
    focal_gamma: float
    eye_pair_fusion: bool
    pretrained: bool
    epoch_samples: Optional[int]
    patience: int
    seed: int
    amp: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the M1 grading pathway.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--experiment", required=True, help="Id from docs/04_experiment_register.md.")
    p.add_argument("--results-dir", type=Path, default=Path("results/stage_b"))
    p.add_argument("--cache-root", type=Path, nargs="+", default=None,
                   help="Repath the manifest onto these cache roots. Needed whenever the "
                        "cache is mounted somewhere other than where Phase 2 saw it. "
                        "Pass several when the cache is split across published datasets "
                        "(a full build plus a top-up); each dataset resolves to the root "
                        "that holds it.")
    p.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b0")
    p.add_argument("--head", choices=HEADS, default="ordinal_focal")
    p.add_argument("--sampler", choices=SAMPLERS, default="stratified_exposure")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--epoch-samples", type=int, default=None,
                   help="Draws per epoch for stratified_exposure. Defaults to the train size.")
    p.add_argument("--patience", type=int, default=3, help="Early stop on val QWK.")
    p.add_argument("--eye-pair-fusion", action="store_true")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    p.add_argument("--workers", type=int, default=4)   # configs/base.yaml
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Cap rows per split (smoke tests).")
    p.add_argument("--resume", action="store_true", help="Continue from the last checkpoint.")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = args.results_dir / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"

    # ---- data -------------------------------------------------------------
    try:
        train_frame = load_manifest(args.manifest, "train", args.cache_root)
        val_frame = load_manifest(args.manifest, "val", args.cache_root)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.limit:
        train_frame = train_frame.head(args.limit)
        val_frame = val_frame.head(max(2, args.limit // 4))

    print(f"{args.experiment}: {len(train_frame)} train / {len(val_frame)} val   device={device}")
    print(f"  train grades: {grade_counts(train_frame)}")
    print(f"  val   grades: {grade_counts(val_frame)}")

    common = dict(image_size=args.image_size, eye_pair_fusion=args.eye_pair_fusion, seed=args.seed)
    train_set = GradingDataset(train_frame, train=True, **common)
    val_set = GradingDataset(val_frame, train=False, **common)

    sampler = make_sampler(train_set.grades, args.sampler, args.epoch_samples, args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    # ---- model ------------------------------------------------------------
    model = GradingModel(
        backbone=args.backbone, pretrained=args.pretrained, dropout=args.dropout,
        eye_pair_fusion=args.eye_pair_fusion, head=args.head,
    ).to(device)
    criterion = GradingLoss(head=args.head, focal_gamma=args.focal_gamma)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda e: lr_lambda(e, args.warmup_epochs, args.epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = Config(
        manifest=str(args.manifest), experiment=args.experiment, backbone=args.backbone,
        head=args.head, sampler=args.sampler, image_size=args.image_size,
        batch_size=args.batch_size, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        focal_gamma=args.focal_gamma, eye_pair_fusion=args.eye_pair_fusion,
        pretrained=args.pretrained, epoch_samples=args.epoch_samples,
        patience=args.patience, seed=args.seed, amp=args.amp,
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    start_epoch, best_qwk, best_epoch, history = 0, -math.inf, -1, []
    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        check_resume_config(state.get("config", {}), asdict(config))
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_qwk, best_epoch = state["best_qwk"], state["best_epoch"]
        history = state["history"]
        print(f"  resumed from epoch {state['epoch']} (best val QWK {best_qwk:.4f})")

    # ---- train ------------------------------------------------------------
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimiser, scaler, device, args.amp)
        val_metrics = evaluate(model, val_loader, criterion, device, args.amp)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimiser.param_groups[0]["lr"],
            "train": train_stats,
            "val": val_metrics,
            "seconds": round(time.time() - epoch_start, 1),
        }
        history.append(record)

        recall1 = val_metrics["per_class_recall"]["1"]
        print(
            f"  epoch {epoch:2d}  train_loss {train_stats['loss']:.4f}  "
            f"val_qwk {val_metrics['qwk']:.4f}  val_f1 {val_metrics['macro_f1']:.4f}  "
            f"g1_recall {recall1:.3f}  ({record['seconds']:.0f}s)",
            flush=True,
        )

        improved = val_metrics["qwk"] > best_qwk
        if improved:
            best_qwk, best_epoch = val_metrics["qwk"], epoch
            torch.save({"model": model.state_dict(), "config": asdict(config),
                        "epoch": epoch, "val": val_metrics}, out_dir / "best.pt")

        # Written every epoch, not at the end: a killed session must cost one
        # epoch, never the run.
        torch.save({
            "model": model.state_dict(), "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_qwk": best_qwk, "best_epoch": best_epoch,
            "history": history, "config": asdict(config),
        }, checkpoint_path)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if epoch - best_epoch >= args.patience:
            print(f"  early stop: no val QWK improvement in {args.patience} epochs")
            break

    best_record = next((r for r in history if r["epoch"] == best_epoch), None)
    summary = {
        "experiment": args.experiment,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_qwk": best_qwk,
        "best_val": best_record["val"] if best_record else None,
        "minutes": round((time.time() - started) / 60, 1),
        "config": asdict(config),
        "train_grades": grade_counts(train_frame),
        "val_grades": grade_counts(val_frame),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  best epoch {best_epoch}: val QWK {best_qwk:.4f}")
    if summary["best_val"]:
        best = summary["best_val"]
        print(f"  macro_f1 {best['macro_f1']:.4f}  mae {best['mae']:.4f}  "
              f"ece {best['ece']:.4f}  auroc_rdr {best.get('auroc_rdr', float('nan')):.4f}")
        print("  per-class F1:     "
              + "  ".join(f"{k}:{v:.2f}" for k, v in best["per_class_f1"].items()))
        print("  per-class recall: "
              + "  ".join(f"{k}:{v:.2f}" for k, v in best["per_class_recall"].items()))

        # B1 is decided here. Neither QWK nor grade-1 recall can decide it alone:
        # skipping grade 1 entirely still scores ~0.97 QWK, and predicting grade 1
        # for every image scores 1.000 grade-1 recall.
        if best["distinct_predictions"] == 1:
            print(f"  WARNING: the model predicts one grade for every image "
                  f"(distinct_predictions=1).")
            print("           Per-class recall is unreadable in this state and this run")
            print("           decides nothing. The model is undertrained, not answering B1.")
        elif best["per_class_f1"]["1"] < 0.05:
            print("  WARNING: grade-1 F1 is near zero - the model has learned to skip the")
            print("           class. Grade 1 is microaneurysms only, and skipping it still")
            print("           scores ~0.97 QWK because grade 1 sits one step from grade 0.")
            print("           Judge B1 on grade-1 F1, not on QWK.")
    print(f"  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
