#!/usr/bin/env python3
"""Train M2b, the optic-disc / fovea regressor (Phase 4, experiment C1).

Reads the IDRiD manifest, whose Part C centres prepare_manifest.py already
re-projected into cache coordinates, and fits ResNet18 + a 4-output head.

C1's gate is a **mean landmark error below 0.5 disc diameters**. Below that,
quadrant assignment is reliable enough for M3 to reason over; above it, M3 must
fall back to a count-only rule and the 4-2-1 logic is off the table. The script
applies the gate and says which way it landed.

    python scripts/train_geometry.py --manifest manifests/idrid.csv \\
        --experiment C1_geometry --cache-root /kaggle/input/.../cache512
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
from verify_dr.data.geometry import (  # noqa: E402
    GeometryDataset, load_geometry_manifest, patient_split,
)
from verify_dr.evaluation.geometry_metrics import (  # noqa: E402
    C1_GATE_DD, geometry_metrics,
)
from verify_dr.models.evidence import GeometryModel  # noqa: E402


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
def evaluate(model, loader, criterion, device, amp: bool, image_size: int) -> Dict[str, object]:
    model.eval()
    preds, truths, losses = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(image)
            losses.append(float(criterion(out, target).detach()))
        preds.append(out.float().cpu().numpy())
        truths.append(target.float().cpu().numpy())

    metrics = geometry_metrics(np.concatenate(preds), np.concatenate(truths), image_size)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_epoch(model, loader, criterion, optimiser, scaler, device, amp: bool) -> float:
    model.train()
    total, seen = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            loss = criterion(model(image), target)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        n = target.size(0)
        total += float(loss.detach()) * n
        seen += n
    return total / max(1, seen)


ARCHITECTURE_KEYS = ("image_size",)
TRAJECTORY_KEYS = ("lr", "weight_decay", "dropout", "epochs", "batch_size", "manifest", "seed")


def check_resume_config(stored: Dict[str, object], current: Dict[str, object]) -> None:
    fatal = [(k, stored.get(k), current[k]) for k in ARCHITECTURE_KEYS
             if stored.get(k) != current[k]]
    if fatal:
        lines = "\n".join(f"    {k}: checkpoint={o!r}  now={n!r}" for k, o, n in fatal)
        raise SystemExit(
            f"error: --resume against a checkpoint built differently:\n{lines}\n"
            "  Use a new --experiment id, or drop --resume to start clean."
        )
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
    dropout: float
    val_frac: float
    pretrained: bool
    patience: int
    seed: int
    amp: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the M2b optic-disc / fovea regressor (C1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--experiment", required=True)
    p.add_argument("--results-dir", type=Path, default=Path("results/stage_c"))
    p.add_argument("--cache-root", type=Path, nargs="+", default=None)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--cache-size", type=int, default=512,
                   help="Pixel size the manifest's coordinates are expressed in.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stop on val mean error in disc diameters.")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = args.results_dir / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"

    try:
        frame = load_geometry_manifest(args.manifest, args.cache_root)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.limit:
        frame = frame.head(args.limit)

    splits = patient_split(frame, args.val_frac, args.seed)
    print(f"{args.experiment}: {len(splits['train'])} train / {len(splits['val'])} val "
          f"  device={device}")

    common = dict(image_size=args.image_size, cache_size=args.cache_size, seed=args.seed)
    train_set = GeometryDataset(splits["train"], train=True, **common)
    val_set = GeometryDataset(splits["val"], train=False, **common)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")

    # The control. A constant predictor does respectably on stereotyped fundus
    # framing, so a model that cannot beat it has learned the average layout rather
    # than this image's landmarks.
    constant = np.repeat(train_set.targets.mean(axis=0, keepdims=True), len(val_set), axis=0)
    baseline = geometry_metrics(constant, val_set.targets, args.image_size)
    print(f"  baseline (predict the training mean): {baseline['mean_error_dd']:.3f} DD, "
          f"{baseline['od_error_px']:.1f} px OD / {baseline['fovea_error_px']:.1f} px fovea")

    model = GeometryModel(pretrained=args.pretrained, dropout=args.dropout).to(device)
    # Smooth L1 per docs/03: quadratic near zero so fine positioning still gets a
    # gradient, linear in the tail so a badly-marked centre cannot dominate.
    criterion = nn.SmoothL1Loss(beta=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda e: lr_lambda(e, args.warmup_epochs, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = Config(
        manifest=str(args.manifest), experiment=args.experiment, image_size=args.image_size,
        batch_size=args.batch_size, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        val_frac=args.val_frac, pretrained=args.pretrained, patience=args.patience,
        seed=args.seed, amp=args.amp,
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    start_epoch, best_dd, best_epoch, history = 0, math.inf, -1, []
    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        check_resume_config(state.get("config", {}), asdict(config))
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_dd, best_epoch, history = state["best_dd"], state["best_epoch"], state["history"]
        print(f"  resumed from epoch {state['epoch']} (best {best_dd:.3f} DD)")

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, scaler,
                                     device, args.amp)
        val = evaluate(model, val_loader, criterion, device, args.amp, args.image_size)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val": val,
                        "seconds": round(time.time() - epoch_start, 1)})
        print(f"  epoch {epoch:3d}  train {train_loss:.5f}  "
              f"val {val['loss']:.5f}  {val['mean_error_dd']:.3f} DD  "
              f"(OD {val['od_error_px']:.1f} px, fovea {val['fovea_error_px']:.1f} px)"
              f"  ({history[-1]['seconds']:.0f}s)", flush=True)

        if val["mean_error_dd"] < best_dd:
            best_dd, best_epoch = val["mean_error_dd"], epoch
            torch.save({"model": model.state_dict(), "config": asdict(config),
                        "epoch": epoch, "val": val}, out_dir / "best.pt")

        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_dd": best_dd, "best_epoch": best_epoch,
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
        "best_mean_error_dd": best_dd,
        "best_val": best, "baseline": baseline,
        "minutes": round((time.time() - started) / 60, 1),
        "config": asdict(config),
        "train_images": len(train_set), "val_images": len(val_set),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  best epoch {best_epoch}: {best_dd:.3f} disc diameters")
    if best:
        print(f"  OD    {best['od_error_px']:.1f} px  ({best['od_error_dd']:.3f} DD)")
        print(f"  fovea {best['fovea_error_px']:.1f} px  ({best['fovea_error_dd']:.3f} DD)")
        print(f"  within {C1_GATE_DD} DD: {best['within_half_dd'] * 100:.1f}% of landmarks")
        print(f"  mean disc diameter: {best['mean_disc_diameter_px']:.1f} px")

    print("\n" + "=" * 72)
    lift = baseline["mean_error_dd"] - best_dd
    if best_dd < C1_GATE_DD:
        print(f"C1 PASSES: {best_dd:.3f} DD < {C1_GATE_DD} gate.")
        print("Quadrant assignment is reliable enough for M3 to reason over, so the")
        print("partial 4-2-1 rule is available.")
    else:
        print(f"C1 FAILS: {best_dd:.3f} DD is at or above the {C1_GATE_DD} gate.")
        print("Quadrant assignment would be too noisy to reason over. M3 must fall back")
        print("to a count-only rule and the thesis has to say so -- docs/00_START_HERE.md")
        print("Phase 4 names this as the designed fallback, not a failure to hide.")
    print(f"Against the constant-predictor baseline ({baseline['mean_error_dd']:.3f} DD): "
          f"{lift:+.3f} DD")
    if lift <= 0:
        print("  WARNING: no better than predicting the training mean. The model has")
        print("           learned the average fundus layout, not this image's landmarks.")
    print("=" * 72)
    print(f"  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
