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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from verify_dr.data.geometry import (  # noqa: E402
    GeometryDataset, load_geometry_manifest, patient_split,
)
from verify_dr.evaluation.geometry_metrics import (  # noqa: E402
    C1_GATE_DD, geometry_metrics,
)
from verify_dr.models.evidence import (  # noqa: E402
    GeometryHeatmapModel, GeometryModel,
)


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
    preds, truths, losses, indices = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(image)
            losses.append(float(criterion(out, target).detach()))
        # A heatmap model's forward gives logits, not coordinates; the metrics and
        # the per-image dump both want coordinates.
        coords = model.coordinates(out) if hasattr(model, "coordinates") else out
        preds.append(coords.float().detach().cpu().numpy())
        truths.append(target.float().cpu().numpy())
        indices.append(batch["index"].cpu().numpy())

    metrics = geometry_metrics(np.concatenate(preds), np.concatenate(truths), image_size)
    metrics["loss"] = float(np.mean(losses))
    # Kept for the per-image dump. C1's mean error is dominated by a heavy tail
    # -- the OD mean ran 2.3x its median on the first real run -- and an
    # aggregate cannot say whether that is many images slightly off or a few
    # catastrophically wrong. Those need different fixes, so record which.
    metrics["_preds"] = np.concatenate(preds)
    metrics["_truths"] = np.concatenate(truths)
    metrics["_indices"] = np.concatenate(indices)
    return metrics


def write_val_errors(path, dataset, arrays, image_size: int, disc_px: float) -> None:
    """Per-image errors for the best epoch, so the tail can be inspected.

    C1's gate is a mean over both landmarks, and a mean says nothing about
    whether a miss is many images slightly off or a few placed somewhere else
    entirely. Those have different causes and different fixes, so every
    validation image gets a row: its predicted and true coordinates in pixels,
    its error per landmark, and that error in disc diameters.
    """
    import csv

    preds, truths = arrays["_preds"], arrays["_truths"]
    indices = arrays["_indices"]
    frame = dataset.frame

    rows = []
    for row, (pred, truth, index) in enumerate(zip(preds, truths, indices)):
        pred_px, truth_px = pred * image_size, truth * image_size
        od_err = float(np.hypot(*(pred_px[0:2] - truth_px[0:2])))
        fov_err = float(np.hypot(*(pred_px[2:4] - truth_px[2:4])))
        record = {
            "image": Path(str(frame.at[int(index), "image_path"])).stem,
            "od_error_px": round(od_err, 1),
            "fovea_error_px": round(fov_err, 1),
            "od_error_dd": round(od_err / disc_px, 3) if disc_px else "",
            "fovea_error_dd": round(fov_err / disc_px, 3) if disc_px else "",
            "od_pred_x": round(float(pred_px[0]), 1), "od_pred_y": round(float(pred_px[1]), 1),
            "od_true_x": round(float(truth_px[0]), 1), "od_true_y": round(float(truth_px[1]), 1),
            "fovea_pred_x": round(float(pred_px[2]), 1), "fovea_pred_y": round(float(pred_px[3]), 1),
            "fovea_true_x": round(float(truth_px[2]), 1), "fovea_true_y": round(float(truth_px[3]), 1),
        }
        # Is the disc predicted on the wrong side of the fovea? In a fundus image
        # the disc sits nasal to the macula, so a sign flip here is a laterality
        # error rather than an imprecise one, and would be the single biggest
        # clue the tail has one systematic cause.
        record["side_flipped"] = int(
            np.sign(truth_px[0] - truth_px[2]) != np.sign(pred_px[0] - pred_px[2]))
        rows.append(record)

    rows.sort(key=lambda r: -r["od_error_px"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class HeatmapLoss(nn.Module):
    """MSE against Gaussian targets, which is what makes the two modes learnable.

    Regressing a coordinate under SmoothL1 forces one number per landmark, so a
    bimodal target (disc left for one eye, right for the other) can only be met
    by committing to a mode -- and C1's first run was wrong on 12% of images for
    exactly that reason. A per-pixel loss instead asks "is the disc here?" at
    every location, so both modes can be represented and inference picks the
    stronger by argmax.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        maps = self.model.target_heatmaps(target, logits.shape[-1])
        return F.mse_loss(torch.sigmoid(logits), maps)


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


# "head" belongs here, not in TRAJECTORY_KEYS: a heatmap checkpoint and a
# regression checkpoint have different parameter sets entirely, and resuming
# one into the other is the silent-success failure the Phase 3 guard exists for.
ARCHITECTURE_KEYS = ("image_size", "head")
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
    head: str
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
    p.add_argument("--head", choices=("heatmap", "regress"), default="heatmap",
                   help="heatmap (default) localises by peak, which is what lets the "
                        "disc sit on either side of the fovea. regress is the original "
                        "coordinate head, kept so C1's first result stays reproducible.")
    p.add_argument("--heatmap-sigma", type=float, default=2.0,
                   help="Gaussian sigma in heatmap pixels.")
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

    if args.head == "heatmap":
        model = GeometryHeatmapModel(pretrained=args.pretrained,
                                     sigma=args.heatmap_sigma).to(device)
        criterion = HeatmapLoss(model)
        print(f"  head: heatmap  {args.image_size // GeometryHeatmapModel.STRIDE} px grid, "
              f"sigma {args.heatmap_sigma}")
    else:
        model = GeometryModel(pretrained=args.pretrained, dropout=args.dropout).to(device)
        # Smooth L1 per docs/03: quadratic near zero so fine positioning still gets a
        # gradient, linear in the tail so a badly-marked centre cannot dominate.
        criterion = nn.SmoothL1Loss(beta=0.05)
        print("  head: coordinate regression")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda e: lr_lambda(e, args.warmup_epochs, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = Config(
        manifest=str(args.manifest), experiment=args.experiment, image_size=args.image_size,
        head=args.head,
        batch_size=args.batch_size, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        val_frac=args.val_frac, pretrained=args.pretrained, patience=args.patience,
        seed=args.seed, amp=args.amp,
    )

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


    # config.json is written AFTER the resume guard, not before.
    # Written first, a REFUSED resume still overwrote it: C1_geometry kept the
    # heatmap run's best.pt, metrics.json and val_errors.csv while its config
    # claimed head=regress. A config that misdescribes the artefacts beside it
    # is worse than none -- it reads as authoritative.
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2),
                                        encoding="utf-8")
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, scaler,
                                     device, args.amp)
        val = evaluate(model, val_loader, criterion, device, args.amp, args.image_size)
        scheduler.step()

        arrays = {k: val.pop(k) for k in ("_preds", "_truths", "_indices")}
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
            write_val_errors(out_dir / "val_errors.csv", val_set, arrays,
                             args.image_size, val["mean_disc_diameter_px"])

        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_dd": best_dd, "best_epoch": best_epoch,
                    "history": history, "config": asdict(config)}, checkpoint_path)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if epoch - best_epoch >= args.patience:
            print(f"  early stop: no improvement in {args.patience} epochs")
            break

    best = next((h["val"] for h in history if h["epoch"] == best_epoch), None)

    # The per-image dump is written when a new best appears, and a resumed run can
    # legitimately finish without one -- it restarts past the early-stop point and
    # exits on the patience check. Re-derive it from best.pt so the diagnostic
    # exists whether the run was fresh or resumed.
    errors_csv = out_dir / "val_errors.csv"
    best_pt = out_dir / "best.pt"
    if best and best_pt.exists() and not errors_csv.exists():
        print("  re-deriving val_errors.csv from best.pt", flush=True)
        state = torch.load(best_pt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        final = evaluate(model, val_loader, criterion, device, args.amp, args.image_size)
        arrays = {k: final.pop(k) for k in ("_preds", "_truths", "_indices")}
        write_val_errors(errors_csv, val_set, arrays, args.image_size,
                         best["mean_disc_diameter_px"])

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

    # Median beside mean. The gate is the mean, as pre-specified -- this is a
    # diagnostic, not a second criterion, and must never be swapped in because
    # it reads better.
    if best and best.get("mean_disc_diameter_px"):
        disc = best["mean_disc_diameter_px"]
        med = (best["od_error_px_median"] + best["fovea_error_px_median"]) / 2 / disc
        print(f"\n  median error {med:.3f} DD   (gate is on the MEAN: "
              f"{best_dd:.3f} DD)")
        if med < 0.5 <= best_dd:
            print("  The median passes and the mean does not, so the miss is a heavy")
            print("  tail: most images are located well and a minority are far out.")
            print("  See val_errors.csv, sorted worst OD error first. The gate still")
            print("  FAILS -- the criterion was fixed before the run.")

    errors_csv = out_dir / "val_errors.csv"
    if errors_csv.exists():
        import csv as _csv
        with open(errors_csv, encoding="utf-8") as handle:
            rows = list(_csv.DictReader(handle))
        flipped = sum(int(r["side_flipped"]) for r in rows)
        print(f"  disc predicted on the wrong side of the fovea: {flipped}/{len(rows)}")
        if flipped:
            worst = [r for r in rows if int(r["side_flipped"])][:3]
            print("    e.g. " + ", ".join(f"{r['image']} ({r['od_error_px']}px)"
                                          for r in worst))

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
