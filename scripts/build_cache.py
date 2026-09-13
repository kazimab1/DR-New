#!/usr/bin/env python3
"""Build the fixed-size VERIFY-DR image cache (Phase 1).

Fundus JPEGs are large and variably sized, so decoding them every epoch makes
JPEG decode -- not the GPU -- the training bottleneck. This script does the
decode once: retinal-field crop, square fit, resize, optional CLAHE, re-encode.
Epoch time on full EyePACS drops from ~25-40 min to ~7-9 min.

Run it CPU-only; it consumes no GPU quota. Save the output as a private Kaggle
dataset and never rebuild it mid-project -- every downstream number depends on
the cache being fixed.

Per image:
    retinal-field crop -> square fit (pad by default) -> resize to --size
    -> CLAHE on the LAB L channel -> JPEG at --quality

Masks, when passed with --mask, take the *image's* geometry (computed from the
image, never from the mask -- masks are mostly black and would crop to nothing),
use nearest-neighbour interpolation, skip CLAHE, and are written as lossless PNG.

Examples
--------
EyePACS, class-folder layout:

    python scripts/build_cache.py \
        --source-root /kaggle/input/eyepacs-original \
        --dataset EyePACS \
        --output-root /kaggle/working/cache512

DDR lesion-segmentation subset with its four annotated channels:

    python scripts/build_cache.py \
        --source-root /kaggle/input/ddr-dataset-credits-to-authors/lesion_segmentation/train/image \
        --dataset DDR \
        --output-root /kaggle/working/cache512 \
        --mask microaneurysm=/kaggle/input/ddr-dataset-credits-to-authors/lesion_segmentation/train/label/MA \
        --mask haemorrhage=/kaggle/input/ddr-dataset-credits-to-authors/lesion_segmentation/train/label/HE \
        --mask hard_exudate=/kaggle/input/ddr-dataset-credits-to-authors/lesion_segmentation/train/label/EX \
        --mask soft_exudate=/kaggle/input/ddr-dataset-credits-to-authors/lesion_segmentation/train/label/SE

Emits <output-root>/<dataset>/cache_report.json, which experiment A0 checks, and
optionally a contact sheet for A0's visual audit of random crops.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm"}
LESION_CHANNELS = ("microaneurysm", "haemorrhage", "hard_exudate", "soft_exudate")
# IDRiD Part A also ships optic-disc masks. Not a lesion, but caching it gives M2b
# a stronger geometry signal than centre coordinates alone, so accept it as a channel.
MASK_CHANNELS = LESION_CHANNELS + ("optic_disc",)
MAX_RECORDED_FAILURES = 50

# cv2 spawns its own thread pool per process, which fights the process pool.
cv2.setNumThreads(1)


# --------------------------------------------------------------------------- IO


def imread(path: Path, unchanged: bool = False) -> Optional[np.ndarray]:
    """Read an image, tolerating non-ASCII paths that break cv2.imread."""
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flag)
    if img is not None:
        return img
    # cv2 has no TIFF support in some minimal builds; PIL usually does.
    try:
        from PIL import Image

        with Image.open(path) as handle:
            arr = np.array(handle.convert("L" if unchanged else "RGB"))
    except Exception:
        return None
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def imwrite(path: Path, img: np.ndarray, params: Sequence[int]) -> bool:
    """Write atomically so an interrupted run never leaves a truncated file.

    A half-written JPEG would be skipped as 'already done' by the resume logic
    on the next run, silently poisoning the cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img, list(params))
    if not ok:
        return False
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        buf.tofile(str(tmp))
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        return False
    return True


# -------------------------------------------------------------------- geometry


def retinal_bbox(bgr: np.ndarray, tol_scale: float) -> Tuple[Tuple[int, int, int, int], bool]:
    """Bounding box of the illuminated retinal disc.

    Returns (x0, y0, x1, y1) and whether a real crop was found. On failure the
    box is the full frame -- a loose crop beats dropping an image -- and the
    fallback is counted so A0 can check the rate.

    The threshold is set above the *measured* surround level rather than from a
    fixed floor. Fundus JPEGs have a compression noise floor in the black
    surround (routinely reaching 15-20/255), and a fixed low cutoff lets that
    noise widen the box to the whole frame -- a crop that silently does nothing
    while still looking successful.
    """
    height, width = bgr.shape[:2]
    full = (0, 0, width, height)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Threshold on a downsampled, median-filtered copy: averaging kills the
    # speckle that defeats a per-pixel threshold, and it is far cheaper on the
    # 3000px+ originals.
    scale = min(1.0, 512.0 / max(height, width))
    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else gray
    )
    small = cv2.medianBlur(small, 5)
    sh, sw = small.shape

    # Corner patches are surround in any uncropped fundus image. In an already
    # cropped one they are retina, which drives the threshold high, collapses
    # the mask, and trips the area guard below -- the right answer there.
    ch, cw = max(2, sh // 8), max(2, sw // 8)
    corners = np.concatenate([
        small[:ch, :cw].ravel(), small[:ch, -cw:].ravel(),
        small[-ch:, :cw].ravel(), small[-ch:, -cw:].ravel(),
    ])
    background = float(np.median(corners))
    tol = max(background + 10.0, float(small.mean()) * tol_scale, 7.0)

    lit = small > tol
    if not lit.any():
        return full, False

    # A row/column must be >2% lit to count, so a bright artefact or a burned-in
    # label cannot stretch the box on its own.
    rows = np.flatnonzero(lit.sum(axis=1) > sw * 0.02)
    cols = np.flatnonzero(lit.sum(axis=0) > sh * 0.02)
    if rows.size == 0 or cols.size == 0:
        return full, False

    inv = 1.0 / scale
    y0, y1 = int(rows[0] * inv), min(height, int((rows[-1] + 1) * inv))
    x0, x1 = int(cols[0] * inv), min(width, int((cols[-1] + 1) * inv))
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return full, False

    # Under a fifth of the frame means the threshold latched onto something that
    # is not the retina.
    if (x1 - x0) * (y1 - y0) < 0.20 * width * height:
        return full, False

    # A box within 2% of the full frame on both axes is not a crop. Report it as
    # a fallback so the A0 rate reflects reality rather than counting a no-op as
    # a success.
    if (x1 - x0) > 0.98 * width and (y1 - y0) > 0.98 * height:
        return full, False

    return (x0, y0, x1, y1), True


def fit_square(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    size: int,
    mode: str,
    interp: int,
) -> np.ndarray:
    """Crop to bbox, make it square, resize to size x size."""
    x0, y0, x1, y1 = bbox
    out = img[y0:y1, x0:x1]
    height, width = out.shape[:2]

    if mode == "pad":
        side = max(height, width)
        top = (side - height) // 2
        left = (side - width) // 2
        out = cv2.copyMakeBorder(
            out, top, side - height - top, left, side - width - left,
            cv2.BORDER_CONSTANT, value=0,
        )
        return cv2.resize(out, (size, size), interpolation=interp)

    # mode == "crop": scale the short side up to `size`, then centre-crop.
    scale = size / min(height, width)
    new_w = max(size, int(round(width * scale)))
    new_h = max(size, int(round(height * scale)))
    out = cv2.resize(out, (new_w, new_h), interpolation=interp)
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return out[top:top + size, left:left + size]


def map_point(x, y, bbox, size, mode):
    """Map a point from source pixels into the cached image's frame.

    IDRiD Part C gives optic-disc and fovea centres in original coordinates, so
    the same crop/fit/resize that produced the cache has to be applied to them.
    Kept beside fit_square deliberately: if one changes the other must too.
    """
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    px, py = x - x0, y - y0

    if mode == "pad":
        side = max(height, width)
        px += (side - width) // 2
        py += (side - height) // 2
        scale = size / side
        return px * scale, py * scale

    scale = size / min(height, width)
    new_w = max(size, int(round(width * scale)))
    new_h = max(size, int(round(height * scale)))
    px *= new_w / width
    py *= new_h / height
    return px - (new_w - size) // 2, py - (new_h - size) // 2


_clahe = None


def apply_clahe(bgr: np.ndarray, clip: float, tiles: int) -> np.ndarray:
    """Contrast-limited equalisation on luminance only, so hue is untouched."""
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ------------------------------------------------------------------ per-image


@dataclass
class Result:
    rel: str
    status: str                      # done | skipped | failed
    cropped: bool = False
    out_bytes: int = 0
    masks: Dict[str, str] = field(default_factory=dict)
    error: str = ""


def find_mask(mask_dir: Path, stem: str) -> Optional[Path]:
    """Locate the mask for an image stem; annotators vary the suffix freely."""
    for suffix in (".tif", ".tiff", ".png", ".gif", ".bmp", ".jpg", ".jpeg"):
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(mask_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def process_one(rel: str, cfg: dict) -> Result:
    src = Path(cfg["source_root"]) / rel
    dst = Path(cfg["image_out"]) / rel
    dst = dst.with_suffix(".jpg")
    result = Result(rel=rel, status="failed")

    mask_targets = {
        channel: Path(cfg["mask_out"]) / channel / f"{Path(rel).stem}.png"
        for channel in cfg["masks"]
    }

    if not cfg["overwrite"]:
        image_ready = dst.exists() and dst.stat().st_size > 0
        masks_ready = all(
            path.exists() and path.stat().st_size > 0
            for channel, path in mask_targets.items()
            if find_mask(Path(cfg["masks"][channel]), Path(rel).stem) is not None
        )
        if image_ready and masks_ready:
            result.status = "skipped"
            result.out_bytes = dst.stat().st_size
            return result

    try:
        bgr = imread(src)
        if bgr is None:
            result.error = "unreadable"
            return result
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

        bbox, cropped = retinal_bbox(bgr, cfg["tol_scale"])
        result.cropped = cropped

        out = fit_square(bgr, bbox, cfg["size"], cfg["fit"], cv2.INTER_AREA)
        if cfg["clahe"]:
            out = apply_clahe(out, cfg["clahe_clip"], cfg["clahe_tiles"])

        if not imwrite(dst, out, (cv2.IMWRITE_JPEG_QUALITY, cfg["quality"])):
            result.error = "encode failed"
            return result
        result.out_bytes = dst.stat().st_size

        # Same bbox, same fit, nearest-neighbour, no CLAHE.
        for channel, mask_dir in cfg["masks"].items():
            found = find_mask(Path(mask_dir), Path(rel).stem)
            if found is None:
                result.masks[channel] = "absent"
                continue
            raw = imread(found, unchanged=True)
            if raw is None:
                result.masks[channel] = "unreadable"
                continue
            if raw.ndim == 3:
                raw = raw.max(axis=2)
            binary = np.where(raw > 0, 255, 0).astype(np.uint8)
            warped = fit_square(binary, bbox, cfg["size"], cfg["fit"], cv2.INTER_NEAREST)
            ok = imwrite(mask_targets[channel], warped, (cv2.IMWRITE_PNG_COMPRESSION, 6))
            result.masks[channel] = "done" if ok else "encode failed"

        result.status = "done"
        return result

    except Exception as exc:  # one bad file must never kill a ten-hour job
        result.error = f"{type(exc).__name__}: {exc}"
        return result


# ----------------------------------------------------------------- discovery


def discover(source_root: Path) -> List[str]:
    files = [
        p for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    ]
    return sorted(str(p.relative_to(source_root)) for p in files)


def contact_sheet(image_out: Path, count: int, seed: int, out_path: Path, thumb: int = 160) -> bool:
    """Grid of random cached crops, for A0's visual audit."""
    cached = sorted(image_out.rglob("*.jpg"))
    if not cached:
        return False
    picks = random.Random(seed).sample(cached, min(count, len(cached)))
    cols = 10
    rows = (len(picks) + cols - 1) // cols
    sheet = np.zeros((rows * thumb, cols * thumb, 3), dtype=np.uint8)
    for i, path in enumerate(picks):
        img = imread(path)
        if img is None:
            continue
        small = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        sheet[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = small
    return imwrite(out_path, sheet, (cv2.IMWRITE_JPEG_QUALITY, 85))


# ---------------------------------------------------------------------- main


def parse_masks(entries: Sequence[str]) -> Dict[str, str]:
    masks: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"--mask expects CHANNEL=DIR, got {entry!r}"
            )
        channel, _, directory = entry.partition("=")
        channel, directory = channel.strip(), directory.strip()
        if channel not in MASK_CHANNELS:
            raise argparse.ArgumentTypeError(
                f"unknown mask channel {channel!r}; expected one of {', '.join(MASK_CHANNELS)}"
            )
        path = Path(directory)
        if not path.is_dir():
            raise argparse.ArgumentTypeError(f"mask directory does not exist: {path}")
        masks[channel] = str(path)
    return masks


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed-size VERIFY-DR image cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", required=True, type=Path,
                        help="Directory scanned recursively for images.")
    parser.add_argument("--dataset", required=True,
                        choices=["EyePACS", "DDR", "IDRiD", "APTOS", "Messidor2"])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--fit", choices=["pad", "crop"], default="pad",
                        help="pad keeps the whole retinal field (needed for quadrant "
                             "reasoning); crop centre-crops and discards the periphery.")
    parser.add_argument("--clahe", dest="clahe", action="store_true", default=True)
    parser.add_argument("--no-clahe", dest="clahe", action="store_false",
                        help="Build an un-equalised cache, for a CLAHE ablation.")
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-tiles", type=int, default=8)
    parser.add_argument("--tol-scale", type=float, default=0.10,
                        help="Retinal-crop threshold as a fraction of mean intensity.")
    parser.add_argument("--mask", action="append", default=[], metavar="CHANNEL=DIR",
                        help=f"Repeatable. Channels: {', '.join(MASK_CHANNELS)}.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N images (smoke tests).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode files that already exist. Default resumes.")
    parser.add_argument("--contact-sheet", type=int, default=100, metavar="N",
                        help="Write a grid of N random crops for the A0 visual audit. 0 disables.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be processed, write nothing.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.source_root.is_dir():
        print(f"error: --source-root is not a directory: {args.source_root}", file=sys.stderr)
        return 2
    if not 1 <= args.quality <= 100:
        print("error: --quality must be in 1..100", file=sys.stderr)
        return 2
    if args.size < 32:
        print("error: --size must be at least 32", file=sys.stderr)
        return 2

    try:
        masks = parse_masks(args.mask)
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    dataset_dir = args.output_root / args.dataset.lower()
    image_out = dataset_dir / "images"
    mask_out = dataset_dir / "masks"

    print(f"scanning {args.source_root} ...", flush=True)
    rels = discover(args.source_root)
    if not rels:
        print(f"error: no images found under {args.source_root}", file=sys.stderr)
        return 1
    if args.limit:
        rels = rels[:args.limit]

    print(f"{args.dataset}: {len(rels)} images -> {image_out}")
    print(f"  size={args.size} fit={args.fit} quality={args.quality} "
          f"clahe={'on' if args.clahe else 'off'} workers={args.workers}")
    if masks:
        print(f"  masks: {', '.join(sorted(masks))}")
    if args.dry_run:
        for rel in rels[:10]:
            print(f"  would process {rel}")
        if len(rels) > 10:
            print(f"  ... and {len(rels) - 10} more")
        return 0

    cfg = {
        "source_root": str(args.source_root),
        "image_out": str(image_out),
        "mask_out": str(mask_out),
        "size": args.size,
        "quality": args.quality,
        "fit": args.fit,
        "clahe": args.clahe,
        "clahe_clip": args.clahe_clip,
        "clahe_tiles": args.clahe_tiles,
        "tol_scale": args.tol_scale,
        "masks": masks,
        "overwrite": args.overwrite,
    }

    counts = {"found": len(rels), "done": 0, "skipped": 0, "failed": 0}
    crop_ok = 0
    total_bytes = 0
    failures: List[Dict[str, str]] = []
    mask_stats = {c: {"written": 0, "absent": 0, "failed": 0} for c in masks}
    started = time.time()
    interrupted = False

    def record(res: Result) -> None:
        nonlocal crop_ok, total_bytes
        counts[res.status] = counts.get(res.status, 0) + 1
        if res.status == "done":
            if res.cropped:
                crop_ok += 1
            total_bytes += res.out_bytes
            for channel, state in res.masks.items():
                if state == "done":
                    mask_stats[channel]["written"] += 1
                elif state == "absent":
                    mask_stats[channel]["absent"] += 1
                else:
                    mask_stats[channel]["failed"] += 1
        elif res.status == "skipped":
            total_bytes += res.out_bytes
        elif res.status == "failed" and len(failures) < MAX_RECORDED_FAILURES:
            failures.append({"path": res.rel, "error": res.error})

    def progress(seen: int) -> None:
        elapsed = time.time() - started
        rate = seen / elapsed if elapsed > 0 else 0.0
        remaining = (len(rels) - seen) / rate if rate > 0 else 0.0
        print(f"  {seen}/{len(rels)}  {rate:.1f} img/s  "
              f"eta {remaining / 60:.1f} min  failed={counts['failed']}", flush=True)

    try:
        if args.workers <= 1:
            for i, rel in enumerate(rels, 1):
                record(process_one(rel, cfg))
                if i % 500 == 0 or i == len(rels):
                    progress(i)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(process_one, rel, cfg): rel for rel in rels}
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        record(future.result())
                    except Exception as exc:
                        counts["failed"] += 1
                        if len(failures) < MAX_RECORDED_FAILURES:
                            failures.append({"path": futures[future], "error": repr(exc)})
                    if i % 500 == 0 or i == len(rels):
                        progress(i)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted -- writing a partial report; re-run to resume", flush=True)

    processed = counts["done"] + counts["skipped"]
    elapsed = time.time() - started

    # Crop statistics are only observable for images this run actually decoded, so
    # a resumed run that skips everything would otherwise overwrite the report with
    # nulls and destroy the A0 evidence. Resuming across dead sessions is the normal
    # Kaggle workflow, so accumulate across runs instead.
    dataset_dir.mkdir(parents=True, exist_ok=True)
    report_path = dataset_dir / "cache_report.json"
    prior: Dict = {}
    if report_path.exists() and not args.overwrite:
        try:
            prior = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = {}
    prior_crop = prior.get("crop") or {}
    detected = int(prior_crop.get("detected") or 0) + crop_ok
    fallback = int(prior_crop.get("fallback_full_frame") or 0) + (counts["done"] - crop_ok)
    measured = detected + fallback

    report = {
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interrupted": interrupted,
        "source_root": str(args.source_root),
        "output_dir": str(dataset_dir),
        "params": {
            "size": args.size, "fit": args.fit, "quality": args.quality,
            "clahe": args.clahe, "clahe_clip": args.clahe_clip,
            "clahe_tiles": args.clahe_tiles, "tol_scale": args.tol_scale,
        },
        "runs": int(prior.get("runs") or 0) + 1,
        "counts": counts,                      # this run only
        "cached": processed,                   # images now in the cache
        "crop": {                              # cumulative across resumed runs
            "detected": detected,
            "fallback_full_frame": fallback,
            "images_measured": measured,
            # A0 gate: this must stay under 0.005.
            "fallback_rate": round(fallback / measured, 5) if measured else None,
        },
        "output": {
            "total_bytes": total_bytes,
            "mean_bytes": round(total_bytes / processed, 1) if processed else 0,
            "total_mib": round(total_bytes / 1048576, 1),
        },
        "masks": mask_stats,
        "elapsed_seconds": round(elapsed, 1),
        "failures": failures,
        "failures_truncated": counts["failed"] > len(failures),
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{args.dataset}: done={counts['done']} skipped={counts['skipped']} "
          f"failed={counts['failed']}  in {elapsed / 60:.1f} min")
    print(f"  cached: {processed}/{counts['found']} images  (run {report['runs']})")
    rate = report["crop"]["fallback_rate"]
    if rate is not None:
        flag = "" if rate <= 0.005 else "   <-- A0 THRESHOLD 0.005 EXCEEDED"
        print(f"  crop fallback rate: {rate:.4f} over {measured} measured{flag}")
    print(f"  cache size: {report['output']['total_mib']} MiB "
          f"(mean {report['output']['mean_bytes']:.0f} B/image)")
    for channel, stat in mask_stats.items():
        print(f"  mask {channel}: written={stat['written']} "
              f"absent={stat['absent']} failed={stat['failed']}")
    if failures:
        print(f"  first failures ({len(failures)} of {counts['failed']} recorded):")
        for item in failures[:5]:
            print(f"    {item['path']}: {item['error']}")
    print(f"  report: {report_path}")

    if args.contact_sheet and not interrupted:
        sheet_path = dataset_dir / "contact_sheet.jpg"
        if contact_sheet(image_out, args.contact_sheet, args.seed, sheet_path):
            print(f"  contact sheet: {sheet_path}  (A0: eyeball this)")

    if interrupted:
        return 130
    return 1 if counts["failed"] and not counts["done"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
