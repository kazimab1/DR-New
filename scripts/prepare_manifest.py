#!/usr/bin/env python3
"""Turn each source's native labelling into a VERIFY-DR manifest (Phase 2).

Reads the 512 px cache built by build_cache.py and each dataset's own labels,
and emits one CSV per dataset following the contract in docs/05_dataset_card.md.

    image_path  grade  dataset  patient_id  eye  source_split  <channel>_mask

Deliberately does NOT assign the `split` column. Splitting is patient-grouped and
belongs to build_variants.py, which owns the disjointness assertion. What is
recorded here is `source_split`: whatever split the mirror itself shipped, kept
for reference only. A mirror's split can put one patient's two eyes on opposite
sides (00_verify_inputs.ipynb Q5 tests this), so adopting it would leak.

Grades and paths
----------------
EyePACS    grade from a 0-4 class-folder component at any depth; patient and eye
           from the <patient>_<left|right> filename.
DDR        grade from DR_grading/{train,valid,test}.txt, lines "<image> <grade>".
           Grade 5 is 'ungradable' and is routed to the quality manifest, never
           the grading manifest.
IDRiD      grade from the Part B CSV. Optional Part C optic-disc and fovea
           centres are re-projected into cache coordinates (see --coords).
APTOS      id_code / diagnosis columns of train.csv.
Messidor2  images and adjudicated grades ship as two separate Kaggle datasets;
           they are joined here on image id.

Examples
--------
    python scripts/prepare_manifest.py --dataset EyePACS \\
        --cache-root /kaggle/working/cache512 --folder-labels \\
        --output manifests/eyepacs_manifest.csv

    python scripts/prepare_manifest.py --dataset DDR \\
        --cache-root /kaggle/working/cache512 \\
        --labels '/kaggle/input/.../DR_grading/train.txt' \\
        --labels '/kaggle/input/.../DR_grading/valid.txt' \\
        --output manifests/ddr_manifest.csv \\
        --quality-output manifests/quality_manifest.csv

    python scripts/prepare_manifest.py --dataset IDRiD \\
        --cache-root /kaggle/working/cache512 \\
        --labels '/kaggle/input/.../IDRiD_Disease Grading_Training Labels.csv' \\
        --coords '/kaggle/input/.../IDRiD_Fovea_Center_Training Set_Markups.csv' \\
        --coords-source-dir '/kaggle/input/.../1. Original Images/a. Training Set' \\
        --output manifests/idrid_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The package too, for verify_dr.data.idrid_tables. The training scripts already
# do this; this script did not, and only needed it once it grew a package import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_cache import IMAGE_SUFFIXES, LESION_CHANNELS, MASK_CHANNELS  # noqa: E402

DATASETS = ("EyePACS", "DDR", "IDRiD", "APTOS", "Messidor2")
GRADE_DIRS = {"0", "1", "2", "3", "4"}
SPLIT_DIRS = {"train", "training", "val", "valid", "validation", "test", "testing"}
EYEPACS_STEM = re.compile(r"^(\d+)_(left|right)$", re.IGNORECASE)
DDR_UNGRADABLE = 5


# --------------------------------------------------------------------- helpers


def canon_split(name: str) -> Optional[str]:
    low = name.lower().strip()
    low = re.sub(r"^[a-z0-9]+\.\s*", "", low)          # "a. Training Set" -> "training set"
    if "train" in low:
        return "train"
    if "val" in low:
        return "val"
    if "test" in low:
        return "test"
    return None


def path_component(rel: Path, allowed) -> Optional[str]:
    """First directory component of rel that is in `allowed` (case-insensitive)."""
    for part in rel.parts[:-1]:
        if part.lower() in allowed:
            return part.lower()
    return None


def cached_images(cache_root: Path, dataset: str) -> Dict[str, Path]:
    """Map image stem -> cached path. Stems are unique within a dataset cache."""
    root = cache_root / dataset.lower() / "images"
    if not root.is_dir():
        raise FileNotFoundError(
            f"no cached images at {root} -- run build_cache.py for {dataset} first"
        )
    found: Dict[str, Path] = {}
    collisions = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            if p.stem in found:
                collisions += 1
            found[p.stem] = p
    if collisions:
        print(f"  warning: {collisions} duplicate stems in the cache; last one wins")
    return found


def cached_masks(cache_root: Path, dataset: str) -> Dict[str, Dict[str, Path]]:
    """channel -> {stem: mask path} for whatever build_cache wrote."""
    root = cache_root / dataset.lower() / "masks"
    out: Dict[str, Dict[str, Path]] = {}
    if not root.is_dir():
        return out
    for channel_dir in sorted(root.iterdir()):
        if not channel_dir.is_dir() or channel_dir.name not in MASK_CHANNELS:
            continue
        out[channel_dir.name] = {
            p.stem: p for p in channel_dir.rglob("*") if p.is_file()
        }
    return out


def read_label_table(path: Path, id_col: Optional[str], grade_col: Optional[str]) -> pd.DataFrame:
    """Read a label file into columns ['id', 'grade'].

    DDR ships whitespace-separated .txt ("<image> <grade>"); the others ship CSV
    or Excel. Column names vary between mirrors, so they are guessed unless the
    caller names them.
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".lst"}:
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=["id", "grade"],
                            engine="python", comment="#")
        return frame

    frame = pd.read_excel(path) if suffix in {".xlsx", ".xls"} else pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]

    def pick(explicit, candidates, fallback_index):
        if explicit:
            if explicit not in frame.columns:
                raise KeyError(f"{path.name}: no column {explicit!r} in {list(frame.columns)}")
            return explicit
        for col in frame.columns:
            if any(k in col.lower() for k in candidates):
                return col
        return frame.columns[fallback_index]

    id_name = pick(id_col, ("image", "id_code", "id", "name", "code"), 0)
    grade_name = pick(grade_col, ("retinopathy", "grade", "diagnosis", "level", "dr"), 1)
    if id_name == grade_name:
        raise KeyError(f"{path.name}: could not separate id and grade columns "
                       f"({list(frame.columns)}); pass --id-col/--grade-col")
    print(f"  {path.name}: id={id_name!r} grade={grade_name!r}")
    return frame[[id_name, grade_name]].rename(columns={id_name: "id", grade_name: "grade"})


def patient_and_eye(dataset: str, stem: str) -> Tuple[str, str, bool]:
    """Return (patient_id, eye, parsed). `parsed` is False when we fell back."""
    if dataset == "EyePACS":
        m = EYEPACS_STEM.match(stem)
        if m:
            return f"EyePACS::{m.group(1)}", m.group(2).lower(), True
    # Messidor-2 pairs eyes per examination, e.g. 20051020_43808_0100_PP
    if dataset == "Messidor2":
        m = re.match(r"^(\d{8}_\d+)_", stem)
        if m:
            return f"Messidor2::{m.group(1)}", "", True
    return f"{dataset}::{stem}", "", False


# ----------------------------------------------------------------- coordinates


def project_coords(
    coords_csv: Path,
    source_dir: Path,
    size: int,
    fit: str,
    tol_scale: float,
) -> Dict[str, Dict[str, float]]:
    """Re-project IDRiD Part C centres into the cached image's coordinate frame.

    The published coordinates are in original pixels, which the cache's crop,
    square fit and resize invalidate. Rather than storing a transform at cache
    time, the geometry is recomputed here with the very same functions, so the
    two can never drift apart.
    """
    import cv2
    from build_cache import imread, map_point, retinal_bbox

    # Read via idrid_tables: mirrors ship these with a UTF-8 BOM, with a title
    # line above the header, and as .xlsx. A single pd.read_csv handles one of
    # the three, and the id column is identified by its values rather than its
    # name because mirrors disagree about that too.
    from verify_dr.data.idrid_tables import id_column, read_markup_table

    frames, errors = read_markup_table(coords_csv)
    if not frames:
        raise ValueError(
            f"{coords_csv.name}: present but unreadable. Tried every encoding and "
            f"header offset; first error was {errors[0] if errors else 'unknown'}."
            + ("\n  -> pip install openpyxl"
               if any("openpyxl" in e for e in errors) else ""))

    frame = id_col = x_col = y_col = None
    for candidate in frames:
        col = id_column(candidate)
        if col is None:
            continue
        xs = next((c for c in candidate.columns
                   if c != col and (c.lower().startswith("x") or "x-" in c.lower())), None)
        ys = next((c for c in candidate.columns
                   if c != col and (c.lower().startswith("y") or "y-" in c.lower())), None)
        if xs is not None and ys is not None:
            frame, id_col, x_col, y_col = candidate, col, xs, ys
            break
    if frame is None:
        raise KeyError(
            f"{coords_csv.name}: readable, but no reading of it has an IDRiD id "
            f"column alongside X and Y columns. Columns seen: "
            f"{list(frames[0].columns)[:8]}")

    kind = "fovea" if "fovea" in coords_csv.name.lower() else "od"
    print(f"  {coords_csv.name}: id={id_col!r} x={x_col!r} y={y_col!r} -> {kind}")

    sources = {p.stem: p for p in source_dir.rglob("*")
               if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}

    out: Dict[str, Dict[str, float]] = {}
    missing = 0
    for _, row in frame.iterrows():
        stem = Path(str(row[id_col]).strip()).stem
        src = sources.get(stem)
        if src is None:
            missing += 1
            continue
        try:
            x, y = float(row[x_col]), float(row[y_col])
        except (TypeError, ValueError):
            continue
        img = imread(src)
        if img is None:
            missing += 1
            continue
        bbox, _ = retinal_bbox(img, tol_scale)
        px, py = map_point(x, y, bbox, size, fit)
        out.setdefault(stem, {})[f"{kind}_x"] = round(px, 2)
        out.setdefault(stem, {})[f"{kind}_y"] = round(py, 2)
        out[stem][f"{kind}_in_frame"] = int(0 <= px < size and 0 <= py < size)
    if missing:
        print(f"  warning: {missing} coordinate rows had no source image in {source_dir}")
    return out


# ------------------------------------------------------------------- builders


def rows_from_folders(cache: Dict[str, Path], cache_root: Path, dataset: str) -> List[dict]:
    """Grade from a 0-4 class-folder component; used for EyePACS."""
    base = cache_root / dataset.lower() / "images"
    rows, ungraded = [], 0
    for stem, path in cache.items():
        rel = path.relative_to(base)
        grade = path_component(rel, GRADE_DIRS)
        if grade is None:
            ungraded += 1
            continue
        split = path_component(rel, SPLIT_DIRS)
        rows.append({"stem": stem, "grade": int(grade),
                     "source_split": canon_split(split) if split else ""})
    if ungraded:
        print(f"  warning: {ungraded} cached images had no 0-4 class folder and were skipped")
    return rows


def rows_from_labels(
    label_files: Sequence[Path], cache: Dict[str, Path],
    id_col: Optional[str], grade_col: Optional[str],
) -> Tuple[List[dict], List[str]]:
    rows, unmatched = [], []
    for path in label_files:
        table = read_label_table(path, id_col, grade_col)
        split = canon_split(path.stem) or ""
        for _, item in table.iterrows():
            stem = Path(str(item["id"]).strip()).stem
            try:
                grade = int(float(item["grade"]))
            except (TypeError, ValueError):
                continue
            if stem not in cache:
                unmatched.append(stem)
                continue
            rows.append({"stem": stem, "grade": grade, "source_split": split})
    return rows, unmatched


# ----------------------------------------------------------------------- main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a VERIFY-DR manifest from a cached dataset and its labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--cache-root", required=True, type=Path,
                        help="Output root used by build_cache.py.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels", action="append", default=[], type=Path,
                        help="Label file (.txt/.csv/.xlsx). Repeatable, one per split.")
    parser.add_argument("--folder-labels", action="store_true",
                        help="Take the grade from a 0-4 class-folder component (EyePACS).")
    parser.add_argument("--no-grades", action="store_true",
                        help="Build from the cached images with grade -1 (unknown). For a "
                             "mask/geometry-only manifest such as IDRiD Part A, whose images "
                             "are not in the Part B grading table.")
    parser.add_argument("--id-col")
    parser.add_argument("--grade-col")
    parser.add_argument("--coords", action="append", default=[], type=Path,
                        help="IDRiD Part C centre table. Repeatable (fovea and OD).")
    parser.add_argument("--coords-source-dir", type=Path,
                        help="Original images the coordinates refer to; required with --coords.")
    parser.add_argument("--size", type=int, default=512, help="Must match build_cache.py.")
    parser.add_argument("--fit", choices=["pad", "crop"], default="pad",
                        help="Must match build_cache.py.")
    parser.add_argument("--tol-scale", type=float, default=0.10,
                        help="Must match build_cache.py.")
    parser.add_argument("--quality-output", type=Path,
                        help="Where DDR grade-5 ungradable rows go.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset

    if not (args.folder_labels or args.labels or args.no_grades):
        print("error: pass --folder-labels, --labels, or --no-grades", file=sys.stderr)
        return 2
    if args.coords and not args.coords_source_dir:
        print("error: --coords requires --coords-source-dir", file=sys.stderr)
        return 2

    print(f"{dataset}: reading cache {args.cache_root}")
    try:
        cache = cached_images(args.cache_root, dataset)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    masks = cached_masks(args.cache_root, dataset)
    print(f"  {len(cache)} cached images"
          + (f"; mask channels: {', '.join(sorted(masks))}" if masks else "; no masks"))

    unmatched: List[str] = []
    if args.no_grades:
        # Grade -1 means "unknown". This manifest serves C1/C2, which need masks and
        # geometry, not grades. build_variants.py drops anything outside 0-4, so such
        # a manifest can never leak into a development variant.
        rows = [{"stem": stem, "grade": -1, "source_split": ""} for stem in sorted(cache)]
        print(f"  --no-grades: {len(rows)} rows with grade -1 (masks/geometry only)")
    elif args.folder_labels:
        rows = rows_from_folders(cache, args.cache_root, dataset)
    else:
        missing = [p for p in args.labels if not p.exists()]
        if missing:
            print(f"error: label file(s) not found: {missing}", file=sys.stderr)
            return 1
        rows, unmatched = rows_from_labels(args.labels, cache, args.id_col, args.grade_col)

    if not rows:
        print("error: no label row matched a cached image.", file=sys.stderr)
        if unmatched:
            print(f"  {len(unmatched)} label ids, e.g. {unmatched[:5]}", file=sys.stderr)
        print(f"  {len(cache)} cached stems, e.g. {sorted(cache)[:5]}", file=sys.stderr)
        print("  The two use different id formats or refer to different image sets.",
              file=sys.stderr)
        print("  IDRiD is the usual case: Part A (segmentation) images are IDRiD_01-81",
              file=sys.stderr)
        print("  while the Part B grading table lists IDRiD_001-516 -- different images.",
              file=sys.stderr)
        print("  For a masks/geometry-only manifest, re-run with --no-grades.",
              file=sys.stderr)
        return 1

    coords: Dict[str, Dict[str, float]] = {}
    for table in args.coords:
        if not table.exists():
            print(f"error: coordinate table not found: {table}", file=sys.stderr)
            return 1
        for stem, values in project_coords(
            table, args.coords_source_dir, args.size, args.fit, args.tol_scale
        ).items():
            coords.setdefault(stem, {}).update(values)

    # Asking for coordinates and projecting none is a failure, not a quiet zero.
    # It is how C1 came to have no training targets while every run still looked
    # successful: IDRiD Part A (IDRiD_01-81, the mask set) and Part B (IDRiD_001-516,
    # the graded set the Part C centres cover) are different images, so a
    # --coords-source-dir pointing at the wrong part matches nothing at all.
    if args.coords and not coords:
        print("\nerror: --coords was given but no coordinate row matched an image in "
              f"{args.coords_source_dir}.", file=sys.stderr)
        print("  IDRiD ships two image sets, and their names look alike:",
              file=sys.stderr)
        print("    Part A  IDRiD_01-81    .../A. Segmentation/1. Original Images/",
              file=sys.stderr)
        print("    Part B  IDRiD_001-516  .../B. Disease Grading/1. Original Images/",
              file=sys.stderr)
        print("  The Part C centre tables cover Part B. Point --coords-source-dir at "
              "Part B's\n  originals (the parent, so Training and Testing are both "
              "covered).", file=sys.stderr)
        print("  Writing this manifest anyway would leave C1 with no targets, so it "
              "is not written.", file=sys.stderr)
        return 1

    # ---- assemble -----------------------------------------------------------
    records, quality, dropped, fallback = [], [], [], 0
    seen = set()
    for row in rows:
        stem = row["stem"]
        if stem in seen:                       # a stem listed in two label files
            continue
        seen.add(stem)
        grade = row["grade"]

        patient, eye, parsed = patient_and_eye(dataset, stem)

        record = {
            "image_path": str(cache[stem]),
            "grade": grade,
            "dataset": dataset,
            "patient_id": patient,
            "eye": eye,
            "source_split": row.get("source_split", ""),
        }
        for channel in MASK_CHANNELS:
            hit = masks.get(channel, {}).get(stem)
            record[f"{channel}_mask"] = str(hit) if hit else ""
        record.update(coords.get(stem, {}))

        if grade == DDR_UNGRADABLE and dataset == "DDR":
            quality.append(record)             # ungradable: quality head, not grading
        elif 0 <= grade <= 4 or (args.no_grades and grade == -1):
            records.append(record)
            if not parsed:                     # only count what reaches the manifest
                fallback += 1
        else:
            dropped.append((stem, grade))      # out of range: never silently discarded

    if not records:
        print("error: every row was filtered out -- check the grade column", file=sys.stderr)
        return 1

    frame = pd.DataFrame(records).sort_values("image_path").reset_index(drop=True)

    # ---- report -------------------------------------------------------------
    grades = Counter(frame["grade"])
    patients = frame["patient_id"].nunique()
    print(f"\n{dataset}: {len(frame)} rows, {patients} patients "
          f"({len(frame)/max(1, patients):.2f} images/patient)")
    print("  grade distribution:" if not args.no_grades
          else "  grades: unknown (-1), masks/geometry only")
    for g in sorted(grades):
        print(f"    {g}: {grades[g]:7d}  ({100*grades[g]/len(frame):5.1f}%)")
    if quality:
        print(f"  ungradable rows routed to the quality manifest: {len(quality)}")
    if dropped:
        print(f"  warning: {len(dropped)} rows had a grade outside 0-4 and were dropped "
              f"(e.g. {dropped[:3]})")
    if unmatched:
        print(f"  warning: {len(unmatched)} label rows had no cached image "
              f"(e.g. {unmatched[:3]}) -- did build_cache.py cover this split?")
    missing_from_labels = len(cache) - len(seen)
    if missing_from_labels > 0:
        print(f"  warning: {missing_from_labels} cached images were not in any label file")
    if fallback:
        print(f"  WARNING: {fallback} rows fell back to a per-image patient id.")
        print("           Those images cannot be grouped by patient. For any dataset with")
        print("           two eyes per person this leaks across splits and inflates results.")
    for channel in MASK_CHANNELS:
        n = int((frame[f"{channel}_mask"] != "").sum())
        if n:
            print(f"  mask {channel:15s} {n} rows")
    if coords:
        in_frame = sum(1 for v in coords.values()
                       if v.get("od_in_frame", 1) and v.get("fovea_in_frame", 1))
        print(f"  coordinates projected for {len(coords)} images "
              f"({in_frame} with every point inside the cached frame)")

    if args.dry_run:
        print("\ndry run -- nothing written")
        print(frame.head(5).to_string(index=False))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"\n  wrote {args.output}  ({len(frame)} rows)")

    if quality:
        target = args.quality_output or args.output.with_name("quality_manifest.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        qframe = pd.DataFrame(quality)
        if target.exists():                    # accumulate across datasets
            qframe = pd.concat([pd.read_csv(target), qframe], ignore_index=True)
            qframe = qframe.drop_duplicates("image_path")
        qframe.to_csv(target, index=False)
        print(f"  wrote {target}  ({len(qframe)} ungradable rows)")

    report = {
        "dataset": dataset,
        "rows": len(frame),
        "patients": int(patients),
        "images_per_patient": round(len(frame) / max(1, patients), 3),
        "grades": {str(k): int(v) for k, v in sorted(grades.items())},
        "ungradable": len(quality),
        "patient_id_fallbacks": fallback,
        "dropped_out_of_range": len(dropped),
        "labels_without_image": len(unmatched),
        "images_without_label": max(0, missing_from_labels),
        "mask_rows": {c: int((frame[f"{c}_mask"] != "").sum()) for c in MASK_CHANNELS
                      if (frame[f"{c}_mask"] != "").any()},
        "coords_projected": len(coords),
        "cache": {"size": args.size, "fit": args.fit, "tol_scale": args.tol_scale},
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
