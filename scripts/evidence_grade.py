#!/usr/bin/env python
"""C4 — evidence-only grading.

Runs M2a over graded images, turns its predicted masks into structured facts,
applies M3's rules, and compares the resulting evidence grade with the true one.

**This is the project's falsification test**, not a performance report. The
evidence pathway has to be *informative but weaker* than the grader:

  as good as M1   -> M1 is redundant and the two-pathway design is unmotivated
  noise           -> disagreement between them carries no signal and H1 cannot
                     hold for the stated reason

Either extreme is reported as a negative result with analysis. Neither is worked
around by tuning the evidence path against test data.

    python scripts/evidence_grade.py \\
        --manifest manifests/ddr_manifest.csv \\
        --checkpoint results/C2_lesions/best.pt \\
        --experiment C4_evidence_only --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_dr.data.segmentation import (  # noqa: E402
    LESION_NAMES, SegmentationDataset, load_segmentation_manifest,
)
from verify_dr.evaluation.metrics import quadratic_weighted_kappa  # noqa: E402
from verify_dr.models.evidence import LesionSegmenter  # noqa: E402
from verify_dr.reasoning import extract_facts, grade  # noqa: E402


def load_geometry(checkpoint: Optional[Path], device):
    if checkpoint is None:
        return None
    from verify_dr.models.evidence import GeometryHeatmapModel, GeometryModel

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    head = state.get("config", {}).get("head", "regress")
    model = (GeometryHeatmapModel(pretrained=False) if head == "heatmap"
             else GeometryModel(pretrained=False)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"  geometry from {checkpoint.name} (head={head})")
    return model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path, nargs="+")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="M2a weights, e.g. results/C2_lesions/best.pt")
    p.add_argument("--geometry-checkpoint", type=Path, default=None,
                   help="M2b weights. Without --trust-geometry this is only "
                        "recorded, never used for quadrants.")
    p.add_argument("--trust-geometry", action="store_true",
                   help="Let R4 use quadrants. Set this ONLY when C1 passed its "
                        "0.5 DD gate: a wrong quadrant frame makes R4 confidently "
                        "wrong rather than cautious.")
    p.add_argument("--experiment", required=True)
    p.add_argument("--results-dir", type=Path, default=Path("results/stage_c"))
    p.add_argument("--cache-root", type=Path, nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-lesion-px", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    started = time.time()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = args.results_dir / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    frames = [load_segmentation_manifest(m, args.cache_root, require_any_mask=False)
              for m in args.manifest]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("image_path")
    if args.datasets:
        wanted = {d.lower() for d in args.datasets}
        frame = frame[frame["dataset"].str.lower().isin(wanted)]

    # Grades are the whole point here, so rows without one are dropped loudly.
    before = len(frame)
    frame = frame[pd.to_numeric(frame["grade"], errors="coerce").fillna(-1) >= 0]
    frame = frame.reset_index(drop=True)
    if len(frame) != before:
        print(f"  {before - len(frame)} of {before} rows have no grade and are dropped")
    if frame.empty:
        print("error: no graded rows. C4 compares evidence grades with true grades, "
              "so a manifest built from the cache (grade -1) cannot be used -- pass "
              "the Phase 2 manifest that carries grades.", file=sys.stderr)
        return 1
    if args.limit:
        frame = frame.head(args.limit)

    dataset = SegmentationDataset(frame, args.image_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    model = LesionSegmenter(pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"C4 {args.experiment}: {len(frame)} graded images   device={device}")
    print(f"  M2a from {args.checkpoint.name}")

    geometry_model = load_geometry(args.geometry_checkpoint, device)
    if geometry_model is not None and not args.trust_geometry:
        print("  geometry loaded but NOT trusted: R4 will decline and grading is "
              "count-only (C1 did not pass its gate)")

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            probs = torch.sigmoid(model(image)).float().cpu().numpy()

            geom = None
            if geometry_model is not None:
                out = geometry_model(image)
                coords = (geometry_model.coordinates(out)
                          if hasattr(geometry_model, "coordinates") else out)
                geom = coords.float().cpu().numpy() * args.image_size

            for i, index in enumerate(batch["index"].tolist()):
                facts = extract_facts(
                    probs[i], threshold=args.threshold,
                    geometry=None if geom is None else geom[i],
                    geometry_trusted=args.trust_geometry,
                    geometry_reason="" if args.trust_geometry else "C1 gate not passed",
                    min_lesion_px=args.min_lesion_px)
                verdict = grade(facts)
                row = {
                    "image": Path(str(frame.at[index, "image_path"])).stem,
                    "dataset": frame.at[index, "dataset"],
                    "true_grade": int(frame.at[index, "grade"]),
                    "evidence_grade": verdict.evidence_grade,
                    "rule": verdict.rules_fired[0].split(":")[0],
                    "max_excludable_grade": verdict.max_excludable_grade,
                    "quadrants_used": int(verdict.quadrants_used),
                }
                row.update({f"n_{n}": facts.counts[n] for n in LESION_NAMES})
                rows.append(row)

    truth = np.array([r["true_grade"] for r in rows])
    pred = np.array([r["evidence_grade"] for r in rows])
    qwk = quadratic_weighted_kappa(truth, pred, k=5)

    # A reasoner that emits one grade for everything can still score respectably
    # on some metrics; report the spread so that is visible rather than inferred.
    distinct = int(len(set(pred.tolist())))
    confusion = [[int(((truth == t) & (pred == q)).sum()) for q in range(5)]
                 for t in range(5)]

    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": args.experiment,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(rows),
        "qwk": float(qwk),
        "distinct_evidence_grades": distinct,
        "mean_absolute_error": float(np.abs(truth - pred).mean()),
        "exact_agreement": float((truth == pred).mean()),
        "within_one": float((np.abs(truth - pred) <= 1).mean()),
        "quadrants_used": bool(args.trust_geometry),
        "confusion_true_by_evidence": confusion,
        "evidence_grade_distribution": {str(g): int((pred == g).sum()) for g in range(5)},
        "true_grade_distribution": {str(g): int((truth == g).sum()) for g in range(5)},
        "rules_fired": {r: sum(1 for x in rows if x["rule"] == r)
                        for r in sorted({x["rule"] for x in rows})},
        "checkpoint": str(args.checkpoint),
        "geometry_checkpoint": str(args.geometry_checkpoint) if args.geometry_checkpoint else None,
        "threshold": args.threshold,
        "minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  QWK vs true grade    {qwk:.4f}")
    print(f"  exact agreement      {summary['exact_agreement'] * 100:.1f}%")
    print(f"  within one grade     {summary['within_one'] * 100:.1f}%")
    print(f"  distinct grades      {distinct} of 5")
    print(f"  rules fired          {summary['rules_fired']}")
    print(f"  evidence grades      {summary['evidence_grade_distribution']}")
    print(f"  true grades          {summary['true_grade_distribution']}")
    if distinct == 1:
        print("\n  COLLAPSED: one grade for every image. QWK is meaningless here -- "
              "the reasoner is not discriminating, whatever the number says.")
    print(f"\n  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
