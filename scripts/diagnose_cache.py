#!/usr/bin/env python3
"""Show what is actually in the attached 512 px cache.

Answers the question every phase from 3 onward starts with: is the data really
there, and which published dataset is each piece coming from? The cache is
routinely split across two Kaggle datasets -- the full build plus the IDRiD mask
top-up -- and a listing that looks at one root, or counts only direct children
of images/, reports missing data that is not missing.

    python scripts/diagnose_cache.py                 # /kaggle/input and /kaggle/working
    python scripts/diagnose_cache.py --base some/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def find_roots(bases: Sequence[Path]) -> List[Path]:
    """build_cache.py writes <root>/<dataset>/cache_report.json, so a report file
    identifies its root two levels up. Order is preserved and duplicates dropped."""
    return list(dict.fromkeys(
        report.parent.parent
        for base in bases if base.exists()
        for report in base.rglob("cache_report.json")
    ))


A0_MAX_FALLBACK = 0.005


def a0_report(roots: Sequence[Path]) -> None:
    """Read every cache_report.json and apply experiment A0's gate.

    A0 asks whether the 512 px cache preserved the data. The evidence has been
    sitting in the published cache since Phase 1; this reads it out so the answer
    lands in the register instead of being assumed.
    """
    import json

    print("\n" + "=" * 72)
    print("A0 - did the 512 px cache preserve the data?")
    print("=" * 72)

    verdicts = []
    for root in roots:
        for report_path in sorted(root.glob("*/cache_report.json")):
            try:
                r = json.loads(report_path.read_text())
            except (OSError, ValueError) as exc:
                print(f"\n  {report_path.parent.name}: unreadable ({exc})")
                continue

            name = r.get("dataset", report_path.parent.name)
            crop = r.get("crop") or {}
            rate = crop.get("fallback_rate")
            counts = r.get("counts") or {}
            failed = counts.get("failed", 0)

            print(f"\n  {name}")
            print(f"    cached            {r.get('cached', '?')} of {counts.get('found', '?')} found"
                  f"   (over {r.get('runs', '?')} run(s))")
            print(f"    crop detected     {crop.get('detected', '?')}")
            print(f"    crop fell back    {crop.get('fallback_full_frame', '?')} "
                  f"of {crop.get('images_measured', '?')} measured")
            if rate is None:
                print("    fallback rate     not recorded -- cannot judge A0 for this dataset")
                verdicts.append((name, None))
            else:
                ok = rate <= A0_MAX_FALLBACK
                print(f"    fallback rate     {rate:.5f}   gate {A0_MAX_FALLBACK}   "
                      f"{'within gate' if ok else 'ABOVE GATE'}")
                if not ok:
                    print("                      ^ ambiguous on its own. A high rate means")
                    print("                        either a broken crop OR a source whose")
                    print("                        images are already cropped, where the")
                    print("                        fallback is the correct answer. Only the")
                    print("                        contact sheet separates them -- verified")
                    print("                        benign for eyepacs and aptos, Sep 2026.")
                verdicts.append((name, ok))
            if failed:
                print(f"    failures          {failed}  <- investigate before the freeze")
            if r.get("interrupted"):
                print("    interrupted       True  <- this run did not finish cleanly")
            masks = r.get("masks") or {}
            if masks:
                print(f"    mask channels     {', '.join(sorted(masks))}")
            print(f"    size              {(r.get('output') or {}).get('total_mib', '?')} MiB")

    judged = [ok for _, ok in verdicts if ok is not None]
    print("\n  " + "-" * 68)
    if not verdicts:
        print("  No cache_report.json found. A0 cannot be closed.")
    elif all(judged) and len(judged) == len(verdicts):
        print(f"  Crop gate PASSES for all {len(judged)} datasets.")
        print("  Still outstanding for A0: per-grade counts reconciled against the")
        print("  manifests, and the 100-crop visual audit (contact_sheet.jpg in each")
        print("  dataset directory -- open it and look before signing A0 off).")
    else:
        bad = [n for n, ok in verdicts if ok is False]
        unknown = [n for n, ok in verdicts if ok is None]
        if bad:
            print(f"  Above the crop gate: {', '.join(bad)}")
            print("  Check each one's contact_sheet.jpg before concluding anything.")
        if unknown:
            print(f"  No rate recorded for: {', '.join(unknown)}")
        print("  A0 cannot be signed off as it stands.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, nargs="+",
                        default=[Path("/kaggle/input"), Path("/kaggle/working")])
    parser.add_argument("--reports", action="store_true",
                        help="Print each cache_report.json and apply the A0 gate "
                             "(crop fallback rate must stay at or under 0.005).")
    args = parser.parse_args(argv)

    roots = find_roots(args.base)
    if not roots:
        print(f"No cache found under {[str(b) for b in args.base]}.", file=sys.stderr)
        print("Attach 'verify-dr-cache-512' under Add Data.", file=sys.stderr)
        return 1

    print(f"{len(roots)} cache root(s):")
    for root in roots:
        print("   ", root)

    empty = []
    for root in roots:
        print(f"\n=== {root}")
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            images = d / "images"
            if not images.is_dir():
                print(f"  {d.name:<12} no images/ directory")
                continue

            direct = list(images.glob("*"))
            loose = [x for x in direct if x.is_file()]
            subdirs = [x for x in direct if x.is_dir()]
            # rglob, not glob: counting direct children only makes a nested
            # layout report 1 image, which reads as catastrophic data loss.
            found = [f for f in images.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES]
            masks = sorted(m.name for m in (d / "masks").iterdir()) if (d / "masks").is_dir() else []

            print(f"  {d.name:<12} {len(found):>7} images   "
                  f"({len(loose)} loose, {len(subdirs)} subdirs)   masks: {masks or 'none'}")
            if subdirs:
                print(f"               nested under: {subdirs[0].name}/")
            if found:
                print(f"               sample: {found[0].relative_to(root)}")
            else:
                print("               *** NO IMAGES ***")
                empty.append(f"{root.name}/{d.name}")

    if args.reports:
        a0_report(roots)

    print("\nMasks are a Phase 4 concern: M1 grades whole images and reads images and")
    print("grades only. Zero mask channels blocks nothing in Phase 3.")
    if empty:
        print(f"\nEmpty: {', '.join(empty)}")
        print("If a dataset your manifest needs is listed there, re-run Phase 1 for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
