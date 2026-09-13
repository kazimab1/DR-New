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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, nargs="+",
                        default=[Path("/kaggle/input"), Path("/kaggle/working")])
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

    print("\nMasks are a Phase 4 concern: M1 grades whole images and reads images and")
    print("grades only. Zero mask channels blocks nothing in Phase 3.")
    if empty:
        print(f"\nEmpty: {', '.join(empty)}")
        print("If a dataset your manifest needs is listed there, re-run Phase 1 for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
