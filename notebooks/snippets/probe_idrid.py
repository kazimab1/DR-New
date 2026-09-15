"""Paste into any Kaggle cell with the raw IDRiD attached. Reports only.

Depends on nothing in this repo, so it cannot fail for the reasons the 01c
detection has been failing. Run it, paste the whole output.
"""
from pathlib import Path
import pandas as pd

INPUT = Path("/kaggle/input")
IMG = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TAB = {".csv", ".xlsx", ".xls"}

def roots(max_depth=3):
    """Every directory /kaggle/input exposes, breadth-first.

    Kaggle sometimes mounts everything under a datasets/ or competitions/
    wrapper, so /kaggle/input can have exactly one child and iterdir() sees
    only that.
    """
    level, out = [INPUT], []
    for _ in range(max_depth):
        nxt = []
        for d in level:
            try:
                kids = sorted(c for c in d.iterdir() if c.is_dir())
            except OSError:
                continue
            out.extend(kids); nxt.extend(kids)
        level = nxt
    return out

print("EVERYTHING /kaggle/input EXPOSES")
for d in roots():
    print("   " + "  " * (len(d.relative_to(INPUT).parts) - 1) + d.name)

def norm(s):
    return s.lower().replace("-", "").replace("_", "").replace("%20", "")

mounts = []
for d in roots():
    if not d.is_dir() or norm(d.name).startswith("verifydr"):
        continue
    if "idrid" in norm(d.name) or all(k in norm(d.name) for k in ("diabetic", "retinopathy")):
        if not any(str(d).startswith(str(m) + "/") for m in mounts):
            mounts.append(d)

print(f"\n{len(mounts)} IDRiD-looking mount(s): {[m.name for m in mounts]}")

for m in mounts:
    print("\n" + "=" * 74)
    print(m.name)
    print("=" * 74)

    files = [f for f in m.rglob("*") if f.is_file()]
    print(f"{len(files)} files total")

    print("\n-- directories holding images --")
    dirs = {}
    for f in files:
        if f.suffix.lower() in IMG:
            dirs.setdefault(f.parent, []).append(f.stem)
    for d, stems in sorted(dirs.items(), key=lambda kv: -len(kv[1])):
        s = sorted(stems)
        print(f"   {len(s):>5}  {s[0]} .. {s[-1]}")
        print(f"          {d.relative_to(m)}")

    tables = [f for f in files if f.suffix.lower() in TAB]
    print(f"\n-- {len(tables)} table file(s) --")
    for t in tables:
        print(f"   {t.relative_to(m)}")
        for how, kw in (("csv", dict(encoding='utf-8-sig')), ("csv", dict(encoding='latin-1')),
                        ("excel", {})):
            if (how == "excel") != (t.suffix.lower() in {".xlsx", ".xls"}):
                continue
            try:
                f = pd.read_excel(t, **kw) if how == "excel" else pd.read_csv(t, **kw)
            except Exception as exc:
                print(f"      READ FAILED ({how}): {type(exc).__name__}: {exc}")
                continue
            print(f"      shape {f.shape}  columns {[str(c) for c in f.columns][:8]}")
            print("      first 2 rows:")
            for line in f.head(2).to_string(index=False).splitlines():
                print("        ", line)
            break
