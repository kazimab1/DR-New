"""Reading IDRiD's published markup tables.

These tables decide whether C1 has any training data, and three properties of
them have each broken a run:

* the file name carries no reliable keyword -- mirrors ship
  ``IDRiD_OD_Center_Training Set_Markups.csv``, ``..._Markups.xlsx`` and
  ``IDRiD_Localization_Groundtruth_Testing.csv`` for the same content;
* the encoding and header row vary -- a UTF-8 BOM, or a title line above the
  header;
* ``.xlsx`` needs an engine that is not installed everywhere.

So tables are identified by **what is inside them** rather than by their names,
and the three ways the search can come up empty -- no files, files that cannot
be read, files that are not coordinate tables -- are reported apart, because
each has a different fix and an earlier version collapsed all three into
"no Part C coordinate tables found".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

TABLE_EXT = {".csv", ".xlsx", ".xls"}

#: IDRiD names every image IDRiD_<digits>, in both Part A and Part B.
STEM_RE = re.compile(r"^idrid[_\-]?\d+$", re.I)


def read_markup_table(path: Path) -> Tuple[List[pd.DataFrame], List[str]]:
    """Every plausible reading of one table: (frames, errors).

    Both are returned because an empty ``frames`` with a populated ``errors``
    means "present but unreadable", which is not the same finding as absent.
    """
    frames: List[pd.DataFrame] = []
    errors: List[str] = []
    if Path(path).suffix.lower() in {".xlsx", ".xls"}:
        attempts = [("excel", {"header": h}) for h in (0, 1, 2)]
    else:
        attempts = [("csv", {"encoding": e, "header": h})
                    for e in ("utf-8-sig", "latin-1") for h in (0, 1, 2)]
    for kind, kwargs in attempts:
        try:
            frame = pd.read_excel(path, **kwargs) if kind == "excel" \
                else pd.read_csv(path, **kwargs)
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        frame.columns = [str(c).strip() for c in frame.columns]
        frames.append(frame)
    return frames, errors


def id_column(frame: pd.DataFrame) -> Optional[str]:
    """The image-id column, found by its values rather than its name.

    Mirrors label it 'Image No', 'Image name', or leave it unnamed; the values
    are always IDRiD_NNN.
    """
    for col in frame.columns:
        values = frame[col].dropna().astype(str).str.strip()
        if len(values) < 5:
            continue
        stems = [Path(v).stem for v in values]
        if sum(bool(STEM_RE.match(s)) for s in stems) >= 0.8 * len(stems):
            return col
    return None


def _xy_columns(frame: pd.DataFrame, id_col: str) -> List[str]:
    return [c for c in frame.columns
            if c != id_col and pd.api.types.is_numeric_dtype(frame[c])]


def _has_xy(numeric: Sequence[str]) -> bool:
    low = [c.lower() for c in numeric]
    return (any(c.startswith("x") or "x-" in c for c in low)
            and any(c.startswith("y") or "y-" in c for c in low))


def find_coord_tables(root: Path):
    """Coordinate tables under `root`.

    Returns ``(found, not_coords, unreadable)`` where ``found`` maps path ->
    ``(ids, id_column)``. A grading table carries IDRiD ids too, so the X/Y
    pair is what distinguishes a coordinate table from one.
    """
    found: Dict[Path, Tuple[set, str]] = {}
    not_coords: List[Tuple[Path, str]] = []
    unreadable: List[Tuple[Path, str]] = []

    for path in sorted(p for p in Path(root).rglob("*")
                       if p.is_file() and p.suffix.lower() in TABLE_EXT):
        frames, errors = read_markup_table(path)
        if not frames:
            unreadable.append((path, errors[0] if errors else "unreadable"))
            continue
        hit, near = None, None
        for frame in frames:
            col = id_column(frame)
            if col is None:
                continue
            numeric = _xy_columns(frame, col)
            if _has_xy(numeric):
                hit = (frame, col)
                break
            if near is None:
                near = numeric
        if hit is not None:
            frame, col = hit
            ids = {Path(str(v).strip()).stem for v in frame[col].dropna()}
            found[path] = ({i for i in ids if STEM_RE.match(i)}, col)
        elif near is not None:
            not_coords.append((path, f"IDRiD ids but no X/Y pair (numeric: {near[:3]})"))
        else:
            not_coords.append((path, "no IDRiD id column"))
    return found, not_coords, unreadable


def report(found, not_coords, unreadable) -> str:
    """One block naming what was found and, when nothing was, which fix applies."""
    lines = []
    if found:
        lines.append("coordinate tables:")
        for path, (ids, col) in found.items():
            lines.append(f"   {path.name}")
            lines.append(f"      id column {col!r} · {len(ids)} ids · "
                         f"e.g. {sorted(ids)[:2]}")
    if not_coords:
        lines.append("readable, but not coordinate tables:")
        for path, why in not_coords:
            lines.append(f"   {path.name}: {why}")
    if unreadable:
        lines.append("PRESENT BUT UNREADABLE:")
        for path, why in unreadable:
            lines.append(f"   {path.name}: {why}")
        if any("openpyxl" in why for _p, why in unreadable):
            lines.append("   -> run  !pip install openpyxl  and re-run this cell.")
    if not (found or not_coords or unreadable):
        lines.append("no .csv/.xlsx tables under this root at all.")
    return "\n".join(lines)
