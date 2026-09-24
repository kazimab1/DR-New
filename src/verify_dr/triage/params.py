"""fitted_params.json: written by fit_params.py, committed, then read by analyse.py.

ANALYSIS_PLAN.md s10 step 3 commits the fitted values before any locked image is
read. The file carries a digest over its own canonical form -- sorted keys, no
whitespace, Python's round-trip float repr -- so it can be pasted through a chat
window and re-indented without breaking verification, while any change to a
*value* is caught.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

PARAMS_NAME = "fitted_params.json"


def _plain(obj: Any) -> Any:
    """numpy scalars and tuples -> JSON types; refuses NaN and infinity."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("fitted parameters must be finite")
    return obj


def canonical(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != "digest"}
    return json.dumps(_plain(body), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(obj: dict) -> str:
    return hashlib.sha256(canonical(obj).encode("ascii")).hexdigest()


def _one_line_lists(text: str) -> str:
    """Put every list of plain values on one line: a third of the length to paste."""
    return re.sub(r"\[\s*([^\[\]{}]*?)\s*\]",
                  lambda m: "[" + " ".join(m.group(1).split()) + "]", text)


def write_params(obj: dict, path: Path) -> str:
    obj = _plain(obj)
    obj["digest"] = digest(obj)
    Path(path).write_text(_one_line_lists(json.dumps(obj, indent=1, sort_keys=True)) + "\n")
    return obj["digest"]


def load_params(path: Path) -> dict:
    """The parameters, after checking no value changed since they were fitted."""
    obj = json.loads(Path(path).read_text())
    if "digest" not in obj:
        raise ValueError(f"{path}: no digest -- not written by fit_params.py")
    actual = digest(obj)
    if actual != obj["digest"]:
        raise ValueError(f"{path}: digest mismatch ({actual[:12]}... vs recorded "
                         f"{obj['digest'][:12]}...). A fitted value was changed after fitting.")
    return obj
