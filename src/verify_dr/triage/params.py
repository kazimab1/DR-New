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
import subprocess
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


class StepThreeError(RuntimeError):
    """Step 3 of ANALYSIS_PLAN.md s10 has not happened, or not as the plan orders it."""


def step3_record(repo: Path) -> dict:
    """The committed parameters -- after checking step 3 happened as s10 orders it.

    The locked pass (step 4) and the unblinding (step 5) both start here, and refuse
    unless `preregistration/fitted_params.json`

      1. exists in the repository,
      2. verifies against its own digest (no value changed since fitting),
      3. is tracked by git and unmodified -- it is the committed file, and
      4. has its full digest recorded in PREREGISTRATION.md.

    Works in a shallow clone: nothing here needs history.
    """
    repo = Path(repo)
    path = repo / "preregistration" / PARAMS_NAME
    rel = str(path.relative_to(repo))
    if not path.exists():
        raise StepThreeError(f"{rel} is not in the repository: the fitted parameters have not "
                             "been committed (ANALYSIS_PLAN.md s10, step 3). Nothing locked "
                             "may be read before they are.")
    params = load_params(path)
    try:
        tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                                 capture_output=True, text=True, timeout=30)
        clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as err:
        raise StepThreeError(f"git could not be asked about {rel} ({err})") from err
    if tracked.returncode != 0:
        raise StepThreeError(f"{rel} exists but is not tracked by git: it was not committed.")
    if clean.returncode != 0:
        raise StepThreeError(f"{rel} differs from its committed version.")
    prereg = (repo / "preregistration" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    if params["digest"] not in prereg:
        raise StepThreeError(f"PREREGISTRATION.md does not record the digest {params['digest'][:12]}"
                             f"... of the committed parameters (step 3 records it).")
    return params

