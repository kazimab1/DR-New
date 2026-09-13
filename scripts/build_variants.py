#!/usr/bin/env python3
"""Assemble the VERIFY-DR development variants and locked externals (Phase 2).

Takes the per-dataset manifests from prepare_manifest.py and produces:

    eyepacs_full.csv          all EyePACS, patient-grouped train/val/calibration/test
    eyepacs_balanced_<N>.csv  identical except train is capped at N images per grade
    eyepacs_ddr_full.csv      identical except DDR is added to train
    aptos_external.csv        LOCKED, every row split=test
    messidor2_external.csv    LOCKED, every row split=test
    dataset_plan.json         every count, shortfall, policy and seed

The three development variants **share their validation, calibration and test rows
byte for byte**. Only the training rows differ. That is what makes B6 (does dataset
balancing help?) and B7/H3 (does DDR improve transfer?) controlled comparisons rather
than two unrelated experiments.

Splitting
---------
Patients, never images. A patient's stratification label is their **worst** grade
across both eyes, so rare grades stay represented in every split. The script asserts
that no patient appears in two splits and exits non-zero if one does — see
docs/02_research_protocol.md Rule 3. Do not disable that assertion.

--eyepacs-split regroup (default)
    Ignore whatever split the mirror shipped and build a fresh patient-grouped one.
    Always safe. But the resulting EyePACS test set is NOT the official competition
    test set, so its QWK is **not** comparable to the Kaggle leaderboard (Rule 6).

--eyepacs-split source
    Honour the manifest's source_split. Only legitimate when the mirror really does
    carry the official competition split. The script verifies patient-disjointness
    first and refuses to proceed if the mirror's split leaks.

Either way the choice is recorded in dataset_plan.json under
`eyepacs.split_policy`, so the thesis can state honestly which protocol produced
each number.

Example
-------
    python scripts/build_variants.py \\
        --eyepacs manifests/eyepacs_manifest.csv \\
        --ddr manifests/ddr_manifest.csv \\
        --aptos manifests/aptos_manifest.csv \\
        --messidor2 manifests/messidor2_manifest.csv \\
        --output-dir manifests --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "calibration", "test")
REQUIRED = {"image_path", "grade", "dataset", "patient_id"}


# --------------------------------------------------------------------- loading


def read_manifest(path: Path, expect: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    wrong = set(frame["dataset"].unique()) - {expect}
    if wrong:
        raise ValueError(f"{path}: expected dataset {expect!r}, also found {sorted(wrong)}")

    before = len(frame)
    frame = frame.drop_duplicates("image_path").reset_index(drop=True)
    if len(frame) != before:
        print(f"  {path.name}: dropped {before - len(frame)} duplicate image_path rows")

    frame["grade"] = pd.to_numeric(frame["grade"], errors="coerce")
    frame = frame.dropna(subset=["grade", "patient_id"])
    frame["grade"] = frame["grade"].astype(int)
    out_of_range = (~frame["grade"].between(0, 4)).sum()
    if out_of_range:
        print(f"  {path.name}: dropped {out_of_range} rows with a grade outside 0-4")
        frame = frame[frame["grade"].between(0, 4)]
    if "source_split" not in frame:
        frame["source_split"] = ""
    frame["source_split"] = frame["source_split"].fillna("")
    return frame.reset_index(drop=True)


# -------------------------------------------------------------------- splitting


def patient_labels(frame: pd.DataFrame) -> pd.Series:
    """One stratification label per patient: their worst grade across both eyes.

    Using the max rather than a per-image grade keeps a patient indivisible while
    still stratifying on the clinically meaningful value.
    """
    return frame.groupby("patient_id")["grade"].max()


def assign_splits(
    pool: pd.DataFrame, val_frac: float, cal_frac: float, seed: int
) -> Dict[str, str]:
    """Patient -> train/val/calibration, stratified by worst grade."""
    labels = patient_labels(pool)
    rng = np.random.default_rng(seed)
    assignment: Dict[str, str] = {}

    for grade in sorted(labels.unique()):
        patients = labels.index[labels == grade].to_numpy()
        patients = np.sort(patients)          # deterministic before shuffling
        rng.shuffle(patients)
        n = len(patients)
        n_val = int(round(n * val_frac))
        n_cal = int(round(n * cal_frac))
        # With very few patients in a rare grade, never starve train.
        n_val = min(n_val, max(0, n - 1))
        n_cal = min(n_cal, max(0, n - n_val - 1))
        for i, pid in enumerate(patients):
            if i < n_val:
                assignment[pid] = "val"
            elif i < n_val + n_cal:
                assignment[pid] = "calibration"
            else:
                assignment[pid] = "train"
    return assignment


def assert_disjoint(frame: pd.DataFrame, label: str) -> None:
    """Rule 3. A patient in two splits inflates every number downstream."""
    spread = frame.groupby("patient_id")["split"].nunique()
    leaked = spread[spread > 1]
    if len(leaked):
        examples = [
            (pid, sorted(frame.loc[frame["patient_id"] == pid, "split"].unique()))
            for pid in leaked.index[:5]
        ]
        raise AssertionError(
            f"{label}: {len(leaked)} patients appear in more than one split.\n"
            f"  examples: {examples}\n"
            "  This is the leakage docs/02_research_protocol.md Rule 3 forbids."
        )


def check_source_split(frame: pd.DataFrame) -> Tuple[bool, int]:
    """Would honouring the mirror's own split leak across patients?"""
    marked = frame[frame["source_split"] != ""]
    if marked.empty:
        return False, 0
    spread = marked.groupby("patient_id")["source_split"].nunique()
    return True, int((spread > 1).sum())


# ------------------------------------------------------------------- reporting


def grade_counts(frame: pd.DataFrame) -> Dict[str, int]:
    return {str(g): int(n) for g, n in sorted(Counter(frame["grade"]).items())}


def summarise(frame: pd.DataFrame, name: str) -> dict:
    rows = []
    for split in SPLITS:
        part = frame[frame["split"] == split]
        if part.empty:
            continue
        rows.append({
            "split": split,
            "images": len(part),
            "patients": int(part["patient_id"].nunique()),
            "grades": grade_counts(part),
            "datasets": {k: int(v) for k, v in Counter(part["dataset"]).items()},
        })
    print(f"\n{name}")
    print(f"  {'split':<12}{'images':>9}{'patients':>10}   grade distribution")
    for row in rows:
        total = row["images"]
        dist = "  ".join(f"{g}:{100*n/total:4.1f}%" for g, n in row["grades"].items())
        print(f"  {row['split']:<12}{total:>9}{row['patients']:>10}   {dist}")
    return {"name": name, "images": len(frame), "splits": rows}


# -------------------------------------------------------------------- variants


def balanced_train(
    frame: pd.DataFrame, n_per_grade: int, seed: int
) -> Tuple[pd.DataFrame, dict]:
    """Cap train at n_per_grade images per grade, sampling without replacement.

    Only train is touched: val, calibration and test must stay identical to the
    full variant or B6 stops being a controlled comparison. Shortfalls on rare
    grades are recorded, never made up by duplication.
    """
    train = frame[frame["split"] == "train"]
    held = frame[frame["split"] != "train"]

    parts, selected, shortfall = [], {}, {}
    for grade in range(5):
        group = train[train["grade"] == grade]
        take = min(len(group), n_per_grade)
        selected[str(grade)] = int(take)
        if take < n_per_grade:
            shortfall[str(grade)] = int(n_per_grade - take)
        if take:
            parts.append(group.sample(n=take, random_state=seed + grade))

    if not parts:
        raise ValueError("balanced variant: no training rows survived")
    out = pd.concat(parts + [held], ignore_index=True)
    info = {
        "requested_per_grade": n_per_grade,
        "selected_per_grade": selected,
        "shortfall_per_grade": shortfall,
        "sampling": "without_replacement_cap",
        "note": "train only; val/calibration/test identical to eyepacs_full",
    }
    return out, info


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["split", "image_path"]).to_csv(path, index=False)
    print(f"  wrote {path}  ({len(frame)} rows)")


# ------------------------------------------------------------------------ main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build VERIFY-DR development variants and locked external manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eyepacs", required=True, type=Path)
    parser.add_argument("--ddr", type=Path)
    parser.add_argument("--aptos", type=Path)
    parser.add_argument("--messidor2", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--calibration-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Only used when --eyepacs-split regroup.")
    parser.add_argument("--balanced-n", type=int, default=1000)
    parser.add_argument("--eyepacs-split", choices=["regroup", "source"], default="regroup")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir

    print("reading manifests")
    try:
        eyepacs = read_manifest(args.eyepacs, "EyePACS")
        ddr = read_manifest(args.ddr, "DDR") if args.ddr else None
        aptos = read_manifest(args.aptos, "APTOS") if args.aptos else None
        messidor2 = read_manifest(args.messidor2, "Messidor2") if args.messidor2 else None
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"  EyePACS {len(eyepacs)} images, {eyepacs['patient_id'].nunique()} patients")
    ratio = len(eyepacs) / max(1, eyepacs["patient_id"].nunique())
    if ratio < 1.5:
        print(f"  WARNING: {ratio:.2f} images per patient. EyePACS should be near 2.0.")
        print("           Patient parsing probably failed in prepare_manifest.py, which")
        print("           means eye pairs are not grouped and the split will leak.")

    # ---- decide the EyePACS test split -------------------------------------
    has_source, leaks = check_source_split(eyepacs)
    policy: Dict[str, object] = {"mode": args.eyepacs_split}

    if args.eyepacs_split == "source":
        if not has_source:
            print("error: --eyepacs-split source but the manifest has no source_split",
                  file=sys.stderr)
            return 1
        if leaks:
            print(f"error: the mirror's own split puts {leaks} patients in more than one "
                  "split.\n  Honouring it would leak (Rule 3). Re-run with "
                  "--eyepacs-split regroup.", file=sys.stderr)
            return 1
        test = eyepacs[eyepacs["source_split"] == "test"].copy()
        pool = eyepacs[eyepacs["source_split"] != "test"].copy()
        policy["source"] = "mirror source_split"
        policy["leaderboard_comparable"] = "only if this mirror ships the official split"
    else:
        labels = patient_labels(eyepacs)
        rng = np.random.default_rng(args.seed)
        test_patients = set()
        for grade in sorted(labels.unique()):
            patients = np.sort(labels.index[labels == grade].to_numpy())
            rng.shuffle(patients)
            n_test = min(int(round(len(patients) * args.test_frac)), max(0, len(patients) - 1))
            test_patients.update(patients[:n_test])
        mask = eyepacs["patient_id"].isin(test_patients)
        test, pool = eyepacs[mask].copy(), eyepacs[~mask].copy()
        policy["source"] = f"regrouped, test_frac={args.test_frac}"
        # Rule 6: a regrouped test set is not the official one.
        policy["leaderboard_comparable"] = False
        if has_source:
            print(f"  note: the mirror shipped its own split "
                  f"({'leaks' if leaks else 'patient-disjoint'}); regrouping anyway")

    if test.empty or pool.empty:
        print("error: the EyePACS test/development split came out empty", file=sys.stderr)
        return 1

    # ---- carve train / val / calibration out of the development pool -------
    assignment = assign_splits(pool, args.val_frac, args.calibration_frac, args.seed)
    pool = pool.assign(split=pool["patient_id"].map(assignment))
    test = test.assign(split="test")
    full = pd.concat([pool, test], ignore_index=True)

    try:
        assert_disjoint(full, "eyepacs_full")
    except AssertionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n  patient-disjointness assertion: PASS")

    plan: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "eyepacs": {
            "split_policy": policy,
            "val_frac": args.val_frac,
            "calibration_frac": args.calibration_frac,
        },
        "rules": {
            "patient_grouped": "Rule 3 - asserted, never disabled",
            "natural_prevalence_tests": "Rule 4 - balancing touches train only",
            "externals_locked": "Rule 1 - evaluated once, after the freeze",
        },
        "variants": {},
    }

    # ---- the three development variants ------------------------------------
    plan["variants"]["eyepacs_full"] = summarise(full, "eyepacs_full")

    balanced_name = f"eyepacs_balanced_{args.balanced_n}"
    balanced, balance_info = balanced_train(full, args.balanced_n, args.seed)
    assert_disjoint(balanced, balanced_name)
    plan["variants"][balanced_name] = summarise(balanced, balanced_name)
    plan["variants"][balanced_name]["balancing"] = balance_info
    if balance_info["shortfall_per_grade"]:
        print(f"  shortfall vs {args.balanced_n}/grade: "
              f"{balance_info['shortfall_per_grade']} (capped, never duplicated)")

    merged = None
    if ddr is not None:
        # DDR joins train only. Holding val/calibration/test identical across
        # variants is what makes H3 a controlled comparison.
        ddr_train = ddr.assign(split="train")
        merged = pd.concat([full, ddr_train], ignore_index=True)
        assert_disjoint(merged, "eyepacs_ddr_full")
        plan["variants"]["eyepacs_ddr_full"] = summarise(merged, "eyepacs_ddr_full")
        plan["variants"]["eyepacs_ddr_full"]["ddr_policy"] = (
            "DDR added to train only; val/calibration/test identical to eyepacs_full"
        )

    # ---- locked externals ---------------------------------------------------
    externals = []
    for frame, name in ((aptos, "aptos_external"), (messidor2, "messidor2_external")):
        if frame is None:
            continue
        locked = frame.assign(split="test", locked=True)
        externals.append((locked, name))
        plan["variants"][name] = summarise(locked, name)
        plan["variants"][name]["locked"] = True
        plan["variants"][name]["policy"] = (
            "Rule 1 - never used for training, validation, early stopping, "
            "calibration fitting, hyperparameter choice or model selection"
        )

    # ---- identical held-out rows across variants ---------------------------
    held_full = set(full.loc[full["split"] != "train", "image_path"])
    for frame, name in [(balanced, balanced_name)] + ([(merged, "eyepacs_ddr_full")] if merged is not None else []):
        held = set(frame.loc[frame["split"] != "train", "image_path"])
        if held != held_full:
            print(f"error: {name} does not share eyepacs_full's held-out rows "
                  f"(+{len(held - held_full)} / -{len(held_full - held)}); "
                  "the variant comparison would not be controlled", file=sys.stderr)
            return 1
    print("  held-out rows identical across development variants: PASS")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return 0

    print()
    write(full, out_dir / "eyepacs_full.csv")
    write(balanced, out_dir / f"{balanced_name}.csv")
    if merged is not None:
        write(merged, out_dir / "eyepacs_ddr_full.csv")
    for frame, name in externals:
        write(frame, out_dir / f"{name}.csv")

    plan_path = out_dir / "dataset_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"  wrote {plan_path}")

    if not policy.get("leaderboard_comparable"):
        print("\n  NOTE (Rule 6): this EyePACS test set is not the official competition")
        print("  split, so its QWK is NOT comparable to the Kaggle leaderboard. Say so")
        print("  wherever the number appears.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
