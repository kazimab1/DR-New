#!/usr/bin/env python
"""Plan step 2: fit every free parameter on the calibration split (ANALYSIS_PLAN.md s10).

    python scripts/fit_params.py --internal <predictions> \\
        --manifest-full eyepacs_full.csv --manifest-ddr eyepacs_ddr_full.csv --out fitted

Reads the internal pass -- `reference_eyepacs_full`, `reference_eyepacs_ddr_full` and
`calibration` under <predictions> -- and INTERNAL labels only: the calibration split
and the training rows of the OOD reference sample. Per model it fits

    T                 stage 0+1 temperature (s4.2); plus T for stage 1 alone, D1's comparison
    OOD statistics    grade means + Ledoit-Wolf shared precision on the 5,000 references,
                      then the mean/SD of calibration distances and tau_ood = 95th pct (s5.3)
    tau_conf          90th percentile of calibration d_conf (s6.1)
    r                 the disagreement weight in {0, 0.5, 1.5, 2.5} with the best calibration
                      coverage-accuracy AUC; ties go to the smaller r (s5.5)

and writes `fitted_params.json` (every value, pi_src, the sha256 of every input, and a
digest over all of it) plus `ood/<model>.npz`, whose digests the JSON records.
Nothing here reads val, the EyePACS test split, APTOS or Messidor-2.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_dr.calibration import (  # noqa: E402
    EM_MAX_ITER, EM_TOL, T_BOUNDS, coral_probs, em_prior, fit_temperature, grade_prior, nll,
    predicted_confidence, stage01,
)
from verify_dr.evaluation.metrics import brier_score, expected_calibration_error  # noqa: E402
from verify_dr.models.grading import cumulative_to_grade  # noqa: E402
from verify_dr.triage import claims as C  # noqa: E402
from verify_dr.triage import faithfulness as F  # noqa: E402
from verify_dr.triage import ood as O  # noqa: E402
from verify_dr.triage import selective as S  # noqa: E402
from verify_dr.triage import signals as G  # noqa: E402
from verify_dr.triage.params import PARAMS_NAME, write_params  # noqa: E402
from verify_dr.triage.passdata import (  # noqa: E402
    Pass, labels_for, load_pass, parse_model, sha256_file,
)

PLAN = "preregistration/ANALYSIS_PLAN.md -- FINAL 2026-09-23, commit 1f2f9c1"
ECE_BINS = 15
REFERENCES = {"eyepacs_full": "reference_eyepacs_full",
              "eyepacs_ddr_full": "reference_eyepacs_ddr_full"}


def constants() -> Dict[str, object]:
    """Every fixed choice the fitted values depend on, recorded beside them."""
    return {
        "temperature_bounds": list(T_BOUNDS), "em_tol": EM_TOL, "em_max_iter": EM_MAX_ITER,
        "ood_reference_size": O.REFERENCE_SIZE, "ood_reference_seed": O.REFERENCE_SEED,
        "ood_percentile": O.OOD_PERCENTILE, "ood_sd_ddof": 1,
        "conf_percentile": G.CONF_PERCENTILE, "r_grid": list(G.R_GRID), "auc_tie": G.AUC_TIE,
        "max_excludable_grade": G.MAX_EXCLUDABLE_GRADE, "ece_bins": ECE_BINS,
        "coverage_points": list(S.COVERAGE_POINTS), "bootstrap_resamples": C.RESAMPLES,
        "bootstrap_seed": C.SEED, "controls": F.CONTROLS, "min_controls_ok": F.MIN_CONTROLS_OK,
    }


def calibration_quality(z: np.ndarray, yhat: np.ndarray, y: np.ndarray, t01: float, t1: float,
                        pi_src: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {"nll": {}, "ece": {}, "brier": {}}
    for label, p in (("raw", coral_probs(z)), ("stage1", coral_probs(z, t1)),
                     ("stage01", stage01(z, t01, pi_src))):
        conf = predicted_confidence(p, yhat)
        out["nll"][label] = nll(p, y)
        out["ece"][label] = expected_calibration_error(conf, (yhat == y).astype(float), ECE_BINS)
        out["brier"][label] = brier_score(p, y)
    # Descriptive, fitted on nothing: EM's premise. EM assumes the average calibrated
    # posterior equals the prior it was calibrated under. Where it does not, EM drifts
    # even with no shift at all -- here, on the calibration split itself.
    p01 = stage01(z, t01, pi_src)
    em = em_prior(p01, pi_src)
    out["em_premise"] = {"mean_posterior": p01.mean(axis=0).tolist(),
                         "true_prior": grade_prior(y).tolist(),
                         "em_prior_no_shift": em.prior.tolist(),
                         "em_iterations": em.iterations, "em_converged": em.converged}
    return out


def choose_r(d_ev: np.ndarray, d_fa: np.ndarray, correct: np.ndarray) -> tuple:
    """The r with the best calibration AUC; an AUC within AUC_TIE of the best is a tie,
    and ties go to the smaller r -- the grid is scanned upwards (s5.5)."""
    aucs = {r: S.auc(S.dense_rank(G.disagreement(d_ev, d_fa, r)), correct) for r in G.R_GRID}
    best = max(aucs.values())
    chosen = next(r for r in G.R_GRID if aucs[r] >= best - G.AUC_TIE)
    return chosen, aucs


def check_pass(p: Pass, split: str, embeddings_only: bool, what: str) -> None:
    problems = []
    if p.run.get("split") != split:
        problems.append(f"split is {p.run.get('split')!r}, expected {split!r}")
    if bool(p.run.get("embeddings_only")) != embeddings_only:
        problems.append(f"embeddings_only is {p.run.get('embeddings_only')}")
    if p.run.get("locked"):
        problems.append("it is a locked pass")
    if embeddings_only and (p.run.get("sample") != O.REFERENCE_SIZE
                            or p.run.get("sample_seed") != O.REFERENCE_SEED):
        problems.append(f"sample {p.run.get('sample')} seed {p.run.get('sample_seed')}, "
                        f"expected {O.REFERENCE_SIZE} seed {O.REFERENCE_SEED} (s5.3)")
    if problems:
        raise SystemExit(f"error: {what} ({p.root}): " + "; ".join(problems))


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse",
                               "HEAD"], capture_output=True, text=True, timeout=10
                              ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--internal", required=True, type=Path,
                    help="The internal pass's predictions directory.")
    ap.add_argument("--manifest-full", required=True, type=Path)
    ap.add_argument("--manifest-ddr", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    cal = load_pass(args.internal / "calibration")
    check_pass(cal, "calibration", False, "the calibration pass")
    refs = {}
    for variant, job in REFERENCES.items():
        refs[variant] = load_pass(args.internal / job)
        check_pass(refs[variant], "train", True, f"the {variant} OOD reference")
    manifests = {"eyepacs_full": args.manifest_full, "eyepacs_ddr_full": args.manifest_ddr}

    # pi_src: the grade mix of the EyePACS training rows, identical in both variants (s4.2).
    full = pd.read_csv(args.manifest_full)
    train = full[(full["split"].astype(str) == "train")
                 & (full["dataset"].astype(str) == "EyePACS")]
    pi_counts = np.bincount(train["grade"].to_numpy().astype(int), minlength=5)
    pi_src = grade_prior(train["grade"].to_numpy())

    y = labels_for(cal.ids, args.manifest_full, split="calibration")
    images = cal.images
    if not (images["max_excludable_grade"] == G.MAX_EXCLUDABLE_GRADE).all():
        raise SystemExit("error: M3's max_excludable_grade is not 2 everywhere; the pinned "
                         "d_evidence rule assumes count-only M3 (D1)")
    e = images["evidence_grade"].to_numpy().astype(int)
    ctrl_cols = [f"e_c{j:02d}" for j in range(F.CONTROLS)]

    out = args.out
    (out / "ood").mkdir(parents=True, exist_ok=True)
    models: Dict[str, dict] = {}
    print(f"calibration: {len(images)} images, {len(cal.models)} models; "
          f"pi_src from {len(train)} EyePACS training rows: {np.round(pi_src, 4).tolist()}\n")
    header = (f"{'model':<26}{'T':>7}{'T(1)':>7}{'NLL raw':>9}{'NLL cal':>9}{'ECE raw':>9}"
              f"{'ECE cal':>9}{'shrink':>8}{'tau_ood':>9}{'tau_conf':>9}{'r':>5}")
    print(header + "\n" + "-" * len(header))
    for name in cal.models:
        variant, seed = parse_model(name)
        rows = cal.model_rows(name)
        z = rows[[f"z{j}" for j in range(4)]].to_numpy(dtype=np.float64)
        yhat = rows["yhat"].to_numpy().astype(int)
        # In float32, as the pass thresholded them: a logit within ~1e-7 of zero can
        # land on either side of 0.5 depending on precision.
        if not np.array_equal(cumulative_to_grade(torch.tensor(z, dtype=torch.float32)).numpy(),
                              yhat):
            raise SystemExit(f"error: {name}: stored y-hat is not the threshold count of z")
        correct = (yhat == y).astype(float)

        t01 = fit_temperature(z, y, pi_src)
        t1 = fit_temperature(z, y)
        quality = calibration_quality(z, yhat, y, t01, t1, pi_src)

        ref = refs[variant]
        if name not in ref.models:
            raise SystemExit(f"error: {name} is missing from the {variant} reference pass")
        if ref.checkpoint_sha256(name) != cal.checkpoint_sha256(name):
            raise SystemExit(f"error: {name}: the reference and calibration passes used "
                             "different checkpoints")
        y_ref = labels_for(ref.ids, manifests[variant], split="train")
        gauss = O.fit_class_gaussian(ref.embeddings(name), y_ref)
        stats_file = Path("ood") / f"{name}.npz"
        stats_digest = O.save_gaussian(gauss, out / stats_file)
        dist = O.min_mahalanobis(cal.embeddings(name), gauss)
        scale = O.fit_scale(dist)
        z_ood = O.z_score(dist, scale)

        d_conf = 1.0 - predicted_confidence(stage01(z, t01, pi_src), yhat)
        tau_conf = float(np.percentile(d_conf, G.CONF_PERCENTILE))

        outcome = G.faith_outcome(rows["e_orig"].to_numpy(), rows["e_lesion"].to_numpy(),
                                  rows[ctrl_cols].to_numpy())
        d_fa = G.d_faith(outcome)
        d_ev = G.d_evidence(yhat, e, images["max_excludable_grade"].to_numpy())
        r, aucs = choose_r(d_ev, d_fa, correct)

        models[name] = {
            "variant": variant, "seed": seed,
            "checkpoint_sha256": cal.checkpoint_sha256(name),
            "temperature": t01, "temperature_stage1_only": t1,
            "calibration_fit": {"n": int(len(y)), **quality},
            "ood": {"file": stats_file.as_posix(), "digest": stats_digest,
                    "reference_rows": int(len(y_ref)),
                    "class_counts": gauss.class_counts.tolist(), "shrinkage": gauss.shrinkage,
                    "distance_mean": scale.mean, "distance_sd": scale.sd, "tau_ood": scale.tau,
                    "calibration_flagged": float((z_ood >= scale.tau).mean())},
            "tau_conf": tau_conf,
            "r": r, "r_auc": {str(k): v for k, v in aucs.items()},
            "calibration_signals": {
                "accuracy": float(correct.mean()),
                "faith": {label: int((outcome == code).sum()) for label, code in
                          (("faithful", G.FAITHFUL), ("unfaithful", G.UNFAITHFUL),
                           ("undetermined", G.UNDETERMINED), ("no_lesion", G.NO_LESION))},
                "d_evidence": {str(v): int((d_ev == v).sum()) for v in range(5)},
            },
        }
        print(f"{name:<26}{t01:>7.3f}{t1:>7.3f}{quality['nll']['raw']:>9.4f}"
              f"{quality['nll']['stage01']:>9.4f}{quality['ece']['raw']:>9.4f}"
              f"{quality['ece']['stage01']:>9.4f}{gauss.shrinkage:>8.4f}{scale.tau:>9.3f}"
              f"{tau_conf:>9.4f}{r:>5}")

    params = {
        "plan": PLAN,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "pi_src": pi_src.tolist(), "pi_src_counts": pi_counts.tolist(),
        "pi_src_source": f"{args.manifest_full.name}: dataset == EyePACS, split == train",
        "constants": constants(),
        "evidence_checkpoint_sha256": (cal.run.get("evidence_checkpoint") or {}).get("sha256"),
        "pass_constants": cal.run.get("constants"),
        "inputs": {"calibration": cal.fingerprint(),
                   **{REFERENCES[v]: refs[v].fingerprint() for v in REFERENCES},
                   "manifests": {p.name: sha256_file(p) for p in manifests.values()}},
        "models": models,
    }
    digest = write_params(params, out / PARAMS_NAME)
    print(f"\nwrote {out / PARAMS_NAME}\n  digest {digest}")
    print("  r chosen per model:", {m: v["r"] for m, v in models.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
