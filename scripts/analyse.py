#!/usr/bin/env python
"""Every table in ANALYSIS_PLAN.md ss4-8 for one dataset; then the verdicts across them.

    # step 2 -- the rehearsal on val, labelled as such (role external exercises EM too)
    python scripts/analyse.py dataset --pass-dir predictions/val --labels eyepacs_full.csv \\
        --split val --fitted fitted/fitted_params.json --ood-dir fitted \\
        --name val --role external --rehearsal --out analysis/val

    # step 5 -- the unblinding, once per locked set, with committed parameters
    python scripts/analyse.py dataset --pass-dir locked/aptos --labels aptos_external.csv \\
        --fitted preregistration/fitted_params.json --ood-dir <verify-dr-fitted> \\
        --name aptos --role external --unblind --out analysis/aptos

    python scripts/analyse.py verdicts --in-domain analysis/eyepacs_test/results.json \\
        --external analysis/aptos/results.json analysis/messidor2/results.json --out analysis

**Locked data needs --unblind, and --unblind needs committed parameters.** A pass over
the EyePACS test split, APTOS or Messidor-2 is refused before any label is read unless
--unblind is given, and --unblind is refused unless fitted_params.json is tracked by
git and unmodified -- the order s10 fixes: commit, then unblind.

Scores never see a label: every signal is computed from the pass and the fitted
parameters; labels enter only to score the result. y-hat is M1's threshold count on
raw logits in every arm (s3).

**Two analyses, one label join.** The registered analysis is the primary. The amended
one (D12, D13; the plan's addendum) is computed from the same pass and labels, with
the same bootstrap indices, and reported beside it as a deviation analysis.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_dr.calibration import (  # noqa: E402
    BCTS, bcts_probs, coral_probs, em_prior, grade_prior, nll, predicted_confidence,
    prior_correct, stage01,
)
from verify_dr.reasoning import operating_point as OP  # noqa: E402
from verify_dr.evaluation.metrics import (  # noqa: E402
    brier_score, expected_calibration_error, quadratic_weighted_kappa,
)
from verify_dr.triage import claims as C  # noqa: E402
from verify_dr.triage import faithfulness as F  # noqa: E402
from verify_dr.triage import ood as O  # noqa: E402
from verify_dr.triage import selective as S  # noqa: E402
from verify_dr.triage import signals as G  # noqa: E402
from verify_dr.triage.params import load_params  # noqa: E402
from verify_dr.triage.passdata import (  # noqa: E402
    PRIMARY_VARIANT, REPLICATION_VARIANT, labels_for, load_pass,
)

ECE_BINS = 15
ARMS = ("none", "confidence", "ood", "disagreement", "combined")
D4_SIZES = (50, 200, 500)          # plus "all" (s4.4)
D4_DRAWS = 20
LOCKED_DATASETS = {"aptos", "messidor2", "messidor-2", "messidor"}
EXIT_REFUSED = 2


# ------------------------------------------------------------------ guards


def locked_reasons(run: dict, images, split: Optional[str], labels: Path) -> List[str]:
    reasons = []
    if run.get("locked"):
        reasons.append("the pass was run with --locked")
    if split == "test" or run.get("split") == "test":
        reasons.append("the EyePACS test split")
    hit = sorted({d for d in images["dataset"].astype(str) if d.lower() in LOCKED_DATASETS})
    if hit:
        reasons.append(f"locked dataset(s) {hit}")
    if "external" in labels.name.lower():
        reasons.append(f"an external manifest ({labels.name})")
    return reasons


def committed(path: Path) -> Optional[str]:
    """None if `path` is tracked by git and unmodified; otherwise why not."""
    path = Path(path).resolve()
    try:
        tracked = subprocess.run(["git", "-C", str(path.parent), "ls-files", "--error-unmatch",
                                  path.name], capture_output=True, text=True, timeout=20)
        if tracked.returncode != 0:
            return "it is not tracked by git"
        clean = subprocess.run(["git", "-C", str(path.parent), "diff", "--quiet", "HEAD", "--",
                                path.name], capture_output=True, text=True, timeout=20)
        if clean.returncode != 0:
            return "it differs from the committed version"
    except (OSError, subprocess.SubprocessError) as err:
        return f"git could not be asked ({err})"
    return None


def git_commit(path: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(Path(path).resolve().parent), "rev-parse",
                               "HEAD"], capture_output=True, text=True, timeout=10
                              ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------ one model


def calibration_row(p: np.ndarray, yhat: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    conf = predicted_confidence(p, yhat)
    return {"ece": expected_calibration_error(conf, (yhat == y).astype(float), ECE_BINS),
            "nll": nll(p, y), "brier": brier_score(p, y)}


def describe(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(med),
            "q25": float(q25), "q75": float(q75), "share_positive": float((v > 0).mean())}


def faithfulness_tables(e_orig, e_lesion, e_ctrl, outcome, y) -> Dict[str, object]:
    """E1-E3 (s7): Delta_lesion, against the mean control Delta, and by true grade.
    Images with no detected lesion are excluded and counted."""
    has = np.isfinite(e_lesion)
    delta_lesion = e_orig - e_lesion
    ctrl = e_orig[:, None] - e_ctrl
    ok = np.isfinite(ctrl).sum(axis=1)
    mean_ctrl = np.where(ok > 0, np.nansum(np.where(np.isfinite(ctrl), ctrl, 0.0), axis=1)
                         / np.maximum(ok, 1), np.nan)
    excess = delta_lesion - mean_ctrl
    determined = (outcome == G.FAITHFUL) | (outcome == G.UNFAITHFUL)

    def block(mask):
        d = determined & mask
        return {"n": int(mask.sum()),
                "delta_lesion": describe(delta_lesion[mask]),
                "delta_minus_mean_control": describe(excess[mask & (ok > 0)]),
                "determined": int(d.sum()),
                "faithful_share": float((outcome[d] == G.FAITHFUL).mean()) if d.any() else None}

    return {"counts": {"no_lesion": int((~has).sum()),
                       "undetermined": int((outcome == G.UNDETERMINED).sum()),
                       "determined": int(determined.sum()),
                       "no_successful_control": int((has & (ok == 0)).sum())},
            "E1_E2": block(has),
            "E3_by_true_grade": {str(g): block(has & (y == g)) for g in range(5)}}


def blocks(d_ev: np.ndarray, d_fa: np.ndarray, correct: np.ndarray) -> Dict[str, dict]:
    """Descriptive: how many cases each disagreement level holds and how often M1 is
    right there. Not a registered outcome; it shows what the ranking is made of."""
    out = {}
    for name, values in (("d_evidence", d_ev), ("d_faith", d_fa)):
        out[name] = {str(int(v)): {"n": int((values == v).sum()),
                                   "share": float((values == v).mean()),
                                   "m1_accuracy": float(correct[values == v].mean())}
                     for v in np.unique(values)}
    return out


def analyse_model(name: str, rows, emb: np.ndarray, images, y: np.ndarray, params: dict,
                  gauss: O.ClassGaussian, external: bool, amended: bool = False) -> tuple:
    """Every per-model table. `amended` switches exactly three things (addendum A.1-A.2):
    M3's evidence grade (D12), the d_faith gate and r that go with it, and the
    probabilities the calibration outcomes and H2 are computed on (D13). d_conf,
    tau_conf and the OOD values are the registered ones in both analyses."""
    m = params["models"][name]
    pi_src = np.asarray(params["pi_src"])
    z = rows[[f"z{j}" for j in range(4)]].to_numpy(dtype=np.float64)
    yhat = rows["yhat"].to_numpy().astype(int)
    correct = (yhat == y).astype(np.float64)

    p01 = stage01(z, m["temperature"], pi_src)
    calib = {"raw": calibration_row(coral_probs(z), yhat, y),
             "stage1": calibration_row(coral_probs(z, m["temperature_stage1_only"]), yhat, y),
             "stage01": {**calibration_row(p01, yhat, y),
                         "mean_posterior": p01.mean(axis=0).tolist()}}
    if amended:
        a = m["amended"]
        base = bcts_probs(z, BCTS(a["bcts"]["temperature"], np.asarray(a["bcts"]["bias"])))
        base_key, pi_ref = "bcts", np.asarray(params["amended"]["pi_cal"])
        calib["bcts"] = {**calibration_row(base, yhat, y),
                         "mean_posterior": base.mean(axis=0).tolist()}
    else:
        base, base_key, pi_ref = p01, "stage01", pi_src
    calib["h2_base"] = base_key
    if external:
        em = em_prior(base, pi_ref)
        oracle = grade_prior(y)
        calib["em"] = {**calibration_row(prior_correct(base, em.prior, pi_ref), yhat, y),
                       "prior": em.prior.tolist(), "iterations": em.iterations,
                       "converged": em.converged}
        calib["oracle"] = {**calibration_row(prior_correct(base, oracle, pi_ref), yhat, y),
                           "prior": oracle.tolist()}
        d4 = {}
        for size in D4_SIZES:
            if size > len(y):
                continue
            eces = []
            for draw in range(D4_DRAWS):
                sub = np.random.default_rng([size, draw]).choice(len(y), size, replace=False)
                prior = em_prior(base[sub], pi_ref).prior
                eces.append(calibration_row(prior_correct(base, prior, pi_ref), yhat, y)["ece"])
            d4[str(size)] = {"mean": float(np.mean(eces)), "sd": float(np.std(eces, ddof=1)),
                             "min": float(np.min(eces)), "max": float(np.max(eces))}
        d4["all"] = {"mean": calib["em"]["ece"], "sd": 0.0}
        calib["D4_ece_by_sample_size"] = d4

    d_conf = 1.0 - predicted_confidence(p01, yhat)
    scale = O.OODScale(m["ood"]["distance_mean"], m["ood"]["distance_sd"], m["ood"]["tau_ood"])
    z_ood = O.z_score(O.min_mahalanobis(emb, gauss), scale)
    ctrl_cols = [f"e_c{j:02d}" for j in range(F.CONTROLS)]
    e_orig, e_lesion = rows["e_orig"].to_numpy(), rows["e_lesion"].to_numpy()
    e_ctrl = rows[ctrl_cols].to_numpy(dtype=np.float64)
    outcome = G.faith_outcome(e_orig, e_lesion, e_ctrl)
    d_fa = G.d_faith(outcome)
    if amended:
        e, _ = OP.evidence(images, params["amended"]["evidence"]["area_min"])
        d_fa = d_fa * (e >= 1)                  # nothing cited, nothing tested (A.1)
        r = m["amended"]["r"]
    else:
        e = images["evidence_grade"].to_numpy().astype(int)
        r = m["r"]
    d_ev = G.d_evidence(yhat, e, images["max_excludable_grade"].to_numpy())
    level = G.combined_level(d_ev, d_fa, z_ood, m["ood"]["tau_ood"], e, d_conf, m["tau_conf"])

    ranks = {"none": S.dense_rank(np.zeros(len(y))),
             "confidence": S.dense_rank(d_conf),
             "ood": S.dense_rank(z_ood),
             "disagreement": S.dense_rank(G.disagreement(d_ev, d_fa, r)),
             "combined": S.dense_rank(level, d_conf)}
    arms = {}
    for arm, rank in ranks.items():
        arms[arm] = {**S.summary(rank, correct),
                     **{f"F3@{int(round(c * 100))}": asdict(S.referral_at(rank, yhat, y, c))
                        for c in S.COVERAGE_POINTS}}
    binary = S.dense_rank(G.disagreement((d_ev > 0).astype(int), d_fa, r))
    result = {
        "variant": m["variant"], "seed": m["seed"], "r": r,
        "accuracy": float(correct.mean()), "qwk": quadratic_weighted_kappa(y, yhat),
        "calibration": calib,
        "arms": arms,
        "F4": {"magnitude": arms["disagreement"], "binary": S.summary(binary, correct),
               "magnitude_plus_unobservable": "identical to magnitude: the set never varies"},
        "F5": G.action_table(level, yhat, y),
        "signals": {"ood_flagged": float((z_ood >= m["ood"]["tau_ood"]).mean()),
                    "conf_flagged": float((d_conf >= m["tau_conf"]).mean()),
                    "blocks": blocks(d_ev, d_fa, correct)},
        "faithfulness": faithfulness_tables(e_orig, e_lesion, e_ctrl, outcome, y),
    }
    curves = {arm: S.accuracy_curve(rank, correct).astype(np.float32)
              for arm, rank in ranks.items()}
    boot = {"ranks": ranks, "correct": correct, "p_h2": base, "pi_ref": pi_ref, "yhat": yhat,
            "sizes": {arm: int(rank.max()) + 1 for arm, rank in ranks.items()}}
    return result, curves, boot


def evidence_table(e: np.ndarray, rules, y: np.ndarray) -> Dict[str, object]:
    """Descriptive, not registered: M3's evidence grade against the true grade.

    The disagreement signal is only as specific as M3. This says how often M3 finds
    evidence of disease in images graded 0, and misses it in referable ones.
    """
    e = np.asarray(e).astype(int)
    rules = np.asarray(rules).astype(str)
    confusion = np.zeros((5, 3), dtype=int)
    np.add.at(confusion, (y, np.clip(e, 0, 2)), 1)
    grade0, referable = y == 0, y >= 2
    return {
        "confusion_true_by_evidence": confusion.tolist(),
        "m3_qwk": quadratic_weighted_kappa(y, e),
        "m3_exact": float((e == y).mean()),
        "evidence_any_given_grade0": float((e[grade0] >= 1).mean()) if grade0.any() else None,
        "evidence2_given_grade0": float((e[grade0] == 2).mean()) if grade0.any() else None,
        "evidence0_given_referable": float((e[referable] == 0).mean()) if referable.any() else None,
        "rules": {str(k): int((rules == k).sum()) for k in sorted(set(rules))},
        "evidence_distribution": {str(v): int((e == v).sum()) for v in range(3)},
    }


# ------------------------------------------------------------------ bootstrap


def bootstrap(models: Dict[str, dict], y: np.ndarray, external: bool,
              resamples: int, seed: int) -> Dict[str, dict]:
    """One set of resample indices, shared by every arm and model (s8).

    Under an external role EM is re-run inside every resample: the target prior is
    estimated from the images, so its uncertainty belongs in H2's interval. Holding
    it fixed would narrow the interval -- the reading that favours H2 (s1, rule 3).
    """
    n = len(y)
    harmonic = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])
    out = {name: {"auc": {arm: np.empty(resamples) for arm in ARMS},
                  "ece_gain": np.empty(resamples) if external else None,
                  "qwk": np.empty(resamples)} for name in models}
    started = time.time()
    for b, idx in enumerate(C.resample_indices(n, resamples, seed)):
        yb = y[idx]
        for name, m in models.items():
            c = m["correct"][idx]
            for arm in ARMS:
                rk = m["ranks"][arm][idx]
                size = m["sizes"][arm]
                out[name]["auc"][arm][b] = S.auc_from_groups(
                    np.bincount(rk, minlength=size), np.bincount(rk, weights=c, minlength=size),
                    harmonic)
            yh = m["yhat"][idx]
            out[name]["qwk"][b] = quadratic_weighted_kappa(yb, yh)
            if external:
                pb, pi_ref = m["p_h2"][idx], m["pi_ref"]
                prior = em_prior(pb, pi_ref).prior
                before = expected_calibration_error(predicted_confidence(pb, yh), c, ECE_BINS)
                after = expected_calibration_error(
                    predicted_confidence(prior_correct(pb, prior, pi_ref), yh), c, ECE_BINS)
                out[name]["ece_gain"][b] = before - after
        if (b + 1) % 250 == 0:
            print(f"  bootstrap {b + 1}/{resamples}  ({time.time() - started:.0f}s)", flush=True)
    return out


def effects(results: Dict[str, dict], boot: Dict[str, dict], external: bool) -> Dict[str, dict]:
    """Per-seed effects with paired 95% intervals, and the s8 claim rule on this dataset."""
    by_variant: Dict[str, Dict[int, str]] = {}
    for name, r in results.items():
        by_variant.setdefault(r["variant"], {})[r["seed"]] = name
    out = {}
    for variant, seeds in by_variant.items():
        entry = {}
        auc_gain = {}
        for seed, name in sorted(seeds.items()):
            arms = results[name]["arms"]
            samples = boot[name]["auc"]["disagreement"] - boot[name]["auc"]["confidence"]
            auc_gain[seed] = (arms["disagreement"]["auc"] - arms["confidence"]["auc"],
                              C.interval(samples))
        entry["auc_gain"] = {"per_seed": {str(s): {"effect": e, "interval": list(iv)}
                                          for s, (e, iv) in auc_gain.items()},
                             **C.claim([e for e, _ in auc_gain.values()],
                                       [iv for _, iv in auc_gain.values()])}
        if external:
            ece_gain = {}
            for seed, name in sorted(seeds.items()):
                cal = results[name]["calibration"]
                ece_gain[seed] = (cal[cal["h2_base"]]["ece"] - cal["em"]["ece"],
                                  C.interval(boot[name]["ece_gain"]))
            entry["ece_gain"] = {"per_seed": {str(s): {"effect": e, "interval": list(iv)}
                                              for s, (e, iv) in ece_gain.items()},
                                 **C.claim([e for e, _ in ece_gain.values()],
                                           [iv for _, iv in ece_gain.values()])}
        out[variant] = entry
    if external and {PRIMARY_VARIANT, REPLICATION_VARIANT} <= set(by_variant):
        pairs = sorted(set(by_variant[PRIMARY_VARIANT]) & set(by_variant[REPLICATION_VARIANT]))
        qwk_gain = {}
        for seed in pairs:
            full, ddr = by_variant[PRIMARY_VARIANT][seed], by_variant[REPLICATION_VARIANT][seed]
            qwk_gain[seed] = (results[ddr]["qwk"] - results[full]["qwk"],
                              C.interval(boot[ddr]["qwk"] - boot[full]["qwk"]))
        if qwk_gain:
            out["H3_qwk_gain"] = {"per_seed": {str(s): {"effect": e, "interval": list(iv)}
                                               for s, (e, iv) in qwk_gain.items()},
                                  **C.claim([e for e, _ in qwk_gain.values()],
                                            [iv for _, iv in qwk_gain.values()])}
    return out


# ------------------------------------------------------------------ commands


def run_dataset(args) -> int:
    params = load_params(args.fitted)
    p = load_pass(args.pass_dir)
    external = args.role == "external"

    reasons = locked_reasons(p.run, p.images, args.split, args.labels)
    if reasons and not args.unblind:
        print("REFUSED: locked data -- " + "; ".join(reasons) + ". No label was read.\n"
              "  Locked sets are scored once, at the unblinding, with --unblind "
              "(ANALYSIS_PLAN.md s10 step 5).", file=sys.stderr)
        return EXIT_REFUSED
    if args.unblind:
        if args.rehearsal:
            print("REFUSED: --rehearsal and --unblind together -- a rehearsal never "
                  "reads locked labels.", file=sys.stderr)
            return EXIT_REFUSED
        why = committed(args.fitted)
        if why:
            print(f"REFUSED: --unblind needs committed parameters, and {args.fitted}: {why}. "
                  "Commit fitted_params.json first (s10 step 3). No label was read.",
                  file=sys.stderr)
            return EXIT_REFUSED

    # ---- the pass must be the one the parameters were fitted for ----------------
    problems = []
    if p.run.get("constants") != params.get("pass_constants"):
        problems.append("the pass constants differ from the calibration pass's")
    ev = (p.run.get("evidence_checkpoint") or {}).get("sha256")
    if ev != params.get("evidence_checkpoint_sha256"):
        problems.append("M2's checkpoint differs")
    missing = [m for m in p.models if m not in params["models"]]
    if missing:
        problems.append(f"no fitted parameters for {missing}")
    for name in p.models:
        if name in params["models"] and \
                p.checkpoint_sha256(name) != params["models"][name]["checkpoint_sha256"]:
            problems.append(f"{name}'s checkpoint differs from the fitted one")
    if not p.has_evidence:
        problems.append("the pass has no M2/M3/faithfulness columns (embeddings only?)")
    if problems:
        print("error: " + "; ".join(problems), file=sys.stderr)
        return 1

    gaussians = {name: O.load_gaussian(args.ood_dir / params["models"][name]["ood"]["file"],
                                       params["models"][name]["ood"]["digest"])
                 for name in p.models}

    # ---- the one label join -------------------------------------------------------
    y = labels_for(p.ids, args.labels, split=args.split)

    analyses = {"registered": False}
    if "amended" in params:
        analyses["amended"] = True
    per, curves = {}, {}
    for label, amended in analyses.items():
        results, boot_in = {}, {}
        for name in p.models:
            res, cur, bt = analyse_model(name, p.model_rows(name), p.embeddings(name), p.images,
                                         y, params, gaussians[name], external, amended)
            results[name], boot_in[name] = res, bt
            prefix = "amended__" if amended else ""
            curves.update({f"{prefix}{name}__{arm}": c for arm, c in cur.items()})
        print(f"bootstrap, {label} analysis: {args.resamples} resamples over {len(y)} images, "
              f"seed {C.SEED}", flush=True)
        boot = bootstrap(boot_in, y, external, args.resamples, C.SEED)
        for name, res in results.items():
            for arm in ARMS:
                res["arms"][arm]["auc_interval"] = list(C.interval(boot[name]["auc"][arm]))
        per[label] = {"models": results, "claims_on_this_dataset": effects(results, boot, external)}

    out = {
        "name": args.name, "role": args.role, "rehearsal": bool(args.rehearsal),
        "unblinded": bool(args.unblind), "n": int(len(y)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(Path(__file__)),
        "params_digest": params["digest"], "pass": str(args.pass_dir),
        "pass_fingerprint": p.fingerprint(), "resamples": args.resamples,
        "true_grade_distribution": np.bincount(y, minlength=5).tolist(),
        "evidence": evidence_table(p.images["evidence_grade"].to_numpy(),
                                   p.images["rule"].to_numpy(), y),
        **per["registered"],
    }
    if "amended" in per:
        area_min = params["amended"]["evidence"]["area_min"]
        e_a, rule_a = OP.evidence(p.images, area_min)
        out["amended"] = {"deviations": params["amended"]["deviations"],
                          "evidence_area_min": area_min,
                          "evidence": evidence_table(e_a, rule_a, y), **per["amended"]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(args.out / "curves.npz", **curves)
    print_dataset(out)
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


def verdict_table(inside: dict, outside: Dict[str, dict]) -> dict:
    """The s8 verdicts from one analysis's per-dataset claims."""
    table = {}
    for variant, label in ((PRIMARY_VARIANT, "primary"), (REPLICATION_VARIANT, "replication")):
        h1 = inside.get(variant, {}).get("auc_gain")
        h1p = {n: c.get(variant, {}).get("auc_gain") for n, c in outside.items()}
        h2 = {n: c.get(variant, {}).get("ece_gain") for n, c in outside.items()}
        table[label] = {
            "variant": variant,
            "H1": {"supported": bool(h1 and h1["supported"]), "detail": h1},
            "H1_prime": {"supported": C.on_every_dataset({k: v for k, v in h1p.items() if v})
                         and all(h1p.values()), "detail": h1p},
            "H2": {"supported": C.on_every_dataset({k: v for k, v in h2.items() if v})
                   and all(h2.values()), "detail": h2},
        }
    h3 = {n: c.get("H3_qwk_gain") for n, c in outside.items()}
    table["H3"] = {"supported": C.on_every_dataset({k: v for k, v in h3.items() if v})
                   and all(h3.values()), "detail": h3}
    return table


def run_verdicts(args) -> int:
    inside = json.loads(Path(args.in_domain).read_text())
    outside = {}
    for f in args.external:
        res = json.loads(Path(f).read_text())
        outside[res["name"]] = res
    rehearsal = inside["rehearsal"] or any(r["rehearsal"] for r in outside.values())

    verdicts = {"rehearsal": rehearsal, "in_domain": inside["name"], "external": list(outside),
                "registered": verdict_table(inside["claims_on_this_dataset"],
                                            {n: r["claims_on_this_dataset"]
                                             for n, r in outside.items()}),
                "amended": None}
    if "amended" in inside and all("amended" in r for r in outside.values()):
        verdicts["amended"] = verdict_table(
            inside["amended"]["claims_on_this_dataset"],
            {n: r["amended"]["claims_on_this_dataset"] for n, r in outside.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=float))

    tag = "REHEARSAL -- not a result. " if rehearsal else ""
    print(f"{tag}Verdicts under the s8 claim rule "
          f"(in-domain: {inside['name']}; external: {', '.join(outside)})")
    for key, title in (("registered", "REGISTERED ANALYSIS -- the primary result"),
                       ("amended", "AMENDED ANALYSIS -- deviations D12, D13, reported beside it")):
        table = verdicts[key]
        if table is None:
            continue
        print(f"\n{title}")
        for label in ("primary", "replication"):
            v = table[label]
            print(f"  {label} variant ({v['variant']}):")
            for h, text in (("H1", "H1  disagreement beats confidence in-domain"),
                            ("H1_prime", "H1' ...and on every external set (the thesis)"),
                            ("H2", "H2  EM improves external calibration")):
                print(f"    {text:<48} {'SUPPORTED' if v[h]['supported'] else 'not supported'}")
        print(f"  H3  DDR in training improves external QWK        "
              f"{'SUPPORTED' if table['H3']['supported'] else 'not supported'}")
    return 0


# ------------------------------------------------------------------ printing


def print_analysis(block: dict, role: str, calibrated: str) -> None:
    """One analysis's tables: M3 against the truth, the arms, calibration, the claims."""
    ev = block["evidence"]
    print("\nM3's evidence grade against the true grade (descriptive, not registered):")
    print("  true \\ evidence      0       1       2")
    for g, row in enumerate(ev["confusion_true_by_evidence"]):
        print(f"  grade {g}     " + "".join(f"{v:>8}" for v in row))
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
    print(f"  M3 QWK {ev['m3_qwk']:.3f}; evidence found in {pct(ev['evidence_any_given_grade0'])} "
          f"of grade-0 images; none found in {pct(ev['evidence0_given_referable'])} of "
          "referable ones")

    print("\nCoverage-accuracy AUC per arm, and what the ranking is made of:")
    head = f"  {'model':<26}" + "".join(f"{a:>14}" for a in ARMS) + f"{'dis - conf':>12}"
    print(head)
    for name, r in block["models"].items():
        a = r["arms"]
        print(f"  {name:<26}" + "".join(f"{a[x]['auc']:>14.4f}" for x in ARMS)
              + f"{a['disagreement']['auc'] - a['confidence']['auc']:>+12.4f}")
    for name, r in block["models"].items():
        b = r["signals"]["blocks"]["d_evidence"]
        parts = ", ".join(f"d_ev={k}: {v['share']:.0%} (M1 acc {v['m1_accuracy']:.2f})"
                          for k, v in b.items())
        print(f"  {name:<26} r={r['r']:<4} {parts}")

    print("\nCalibration: ECE raw -> stage 1 -> stages 0+1"
          + (" -> BCTS" if calibrated == "bcts" else "")
          + (" -> EM (oracle)" if role == "external" else ""))
    for name, r in block["models"].items():
        c = r["calibration"]
        line = f"  {name:<26} {c['raw']['ece']:.4f} -> {c['stage1']['ece']:.4f} -> " \
               f"{c['stage01']['ece']:.4f}"
        if calibrated == "bcts":
            line += f" -> {c['bcts']['ece']:.4f}"
        if "em" in c:
            line += f" -> {c['em']['ece']:.4f} ({c['oracle']['ece']:.4f})"
        print(line)
    if role == "external":
        print("\nEM's estimate of the grade mix against the truth (EM needs the mean calibrated")
        print("posterior to match the prior; where it does not, EM drifts even with no shift):")
        fmt = lambda v: "[" + " ".join(f"{x:.3f}" for x in v) + "]"  # noqa: E731
        for name, r in block["models"].items():
            c = r["calibration"]
            print(f"  {name:<26} EM {fmt(c['em']['prior'])} true {fmt(c['oracle']['prior'])}"
                  f"  ({c['em']['iterations']} iterations)")

    print("\nClaim rule on this dataset (s8): per-seed effect [95% interval] -> verdict")
    for variant, entry in block["claims_on_this_dataset"].items():
        items = [("qwk_gain", entry)] if variant == "H3_qwk_gain" else list(entry.items())
        for key, v in items:
            seeds = "  ".join(f"s{s} {d['effect']:+.4f} [{d['interval'][0]:+.4f}, "
                              f"{d['interval'][1]:+.4f}]" for s, d in v["per_seed"].items())
            verdict = "SUPPORTED" if v["supported"] else (
                "not supported (" + ", ".join(k for k in ("direction", "precision", "stability")
                                              if not v[k]) + " failed)")
            print(f"  {variant:<18} {key:<9} {seeds}  -> {verdict}")


def print_dataset(out: dict) -> None:
    tag = "REHEARSAL -- not a result. " if out["rehearsal"] else ""
    print(f"\n{tag}{out['name']}: {out['n']} images, role {out['role']}, "
          f"true grades {out['true_grade_distribution']}")
    print("\n" + "=" * 30 + " REGISTERED ANALYSIS (primary) " + "=" * 30)
    print_analysis(out, out["role"], "stage01")
    if "amended" in out:
        a = out["amended"]
        print("\n" + "=" * 22 + " AMENDED ANALYSIS (deviations D12, D13) " + "=" * 22)
        print(f"M3 area minimums (D12): {a['evidence_area_min']}")
        print_analysis(a, out["role"], "bcts")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dataset", help="Every table for one dataset.")
    d.add_argument("--pass-dir", required=True, type=Path)
    d.add_argument("--labels", required=True, type=Path, help="The manifest holding the grades.")
    d.add_argument("--split", default=None)
    d.add_argument("--fitted", required=True, type=Path)
    d.add_argument("--ood-dir", required=True, type=Path,
                   help="Where fit_params.py wrote ood/<model>.npz.")
    d.add_argument("--name", required=True)
    d.add_argument("--role", required=True, choices=("in_domain", "external"))
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--rehearsal", action="store_true", help="Label every output as rehearsal.")
    d.add_argument("--unblind", action="store_true",
                   help="Score a locked set. Once, after fitted_params.json is committed.")
    d.add_argument("--resamples", type=int, default=C.RESAMPLES)
    v = sub.add_parser("verdicts", help="The s8 verdicts across datasets.")
    v.add_argument("--in-domain", required=True, type=Path)
    v.add_argument("--external", required=True, type=Path, nargs="+")
    v.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.command == "dataset":
        if args.resamples != C.RESAMPLES and not args.rehearsal:
            print(f"error: --resamples {args.resamples} is not the plan's {C.RESAMPLES}; "
                  "only a rehearsal may use fewer", file=sys.stderr)
            return EXIT_REFUSED
        return run_dataset(args)
    return run_verdicts(args)


if __name__ == "__main__":
    sys.exit(main())
