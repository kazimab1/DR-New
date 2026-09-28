#!/usr/bin/env python
"""Phase 8: the error analysis, after the unblinding. EXPLORATORY -- labelled post hoc.

Nothing here can change a registered verdict. Those were fixed by the one label join
(ANALYSIS_PLAN.md s10 step 5) and are retained under PREREGISTRATION.md s9. The
analyses were declared in docs/04_experiment_register.md ("Phase 8") before they first
ran; the numbers P8.0-P8.6 below refer to that declaration.

    # P8.1 -- the secondary outcomes the unblinding computed but did not print
    python scripts/phase8.py secondary --results results/*/results.json

    # P8.0, P8.2-P8.5 -- one locked set at a time
    python scripts/phase8.py errors --pass-dir locked/aptos --labels aptos_external.csv \\
        --fitted preregistration/fitted_params.json --ood-dir <verify-dr-fitted> \\
        --results results/aptos/results.json --out phase8/aptos

    # P8.5 -- draw the figure-1 candidates with M2's lesion outlines (CPU is enough)
    python scripts/phase8.py figures --candidates phase8/*/candidates.csv \\
        --cache-root <cache512> --evidence-checkpoint C2_lesions/best.pt --out phase8/figures

    # P8.6 -- Messidor-2's preprocessing, label-free
    python scripts/phase8.py audit --manifest-dir <manifests> --cache-root <cache512> \\
        --locked <verify-dr-locked> --raw-root <messidor2 mirror> --out phase8/audit

**P8.0 gates everything.** `errors` recomputes every per-image signal the way
analyse.py did and refuses to print a table unless all five arms' AUCs match the
published results.json for every model and both analyses. It also refuses a locked pass
that results.json does not record as unblinded: Phase 8 can never be the first read of
a locked label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_dr.calibration import predicted_confidence, stage01  # noqa: E402
from verify_dr.data.segmentation import LESION_NAMES  # noqa: E402
from verify_dr.reasoning import operating_point as OP  # noqa: E402
from verify_dr.triage import faithfulness as F  # noqa: E402
from verify_dr.triage import ood as O  # noqa: E402
from verify_dr.triage import selective as S  # noqa: E402
from verify_dr.triage import signals as G  # noqa: E402
from verify_dr.triage.params import load_params  # noqa: E402
from verify_dr.triage.passdata import (  # noqa: E402
    PRIMARY_VARIANT, labels_for, load_pass, parse_model,
)

ARMS = ("none", "confidence", "ood", "disagreement", "combined")
ANALYSES = ("registered", "amended")
FIGURE_MODEL = f"H1_{PRIMARY_VARIANT}_s42"     # P8.5: the primary variant's first seed
CANDIDATES_PER_CATEGORY = 8
CONFIDENT_QUANTILE = 0.25                        # "most confident quarter" of y-hat = 0 calls
REPRODUCE_TOL = 1e-12
EXIT_REFUSED = 2

CATEGORIES = {
    "A_figure1_confident_normal_MA_found_truly_diseased":
        "M1 says 0 confidently, M3 finds microaneurysms, the truth is >= 1",
    "B_confident_normal_false_evidence":
        "M1 says 0 confidently and is right, M3 says 2",
    "C_correct_grade2_no_evidence":
        "M1 says 2 and is right, M3 finds nothing",
}


# ------------------------------------------------------------------ helpers


def across(values: Sequence[float]) -> Dict[str, float]:
    """Mean and range over seeds, ignoring NaN."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def fmt(stat: Dict[str, float], digits: int = 3) -> str:
    """'0.812 [0.790-0.830]', or just the value when the range is empty."""
    if not np.isfinite(stat["mean"]):
        return "n/a"
    if abs(stat["max"] - stat["min"]) < 10 ** -(digits + 1):
        return f"{stat['mean']:.{digits}f}"
    return f"{stat['mean']:.{digits}f} [{stat['min']:.{digits}f}-{stat['max']:.{digits}f}]"


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC of `score` for `positive`; ties count one half."""
    from scipy.stats import rankdata

    positive = np.asarray(positive, dtype=bool)
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(np.asarray(score, dtype=np.float64))
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def models_by_variant(names: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in names:
        variant, seed = parse_model(name)
        out.setdefault(variant, []).append((seed, name))
    return {v: [n for _, n in sorted(items)] for v, items in out.items()}


def analysis_block(res: dict, analysis: str) -> dict:
    return res if analysis == "registered" else res["amended"]


def to_plain(obj):
    """numpy scalars and arrays -> JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ------------------------------------------------------------------ P8.1 secondary


def secondary_tables(res: dict) -> dict:
    """Every secondary outcome of one dataset, summarised over each variant's seeds."""
    out = {"name": res["name"], "role": res["role"], "rehearsal": bool(res.get("rehearsal")),
           "n": res["n"], "analyses": {}}
    for analysis in ANALYSES:
        if analysis == "amended" and "amended" not in res:
            continue
        block = analysis_block(res, analysis)
        per_variant = {}
        for variant, names in models_by_variant(block["models"]).items():
            ms = [block["models"][n] for n in names]
            t = {"models": names, "arms": {}, "F4": {}, "F5": {}}
            for arm in ARMS:
                t["arms"][arm] = {
                    "auc": across([m["arms"][arm]["auc"] for m in ms]),
                    **{f"acc@{c}": across([m["arms"][arm][f"acc@{c}"] for m in ms]) for c in (80, 90)},
                    **{f"F3@{c}": {k: across([m["arms"][arm][f"F3@{c}"][k] for m in ms])
                                   for k in ("sensitivity", "specificity", "referable_deferred")}
                       for c in (80, 90)}}
            t["F4"] = {"magnitude": across([m["F4"]["magnitude"]["auc"] for m in ms]),
                       "binary": across([m["F4"]["binary"]["auc"] for m in ms])}
            t["F5"] = {a: {"share": across([m["F5"][a]["share"] for m in ms]),
                           "accuracy": across([m["F5"][a]["accuracy"] for m in ms])}
                       for a in G.ACTIONS}
            cal = [m["calibration"] for m in ms]
            if all("D4_ece_by_sample_size" in c for c in cal):
                sizes = list(cal[0]["D4_ece_by_sample_size"])
                t["D4"] = {s: across([c["D4_ece_by_sample_size"][s]["mean"] for c in cal])
                           for s in sizes}
            if analysis == "registered":
                faith = [m["faithfulness"] for m in ms]
                t["E"] = {
                    "counts": {k: across([f["counts"][k] for f in faith])
                               for k in faith[0]["counts"]},
                    "E1_delta_lesion_median": across([f["E1_E2"]["delta_lesion"].get("median")
                                                      for f in faith]),
                    "E2_excess_mean": across([f["E1_E2"]["delta_minus_mean_control"].get("mean")
                                              for f in faith]),
                    "E2_excess_share_positive": across(
                        [f["E1_E2"]["delta_minus_mean_control"].get("share_positive") for f in faith]),
                    "E2_faithful_share": across([f["E1_E2"]["faithful_share"] for f in faith]),
                    "E3_by_true_grade": {
                        g: {"n": across([f["E3_by_true_grade"][g]["n"] for f in faith]),
                            "faithful_share": across([f["E3_by_true_grade"][g]["faithful_share"]
                                                      for f in faith]),
                            "excess_mean": across([f["E3_by_true_grade"][g]
                                                   ["delta_minus_mean_control"].get("mean")
                                                   for f in faith])}
                        for g in faith[0]["E3_by_true_grade"]}}
            per_variant[variant] = t
        out["analyses"][analysis] = per_variant
    return out


def print_secondary(t: dict) -> None:
    tag = "REHEARSAL -- not a result. " if t["rehearsal"] else ""
    print(f"\n{'#' * 78}\n{tag}{t['name']}: {t['n']} images, role {t['role']}  "
          f"(mean [min-max] over each variant's three seeds)\n{'#' * 78}")
    for analysis, per_variant in t["analyses"].items():
        print(f"\n--- {analysis.upper()} ---")
        for variant, v in per_variant.items():
            print(f"  {variant}: accuracy among accepted cases at fixed coverage")
            for arm in ARMS:
                a = v["arms"][arm]
                print(f"    {arm:<13} acc@80 {fmt(a['acc@80'])}   acc@90 {fmt(a['acc@90'])}")
        v = per_variant[PRIMARY_VARIANT]
        print(f"  F3 ({PRIMARY_VARIANT}): rDR among accepted cases -- sensitivity / specificity"
              " / share of referable deferred")
        for c in (80, 90):
            for arm in ARMS:
                f3 = v["arms"][arm][f"F3@{c}"]
                print(f"    @{c} {arm:<13} {fmt(f3['sensitivity'])}  {fmt(f3['specificity'])}  "
                      f"{fmt(f3['referable_deferred'])}")
        for variant, vv in per_variant.items():
            print(f"  F4 ({variant}): AUC magnitude {fmt(vv['F4']['magnitude'])}   "
                  f"binary {fmt(vv['F4']['binary'])}")
        print(f"  F5 ({PRIMARY_VARIANT}): share and accuracy per action "
              "(ADJACENT: truth within one grade)")
        for action in G.ACTIONS:
            f5 = v["F5"][action]
            print(f"    {action:<19} share {fmt(f5['share'])}   accuracy {fmt(f5['accuracy'])}")
        if "D4" in v:
            print(f"  D4 ({PRIMARY_VARIANT}): ECE after EM on n unlabelled target images: "
                  + "   ".join(f"n={s} {fmt(e)}" for s, e in v["D4"].items()))
        if "E" in v:
            e = v["E"]
            print(f"  E1-E3 ({PRIMARY_VARIANT}), images with a detected lesion:")
            print("    counts: " + ", ".join(f"{k} {fmt(c, 0)}" for k, c in e["counts"].items()))
            print(f"    E1 median delta_lesion {fmt(e['E1_delta_lesion_median'])}")
            print(f"    E2 mean (delta_lesion - mean control delta) {fmt(e['E2_excess_mean'])}; "
                  f"share > 0 {fmt(e['E2_excess_share_positive'])}; "
                  f"faithful share {fmt(e['E2_faithful_share'])}")
            for g, row in e["E3_by_true_grade"].items():
                print(f"    E3 grade {g}: n {fmt(row['n'], 0)}   faithful share "
                      f"{fmt(row['faithful_share'])}   mean excess {fmt(row['excess_mean'])}")


def run_secondary(args) -> int:
    tables = []
    for path in args.results:
        res = json.loads(Path(path).read_text())
        t = secondary_tables(res)
        print_secondary(t)
        tables.append(t)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "secondary.json").write_text(json.dumps(to_plain(tables), indent=1))
        print(f"\nwrote {args.out / 'secondary.json'}")
    return 0


# ------------------------------------------------------------------ P8.0 signals


def signals(name: str, rows, emb: np.ndarray, images: pd.DataFrame, params: dict,
            gauss, amended: bool) -> dict:
    """Every per-image signal of one model, computed exactly as analyse.analyse_model."""
    m = params["models"][name]
    pi_src = np.asarray(params["pi_src"])
    z = rows[[f"z{j}" for j in range(4)]].to_numpy(dtype=np.float64)
    yhat = rows["yhat"].to_numpy().astype(int)
    p01 = stage01(z, m["temperature"], pi_src)
    d_conf = 1.0 - predicted_confidence(p01, yhat)
    scale = O.OODScale(m["ood"]["distance_mean"], m["ood"]["distance_sd"], m["ood"]["tau_ood"])
    z_ood = O.z_score(O.min_mahalanobis(emb, gauss), scale)
    ctrl_cols = [f"e_c{j:02d}" for j in range(F.CONTROLS)]
    outcome = G.faith_outcome(rows["e_orig"].to_numpy(), rows["e_lesion"].to_numpy(),
                              rows[ctrl_cols].to_numpy(dtype=np.float64))
    d_fa = G.d_faith(outcome)
    if amended:
        area_min = params["amended"]["evidence"]["area_min"]
        e, rule = OP.evidence(images, area_min)
        d_fa = d_fa * (e >= 1)
        r = m["amended"]["r"]
    else:
        area_min = OP.FROZEN_AREA_MIN
        e = images["evidence_grade"].to_numpy().astype(int)
        rule = images["rule"].to_numpy().astype(str)
        r = m["r"]
    d_ev = G.d_evidence(yhat, e, images["max_excludable_grade"].to_numpy())
    level = G.combined_level(d_ev, d_fa, z_ood, m["ood"]["tau_ood"], e, d_conf, m["tau_conf"])
    dis = G.disagreement(d_ev, d_fa, r)
    ranks = {"none": S.dense_rank(np.zeros(len(yhat))), "confidence": S.dense_rank(d_conf),
             "ood": S.dense_rank(z_ood), "disagreement": S.dense_rank(dis),
             "combined": S.dense_rank(level, d_conf)}
    return {"yhat": yhat, "d_conf": d_conf, "z_ood": z_ood, "outcome": outcome, "d_fa": d_fa,
            "e": np.asarray(e).astype(int), "rule": np.asarray(rule).astype(str), "d_ev": d_ev,
            "level": level, "dis": dis, "ranks": ranks, "r": r,
            "present": OP.presence(images, area_min)}


def check_results(res: dict, params: dict, p, y: np.ndarray) -> List[str]:
    """Why these results.json cannot be the unblinding of this pass, if they cannot."""
    problems = []
    if res.get("params_digest") != params["digest"]:
        problems.append("results.json was computed with different fitted parameters")
    if res.get("pass_fingerprint") != p.fingerprint():
        problems.append("results.json was computed from a different pass directory")
    if int(res.get("n", -1)) != len(y):
        problems.append(f"results.json covers {res.get('n')} images, the pass {len(y)}")
    elif res.get("true_grade_distribution") != np.bincount(y, minlength=5).tolist():
        problems.append("the labels differ from the ones the unblinding joined")
    locked = bool(p.run.get("locked"))
    if locked and not res.get("unblinded"):
        problems.append("the pass is locked and results.json is not the unblinding: "
                        "Phase 8 never reads a locked label first")
    return problems


# ------------------------------------------------------------------ P8.2-P8.4


def deferral_probability(rank: np.ndarray, coverage: float) -> np.ndarray:
    return 1.0 - S.acceptance_probability(rank, S.k_for(coverage, len(rank)))


def overlap(err: np.ndarray, q_conf: np.ndarray, q_dis: np.ndarray) -> dict:
    """Where the errors go: deferred by confidence only, disagreement only, both, neither.
    Expected shares, tie-breaks independent between the two arms."""
    err = np.asarray(err, dtype=np.float64)
    total = err.sum()
    if total == 0:                      # every key present, so seeds summarise alike
        nan = float("nan")
        return {"errors": 0.0, "confidence_only": nan, "disagreement_only": nan, "both": nan,
                "neither": nan, "error_rate_deferred_confidence": nan,
                "error_rate_deferred_disagreement": nan}
    return {"errors": float(total),
            "confidence_only": float((err * q_conf * (1 - q_dis)).sum() / total),
            "disagreement_only": float((err * (1 - q_conf) * q_dis).sum() / total),
            "both": float((err * q_conf * q_dis).sum() / total),
            "neither": float((err * (1 - q_conf) * (1 - q_dis)).sum() / total),
            "error_rate_deferred_confidence": float((err * q_conf).sum() / max(q_conf.sum(), 1e-12)),
            "error_rate_deferred_disagreement": float((err * q_dis).sum() / max(q_dis.sum(), 1e-12))}


def quintiles(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(edges, values, side="right")


def stratified(sig: dict, correct: np.ndarray) -> dict:
    """P8.3: disagreement's information at fixed confidence."""
    q = quintiles(sig["d_conf"])
    err = correct == 0
    table, weighted, weight = {}, 0.0, 0
    for k in range(5):
        in_q = q == k
        row = {}
        for level in range(3):
            cell = in_q & (sig["d_ev"] == level)
            row[str(level)] = {"n": int(cell.sum()),
                               "correct": int(correct[cell].sum())}
        a = auroc(sig["dis"][in_q], err[in_q])
        row["auroc_disagreement"] = a
        if np.isfinite(a):
            weighted += a * in_q.sum()
            weight += int(in_q.sum())
        table[str(k)] = row
    return {"by_quintile": table,
            "auroc_within_quintiles": float(weighted / weight) if weight else float("nan")}


def taxonomy(sig: dict, y: np.ndarray) -> dict:
    """P8.4: the errors disagreement cannot see, and the correct calls it contradicts."""
    yhat, e = sig["yhat"], sig["e"]
    wrong, right = yhat != y, yhat == y
    speak = yhat <= G.MAX_EXCLUDABLE_GRADE
    n_wrong = max(int(wrong.sum()), 1)
    silent = {"errors": int(wrong.sum()),
              "m3_abstains_yhat_ge3": float((wrong & ~speak).sum() / n_wrong),
              "m3_agrees_with_wrong_grade": float((wrong & speak & (e == yhat)).sum() / n_wrong),
              "visible": float((wrong & speak & (e != yhat)).sum() / n_wrong)}
    right_speak = right & speak
    n_rs = max(int(right_speak.sum()), 1)
    over = right_speak & (e > yhat)
    under = right_speak & (e < yhat)
    present = sig["present"]
    exudate = present["hard_exudate"] | present["soft_exudate"]
    n_over = max(int(over.sum()), 1)
    contradicted = {
        "correct_calls_m3_can_judge": int(right_speak.sum()),
        "contradicted": float((right_speak & (e != yhat)).sum() / n_rs),
        "m3_over_calls": float(over.sum() / n_rs),
        "m3_under_calls": float(under.sum() / n_rs),
        "over_calls_by_rule": {r: float((over & (sig["rule"] == r)).sum() / n_over)
                               for r in ("R2", "R3", "R3*")},
        "over_calls_with": {"microaneurysm": float((over & present["microaneurysm"]).sum() / n_over),
                            "haemorrhage": float((over & present["haemorrhage"]).sum() / n_over),
                            "exudate": float((over & exudate).sum() / n_over)},
        "over_calls_by_yhat": {str(g): float((over & (yhat == g)).sum() / n_over) for g in range(3)},
        "under_calls_by_yhat_e": {f"{g}->{v}": int((under & (yhat == g) & (e == v)).sum())
                                  for g in (1, 2) for v in range(g)},
    }
    return {"silent_errors": silent, "contradicted_correct": contradicted}


def model_tables(sig: dict, y: np.ndarray) -> dict:
    correct = (sig["yhat"] == y).astype(np.float64)
    err = 1.0 - correct
    rdr_err = ((sig["yhat"] >= 2) != (y >= 2)).astype(np.float64)
    ov = {}
    for c in S.COVERAGE_POINTS:
        qc = deferral_probability(sig["ranks"]["confidence"], c)
        qd = deferral_probability(sig["ranks"]["disagreement"], c)
        ov[f"@{int(round(c * 100))}"] = {"exact_grade": overlap(err, qc, qd),
                                          "rdr": overlap(rdr_err, qc, qd)}
    detection = {"confidence": auroc(sig["d_conf"], err == 1),
                 "ood": auroc(sig["z_ood"], err == 1),
                 "disagreement": auroc(sig["dis"], err == 1),
                 "d_evidence": auroc(sig["d_ev"], err == 1),
                 "combined": auroc(sig["ranks"]["combined"], err == 1)}
    return {"accuracy": float(correct.mean()), "overlap": ov, "error_detection_auroc": detection,
            "stratified": stratified(sig, correct), "taxonomy": taxonomy(sig, y)}


def summarise(per_model: Dict[str, dict], names: List[str]) -> dict:
    """Mean and range over `names` for every scalar leaf; counts summed where asked."""
    def walk(key_path):
        vals = []
        for n in names:
            node = per_model[n]
            for k in key_path:
                node = node[k]
            vals.append(node)
        return vals

    def rec(node, path):
        if isinstance(node, dict):
            return {k: rec(v, path + [k]) for k, v in node.items()}
        if isinstance(node, (int, float, np.integer, np.floating)) and not isinstance(node, bool):
            return across(walk(path))
        return node
    return rec(per_model[names[0]], [])


def candidates(sig: dict, y: np.ndarray, images: pd.DataFrame) -> tuple:
    """P8.5, on FIGURE_MODEL under the amended evidence. Deterministic order.

    Returns the candidate rows and, per category, how many images qualify -- zero is
    a finding to report, not an error."""
    yhat, d_conf, e = sig["yhat"], sig["d_conf"], sig["e"]
    zero = yhat == 0
    cut = np.quantile(d_conf[zero], CONFIDENT_QUANTILE) if zero.any() else -np.inf
    confident_zero = zero & (d_conf <= cut)
    masks = {
        "A_figure1_confident_normal_MA_found_truly_diseased":
            confident_zero & (y >= 1) & sig["present"]["microaneurysm"],
        "B_confident_normal_false_evidence": confident_zero & (y == 0) & (e == 2),
        "C_correct_grade2_no_evidence": (yhat == 2) & (y == 2) & (e == 0),
    }
    frames = []
    for category, mask in masks.items():
        idx = np.flatnonzero(mask)
        order = sorted(idx, key=lambda i: (d_conf[i], images["image_id"].iat[i]))
        chosen = order[:CANDIDATES_PER_CATEGORY]
        f = images.iloc[chosen][["image_id", "dataset", "image_path"]
                                + [f"n_{t}" for t in LESION_NAMES]
                                + [f"area_{t}" for t in LESION_NAMES]].copy()
        f.insert(0, "category", category)
        f["qualifying"] = int(mask.sum())
        f["true_grade"] = y[chosen]
        f["m1_grade"] = yhat[chosen]
        f["m1_confidence"] = 1.0 - d_conf[chosen]
        f["evidence_registered"] = images["evidence_grade"].to_numpy()[chosen]
        f["evidence_amended"] = e[chosen]
        f["rule_amended"] = sig["rule"][chosen]
        frames.append(f)
    return (pd.concat(frames, ignore_index=True),
            {category: int(mask.sum()) for category, mask in masks.items()})


def print_errors(name: str, rehearsal: bool, summary: dict,
                 cands: Optional[pd.DataFrame], qualifying: Optional[dict]) -> None:
    tag = "REHEARSAL -- not a result. " if rehearsal else ""
    print(f"\n{'#' * 78}\n{tag}{name} -- Phase 8, EXPLORATORY (mean [min-max] over the "
          f"{PRIMARY_VARIANT} seeds)\n{'#' * 78}")
    for analysis, s in summary.items():
        print(f"\n--- {analysis.upper()} ---")
        for cov in ("@80", "@90"):
            for kind in ("exact_grade", "rdr"):
                o = s["overlap"][cov][kind]
                print(f"  G1 {cov} {kind:<11} errors {fmt(o['errors'], 0)}: confidence only "
                      f"{fmt(o['confidence_only'])}, disagreement only {fmt(o['disagreement_only'])}, "
                      f"both {fmt(o['both'])}, neither {fmt(o['neither'])}; error rate deferred: "
                      f"conf {fmt(o['error_rate_deferred_confidence'])}, dis "
                      f"{fmt(o['error_rate_deferred_disagreement'])}")
        d = s["error_detection_auroc"]
        print("  G1 error-detection AUROC: " + ", ".join(f"{k} {fmt(v)}" for k, v in d.items()))
        st = s["stratified"]
        print(f"  G1 disagreement's AUROC for error within confidence quintiles "
              f"{fmt(st['auroc_within_quintiles'])} (0.5 = nothing beyond confidence)")
        for k in range(5):
            row = st["by_quintile"][str(k)]
            cells = []
            for level in ("0", "1", "2"):
                n, c = row[level]["n"]["mean"], row[level]["correct"]["mean"]
                cells.append(f"d_ev={level}: n {n:.0f} acc {c / n:.2f}" if n else f"d_ev={level}: n 0")
            print(f"    confidence quintile {k + 1} (1 = most confident): " + "; ".join(cells)
                  + f"; AUROC {fmt(row['auroc_disagreement'])}")
        t = s["taxonomy"]
        se = t["silent_errors"]
        print(f"  G3 M1's errors disagreement cannot see: M3 abstains (y-hat >= 3) "
              f"{fmt(se['m3_abstains_yhat_ge3'])}, M3 agrees with the wrong grade "
              f"{fmt(se['m3_agrees_with_wrong_grade'])}; visible {fmt(se['visible'])}")
        cc = t["contradicted_correct"]
        print(f"  G3 correct calls M3 contradicts: {fmt(cc['contradicted'])} of those it can judge "
              f"(over-calls {fmt(cc['m3_over_calls'])}, under-calls {fmt(cc['m3_under_calls'])})")
        print("     over-calls by rule: " + ", ".join(f"{k} {fmt(v)}" for k, v in
                                                     cc["over_calls_by_rule"].items())
              + "; with " + ", ".join(f"{k} {fmt(v)}" for k, v in cc["over_calls_with"].items()))
        print("     under-calls (y-hat->e): " + ", ".join(f"{k} {fmt(v, 0)}" for k, v in
                                                        cc["under_calls_by_yhat_e"].items()))
    if qualifying is not None:
        print(f"\nP8.5 figure candidates ({FIGURE_MODEL}, amended evidence):")
        for category, count in qualifying.items():
            group = cands[cands["category"] == category]
            print(f"  {category}: {count} qualify" + (f"; first {len(group)}:" if len(group) else ""))
            for r in group.itertuples():
                print(f"    {r.image_id:<34} truth {r.true_grade}  M1 {r.m1_grade} "
                      f"(conf {r.m1_confidence:.3f})  M3 {r.evidence_registered} -> "
                      f"{r.evidence_amended} {r.rule_amended}  MA {r.n_microaneurysm} "
                      f"({r.area_microaneurysm} px)")


def run_errors(args) -> int:
    params = load_params(args.fitted)
    p = load_pass(args.pass_dir)
    res = json.loads(Path(args.results).read_text())
    y = labels_for(p.ids, args.labels, split=args.split)
    problems = check_results(res, params, p, y)
    if problems:
        print("REFUSED: " + "; ".join(problems) + ".", file=sys.stderr)
        return EXIT_REFUSED

    frozen_e, _ = OP.evidence(p.images, OP.FROZEN_AREA_MIN)
    if not np.array_equal(frozen_e, p.images["evidence_grade"].to_numpy().astype(int)):
        print("REFUSED: the stored lesion counts do not reproduce M3's stored grade.",
              file=sys.stderr)
        return EXIT_REFUSED

    gaussians = {name: O.load_gaussian(args.ood_dir / params["models"][name]["ood"]["file"],
                                       params["models"][name]["ood"]["digest"])
                 for name in p.models}
    variants = models_by_variant(p.models)
    summary, per_all, cands, qualifying, mismatches = {}, {}, None, None, []
    for analysis in ANALYSES:
        if analysis == "amended" and "amended" not in res:
            continue
        published = analysis_block(res, analysis)["models"]
        per_model = {}
        for name in p.models:
            sig = signals(name, p.model_rows(name), p.embeddings(name), p.images, params,
                          gaussians[name], analysis == "amended")
            correct = (sig["yhat"] == y).astype(np.float64)
            for arm in ARMS:
                mine, theirs = S.auc(sig["ranks"][arm], correct), published[name]["arms"][arm]["auc"]
                if abs(mine - theirs) > REPRODUCE_TOL:
                    mismatches.append(f"{analysis}/{name}/{arm}: {mine!r} vs {theirs!r}")
            per_model[name] = model_tables(sig, y)
            if analysis == "amended" and name == FIGURE_MODEL:
                cands, qualifying = candidates(sig, y, p.images)
        per_all[analysis] = per_model
        summary[analysis] = summarise(per_model, variants[PRIMARY_VARIANT])
    if mismatches:
        print("REFUSED (P8.0): the recomputed signals do not reproduce the published AUCs:\n  "
              + "\n  ".join(mismatches[:10]), file=sys.stderr)
        return EXIT_REFUSED
    print(f"P8.0 reproduction gate passed: {len(p.models)} models x {len(per_all)} analyses x "
          f"{len(ARMS)} arms match results.json to {REPRODUCE_TOL:g}.")

    print_errors(res["name"], bool(res.get("rehearsal")), summary, cands, qualifying)
    args.out.mkdir(parents=True, exist_ok=True)
    out = {"name": res["name"], "rehearsal": bool(res.get("rehearsal")), "exploratory": True,
           "params_digest": params["digest"], "results_created_at": res.get("created_at"),
           "summary_primary": summary, "per_model": per_all,
           "variants": variants, "figure_model": FIGURE_MODEL,
           "figure_candidates_qualifying": qualifying}
    (args.out / "phase8.json").write_text(json.dumps(to_plain(out), indent=1))
    if cands is not None:
        cands.to_csv(args.out / "candidates.csv", index=False)
    print(f"\nwrote {args.out / 'phase8.json'}" + (" and candidates.csv" if cands is not None else ""))
    return 0


# ------------------------------------------------------------------ P8.5 figures


#: Outline colours for M2's four channels, bright against a red-orange fundus.
#: Identity never rests on colour alone: every panel carries the legend in words.
LESION_COLOURS = {"microaneurysm": (0, 229, 255), "haemorrhage": (118, 255, 3),
                  "hard_exudate": (255, 64, 255), "soft_exudate": (255, 255, 255)}
LESION_SHORT = {"microaneurysm": "MA", "haemorrhage": "HE", "hard_exudate": "hard EX",
                "soft_exudate": "soft EX"}


def lesion_masks(probs: np.ndarray, threshold: float, min_px: int) -> Dict[str, np.ndarray]:
    """Per channel: the components M3 counts (probs >= threshold, 8-connected, >= min_px)."""
    import cv2

    out = {}
    for name, channel in zip(LESION_NAMES, probs >= threshold):
        binary = channel.astype(np.uint8)
        mask = np.zeros(binary.shape, dtype=bool)
        if binary.any():
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            keep = np.zeros(n, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
            mask = keep[labels]
        out[name] = mask
    return out


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1: fixed bitmap font
        return ImageFont.load_default()


def render_panel(rgb: np.ndarray, masks: Dict[str, np.ndarray], caption: List[str]):
    """The image, and beside it the same image with each lesion type outlined."""
    import cv2
    from PIL import Image, ImageDraw

    h, w = rgb.shape[:2]
    outlined = rgb.copy()
    for name, mask in masks.items():
        if mask.any():
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(outlined, contours, -1, LESION_COLOURS[name], 1)
    line_h, pad = 18, 6
    canvas = Image.new("RGB", (2 * w + pad, h + pad + line_h * len(caption)), (255, 255, 255))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    canvas.paste(Image.fromarray(outlined), (w + pad, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    for i, text in enumerate(caption):
        draw.text((4, h + pad + i * line_h), text, fill=(20, 24, 31), font=font)
    return canvas


def legend_strip(width: int):
    from PIL import Image, ImageDraw

    strip = Image.new("RGB", (width, 26), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    font = _font(14)
    x = 6
    draw.text((x, 5), "M2's lesion outlines (right of each pair):", fill=(20, 24, 31), font=font)
    x += 300
    for name in LESION_NAMES:
        draw.rectangle([x, 8, x + 22, 18], outline=LESION_COLOURS[name], width=3)
        draw.text((x + 28, 5), LESION_SHORT[name], fill=(20, 24, 31), font=font)
        x += 110
    return strip


def grid(panels: list, columns: int = 2):
    from PIL import Image

    if not panels:
        return None
    w, h = panels[0].size
    rows = (len(panels) + columns - 1) // columns
    legend = legend_strip(columns * w)
    sheet = Image.new("RGB", (columns * w, legend.size[1] + rows * h), (255, 255, 255))
    sheet.paste(legend, (0, 0))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % columns) * w, legend.size[1] + (i // columns) * h))
    return sheet


def run_figures(args, segmenter=None, size: Optional[int] = None) -> int:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import predict as P                                       # the pass's own loaders
    from verify_dr.data.dataset import repath_to_cache
    from verify_dr.reasoning.facts import MIN_LESION_PX

    frame = pd.concat([pd.read_csv(c) for c in args.candidates], ignore_index=True)
    if frame.empty:              # no image qualified: a finding, already counted by `errors`
        print("no figure candidates qualified in " + ", ".join(str(c) for c in args.candidates))
        return 0
    if args.cache_root:
        frame = repath_to_cache(frame, [str(r) for r in args.cache_root])
    device = torch.device("cpu")
    if segmenter is None:
        segmenter, size = P.load_segmenter(args.evidence_checkpoint, device)
    size = size or 512
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for category, group in frame.groupby("category", sort=False):
        panels = []
        for r in group.head(args.per_category).itertuples():
            rgb = P.load_rgb(r.image_path, size)
            with torch.no_grad():
                x = P.normalise(P.to_chw(rgb)[None].to(device))
                probs = torch.sigmoid(segmenter(x).float())[0].cpu().numpy()
            masks = lesion_masks(probs, P.EVIDENCE_THRESHOLD, MIN_LESION_PX)
            counts = ", ".join(f"{LESION_SHORT[t]} {int(getattr(r, f'n_{t}'))} "
                               f"({int(getattr(r, f'area_{t}'))} px)" for t in LESION_NAMES
                               if int(getattr(r, f"n_{t}")) > 0) or "no lesion counted"
            caption = [f"{r.image_id}",
                       f"truth {r.true_grade} | M1 {r.m1_grade} (confidence {r.m1_confidence:.3f})"
                       f" | M3 {r.evidence_registered} registered, {r.evidence_amended} "
                       f"amended ({r.rule_amended})",
                       counts]
            panel = render_panel(rgb, masks, caption)
            stem = str(r.image_id).replace("::", "_")
            panel.save(args.out / f"{category}__{stem}.png")
            panels.append(panel)
        sheet = grid(panels)
        if sheet is not None:
            path = args.out / f"{category}.png"
            sheet.save(path)
            written.append(path)
            print(f"{category}: {len(panels)} panels -> {path}")
    return 0 if written else 1


# ------------------------------------------------------------------ P8.6 audit


def image_stats(rgb: np.ndarray) -> Dict[str, float]:
    """Label-free statistics that tell preprocessing regimes apart."""
    import cv2

    fov = F.field_of_view(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    inside = fov if fov.any() else np.ones(fov.shape, dtype=bool)
    blur = cv2.GaussianBlur(gray, (0, 0), 10)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    near_gray = np.all(np.abs(rgb.astype(np.int16) - 128) <= 16, axis=2)
    corners = np.concatenate([rgb[:16, :16].reshape(-1, 3), rgb[:16, -16:].reshape(-1, 3),
                              rgb[-16:, :16].reshape(-1, 3), rgb[-16:, -16:].reshape(-1, 3)])
    return {"fov_fraction": float(fov.mean()),
            "background_mean": float(rgb[~fov].mean()) if (~fov).any() else float("nan"),
            "corner_mean": float(corners.mean()),
            "red_mean": float(rgb[..., 0][inside].mean()),
            "green_mean": float(rgb[..., 1][inside].mean()),
            "blue_mean": float(rgb[..., 2][inside].mean()),
            "gray_sd": float(gray[inside].std()),
            "saturation_mean": float(hsv[..., 1][inside].mean() / 255.0),
            "near_gray_share": float(near_gray[inside].mean()),
            "laplacian_var": float(lap[inside].var()),
            "highpass_energy": float(np.abs(gray - blur)[inside].mean())}


def summarise_stats(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    frame = pd.DataFrame(rows)
    return {c: {"median": float(frame[c].median()), "q25": float(frame[c].quantile(0.25)),
                "q75": float(frame[c].quantile(0.75))} for c in frame.columns}


AUDIT_SOURCES = (("EyePACS test", "eyepacs_full.csv", "EyePACS", "test"),
                 ("APTOS", "aptos_external.csv", None, None),
                 ("Messidor-2", "messidor2_external.csv", None, None),
                 ("DDR", "eyepacs_ddr_full.csv", "DDR", None))
RAW_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def audit_frames(manifest_dir: Path, cache_roots, per_source: int, seed: int) -> Dict[str, pd.DataFrame]:
    """A seeded sample of each source's cached images. Only paths are read, never a grade."""
    from verify_dr.data.dataset import repath_to_cache

    out = {}
    for label, file, dataset, split in AUDIT_SOURCES:
        path = Path(manifest_dir) / file
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in {"image_path", "dataset", "split"})
        if dataset is not None:
            frame = frame[frame["dataset"].astype(str) == dataset]
        if split is not None and "split" in frame.columns:
            frame = frame[frame["split"].astype(str) == split]
        if len(frame) == 0:
            continue
        frame = frame.sample(n=min(per_source, len(frame)), random_state=seed)
        if cache_roots:
            frame = repath_to_cache(frame, [str(r) for r in cache_roots])
        out[label] = frame.reset_index(drop=True)
    return out


def detection_rates(locked: Path) -> Dict[str, dict]:
    """Per locked set: how often M2 reports each lesion type, frozen rule (label-free)."""
    out = {}
    for images_csv in sorted(Path(locked).rglob("images.csv")):
        images = pd.read_csv(images_csv)
        if not {"n_microaneurysm", "area_microaneurysm"} <= set(images.columns):
            continue
        present = OP.presence(images, OP.FROZEN_AREA_MIN)
        out[images_csv.parent.name] = {
            "n": int(len(images)),
            **{f"{t}_present": float(present[t].mean()) for t in LESION_NAMES},
            **{f"{t}_median_area_when_present":
               float(np.median(images[f"area_{t}"].to_numpy()[present[t]])) if present[t].any()
               else float("nan") for t in LESION_NAMES}}
    return out


def contact_sheet(frames: Dict[str, pd.DataFrame], per_row: int, thumb: int):
    from PIL import Image, ImageDraw

    rows = list(frames.items())
    label_w = 150
    sheet = Image.new("RGB", (label_w + per_row * thumb, len(rows) * thumb), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = _font(16)
    for i, (label, frame) in enumerate(rows):
        draw.text((8, i * thumb + thumb // 2 - 8), label, fill=(20, 24, 31), font=font)
        for j, path in enumerate(frame["image_path"].head(per_row)):
            try:
                with Image.open(path) as img:
                    sheet.paste(img.convert("RGB").resize((thumb, thumb)),
                                (label_w + j * thumb, i * thumb))
            except OSError:
                continue
    return sheet


def run_audit(args) -> int:
    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=True)
    frames = audit_frames(args.manifest_dir, args.cache_root, args.per_source, args.seed)
    result = {"exploratory": True, "label_free": True, "per_source": args.per_source,
              "seed": args.seed, "cached": {}, "raw_messidor2": [], "detection": {}}
    print("P8.6 cached images (median [q25-q75]), after this project's preprocessing:")
    keys = None
    for label, frame in frames.items():
        stats = []
        for path in frame["image_path"]:
            try:
                with Image.open(path) as img:
                    stats.append(image_stats(np.array(img.convert("RGB"))))
            except OSError:
                continue
        if not stats:
            continue
        summary = summarise_stats(stats)
        result["cached"][label] = {"n": len(stats), **summary}
        keys = keys or list(summary)
        print(f"  {label:<13} n={len(stats)}")
        for k in keys:
            s = summary[k]
            print(f"    {k:<18} {s['median']:.3f} [{s['q25']:.3f}-{s['q75']:.3f}]")

    raw_files: List[Path] = []
    for root in args.raw_root or []:
        raw_files += sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in RAW_SUFFIXES)
    if raw_files:
        rng = np.random.default_rng(args.seed)
        pick = [raw_files[i] for i in sorted(rng.choice(len(raw_files),
                                                        size=min(args.raw_sample, len(raw_files)),
                                                        replace=False))]
        print(f"\nP8.6 raw Messidor-2 files ({len(raw_files)} found; {len(pick)} sampled):")
        thumbs = []
        for path in pick:
            with Image.open(path) as img:
                mode, (w, h) = img.mode, img.size
                rgb = np.array(img.convert("RGB").resize((512, int(round(512 * h / w)))))
            s = image_stats(rgb)
            row = {"file": path.name, "width": w, "height": h, "mode": mode,
                   "bytes": path.stat().st_size, **s}
            result["raw_messidor2"].append(row)
            thumbs.append(Image.fromarray(rgb))
            print(f"  {path.name:<28} {w}x{h} {mode:<4} {path.stat().st_size / 1e6:5.2f} MB  "
                  f"corner {s['corner_mean']:6.1f}  near-gray {s['near_gray_share']:.3f}  "
                  f"saturation {s['saturation_mean']:.3f}  high-pass {s['highpass_energy']:.2f}")
        if thumbs:
            tw = 256
            strip = Image.new("RGB", (tw * len(thumbs), tw), (255, 255, 255))
            for i, t in enumerate(thumbs):
                t.thumbnail((tw, tw))
                strip.paste(t, (i * tw, 0))
            strip.save(args.out / "raw_messidor2.png")

    if args.locked:
        result["detection"] = detection_rates(args.locked)
        print("\nP8.6 how often M2 reports each lesion type (frozen rule; label-free):")
        for name, d in result["detection"].items():
            print(f"  {name:<14} n={d['n']:<6} " + "  ".join(
                f"{LESION_SHORT[t]} {d[f'{t}_present']:.1%} (median {d[f'{t}_median_area_when_present']:.0f} px)"
                for t in LESION_NAMES))

    if frames:
        contact_sheet(frames, args.sheet_per_row, 160).save(args.out / "contact_sheet.png")
    (args.out / "audit.json").write_text(json.dumps(to_plain(result), indent=1))
    print(f"\nwrote {args.out / 'audit.json'}")
    return 0


# ------------------------------------------------------------------ main


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("secondary", help="P8.1: transcribe the secondary outcomes.")
    s.add_argument("--results", required=True, type=Path, nargs="+")
    s.add_argument("--out", type=Path, default=None)

    e = sub.add_parser("errors", help="P8.0, P8.2-P8.5 for one set.")
    e.add_argument("--pass-dir", required=True, type=Path)
    e.add_argument("--labels", required=True, type=Path)
    e.add_argument("--split", default=None)
    e.add_argument("--fitted", required=True, type=Path)
    e.add_argument("--ood-dir", required=True, type=Path)
    e.add_argument("--results", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)

    f = sub.add_parser("figures", help="P8.5: draw the candidates with M2's outlines.")
    f.add_argument("--candidates", required=True, type=Path, nargs="+")
    f.add_argument("--cache-root", type=Path, nargs="*", default=None)
    f.add_argument("--evidence-checkpoint", required=True, type=Path)
    f.add_argument("--per-category", type=int, default=CANDIDATES_PER_CATEGORY)
    f.add_argument("--out", required=True, type=Path)

    a = sub.add_parser("audit", help="P8.6: Messidor-2's preprocessing, label-free.")
    a.add_argument("--manifest-dir", required=True, type=Path)
    a.add_argument("--cache-root", type=Path, nargs="*", default=None)
    a.add_argument("--locked", type=Path, default=None)
    a.add_argument("--raw-root", type=Path, nargs="*", default=None)
    a.add_argument("--per-source", type=int, default=200)
    a.add_argument("--raw-sample", type=int, default=12)
    a.add_argument("--sheet-per-row", type=int, default=12)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True, type=Path)

    args = ap.parse_args(argv)
    if args.command == "secondary":
        return run_secondary(args)
    if args.command == "errors":
        return run_errors(args)
    if args.command == "figures":
        return run_figures(args)
    return run_audit(args)


if __name__ == "__main__":
    sys.exit(main())
