# VERIFY-DR — working notes

MSc thesis project. Read `docs/00_START_HERE.md` before doing anything substantive.

## What this project is

An image-level DR grader plus an independently-supervised lesion-evidence pathway,
where *disagreement between them* — not the grader's confidence — decides which
automated gradings to trust.

The contribution is the referral decision, **not** grading accuracy. Never optimise
for QWK at the expense of the disagreement signal.

## Non-negotiable rules

1. **APTOS and Messidor-2 are locked.** Never used for training, validation, early
   stopping, calibration fitting, hyperparameter choice, or model selection. Evaluated
   once, after `preregistration/PREREGISTRATION.md` is committed. The only exception is
   the Saerens–Decock EM step reading *unlabelled* target images.
2. **Patient-grouped splits, always.** EyePACS patient IDs come from
   `^(\d+)_(left|right)$`. The split builder asserts disjointness — never disable it.
3. **Test sets keep natural prevalence.** Balanced variants are a *training* ablation.
4. **M1 and M2 never share weights.** Their independence is the thesis. A shared
   encoder would make them fail together and destroy the signal.
5. **After the Phase 5 freeze, changes are recorded as deviations**, not made silently.
6. **Never compare custom-split numbers to the Kaggle leaderboard.** H1 on the official
   split is the only comparable figure.

## Conventions

- Python, PyTorch. Package under `src/verify_dr/`.
- Configs in `configs/`; one YAML per stage, overriding `base.yaml`.
- Results to `results/<stage>/<experiment_id>/`, written immediately, not at the end.
- Checkpoint every epoch — Kaggle sessions die at ~12 h. Resume, never restart.
- Update `docs/04_experiment_register.md` Status and Result as experiments complete.

## Key documents

| Need | File |
|---|---|
| What to do next | `docs/00_START_HERE.md` |
| Architecture, shapes, losses | `docs/03_model_architecture.md` |
| Protocol rules and hypotheses | `docs/02_research_protocol.md` |
| Experiment log | `docs/04_experiment_register.md` |
| Dataset slugs and verification | `docs/05_dataset_card.md` |
| Script contracts | `scripts/README.md` |

## Status

Phases 1-2 have been run on Kaggle. `verify-dr-cache-512` is published, its IDRiD
masks topped up via `01b_idrid_masks.ipynb`, and the Phase 2 manifests are built.

Phase 3 code is complete and tested: `src/verify_dr/models/{grading,losses}.py`,
`src/verify_dr/data/dataset.py`, `src/verify_dr/evaluation/metrics.py`,
`scripts/train_grading.py`, `notebooks/03_grading_sweeps.ipynb`. Verified end to end
on a synthetic fixture (val QWK 0 -> 1.0, all five per-class recalls 1.00), plus
resume, the architecture guard and cache repathing. `python -m unittest discover -s
tests` covers the load-bearing properties.

**B1 DONE — 512 px.** Grade-1 F1 across 384/512/768 is 0.151 / 0.142 / 0.161:
non-monotone, spread 0.019 on one seed each. Noise, not a trend. 768 also *upsamples*
the 512 px cache. 512 is native, fastest, and keeps Phase 6 at ~21 GPU-h.

**B2 DONE — EfficientNet-B0.** QWK tied with ResNet50 (0.679 vs 0.676) at 0.70× the
cost, and better MAE. ResNet50 does reach grade-1 F1 0.180 vs 0.142 — twice B1's whole
noise band — but via a more liberal grade-1 operating point, not better grading. The
loss and sampler control that, so **B3/B4 carry a recorded prediction**: if B0 reaches
grade-1 F1 ~0.18 there, B2 is settled; if nothing moves grade 1, reopen B2 as a
deviation.

**B3 DONE — focal-ordinal.** First Stage B lever that matters: QWK spans 0.053 across
heads, nearly 2x B1's whole resolution spread. `softmax_ce` ruled out (QWK 0.053 below
`ordinal`, 1.9x the noise floor, and it discards the CORAL structure Phase 7 needs).
`ordinal` vs `ordinal_focal` isolates gamma exactly: focal costs 0.021 QWK (0.8x floor,
inside noise) and buys +83% grade-1 recall and +0.061 macro-F1 (5.6x and 4.2x their
floors). **Focal weighting demonstrably does its stated job — a reportable finding.**

**Noise floors are per-metric**, from B1's observed spread: QWK 0.0276, macro-F1 0.0144,
grade-1 F1 0.0190, grade-1 recall 0.0220, MAE 0.0230. The earlier "+-0.02 QWK" was a
conversational round number; use these.

**B2's prediction is still open** — no head reached ResNet50's grade-1 F1 of 0.180.
B4's `class_balanced` is the remaining test.

Next: **B4** (sampler), 2 runs at 512 / B0 / `ordinal_focal` — `natural` and
`class_balanced`; `stratified_exposure` is the B1 512 run.

### A0 — CLOSED: pass, with two recorded limitations

**Crops.** The EyePACS fallback rate of 0.99919 against a 0.005 gate was benign — the
mirror ships pre-cropped images, so `retinal_bbox` correctly reports "nothing to crop".
Visual audit over **260 crops** (EyePACS 100, APTOS 60, DDR 60, IDRiD 40): retinal
width/frame median **1.000**, zero tiles below 0.75, lit fractions 0.70-0.79 against
pi/4 = 0.785 for an inscribed disc. The 0.005 threshold assumes raw uncropped originals
and is the wrong test for these mirrors.

**Counts.** APTOS, Messidor-2 and DDR reconcile **exactly, every grade** (DDR's 12 522
is the published 13 673 less 1 151 ungradable). EyePACS is **693 short (0.78%)** of the
official 88 702 — upstream of us, since `cache_report.json` shows found = cached =
88 009 with failed = 0. The loss is **non-random by grade** (chi-square 105.6, 4 df;
grade 4 lost at 3x grade 0's rate), most likely because mirrors drop unreadable files
and severe DR correlates with media opacity. Prevalence impact is negligible: rDR
19.34% -> 19.23%. **State it in the thesis limitations.**

**Structure.** 1.989 images/patient, split fractions exact, held-out rows identical
across variants, no balancing shortfall.

**Outstanding:** Messidor-2 visual audit (locked until unblinding; counts already exact).

### Phase 6 budget — corrected

Earlier projections assumed 57 656 training rows. The real figure is **59 842**, so at
46.0 img/s the frozen recipe needs **21.7 GPU-h**, not 20.9 — slightly over the ~20 h
budget. ResNet50 would be 31.1 h and fusion 47.3 h. Decisions unchanged; trim epochs or
seeds at the freeze and record it in the pre-registration.

### One thing to carry into Phase 3

B1 is decided on **grade-1 F1**, not QWK and not grade-1 recall. Both of the obvious
rules fail, and both failures are reproduced in `tests/test_grading.py`:

- A model that never predicts grade 1 still scores **~0.97 QWK** — grade 1 is one
  step from grade 0, and quadratic weights barely punish that error.
- A model that predicts grade 1 for *every* image scores **1.000 grade-1 recall**.

So check `distinct_predictions` first (1 means the run collapsed and decides
nothing), then read grade-1 F1. `train_grading.py` and the notebook both apply this.
