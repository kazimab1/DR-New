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

Next: **B3** (head) at 512 with B0. Two runs — `softmax_ce` and plain `ordinal`;
`ordinal_focal` is already the B1 512 run.

### One thing to carry into Phase 3

B1 is decided on **grade-1 F1**, not QWK and not grade-1 recall. Both of the obvious
rules fail, and both failures are reproduced in `tests/test_grading.py`:

- A model that never predicts grade 1 still scores **~0.97 QWK** — grade 1 is one
  step from grade 0, and quadratic weights barely punish that error.
- A model that predicts grade 1 for *every* image scores **1.000 grade-1 recall**.

So check `distinct_predictions` first (1 means the run collapsed and decides
nothing), then read grade-1 F1. `train_grading.py` and the notebook both apply this.
