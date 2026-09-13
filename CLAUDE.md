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

Phases 1-2 complete in code, notebooks 00/01/01b/02 runnable. Phase 1 has been run
on Kaggle; `verify-dr-cache-512` is published but its IDRiD masks are missing (a
mask-naming bug, now fixed) and are being topped up via `01b_idrid_masks.ipynb`.
Phase 2 has not been run. Next: Phase 3 model code (M1). No experiments run.
