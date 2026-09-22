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
IDRiD needs one more top-up — `01c_idrid_grading.ipynb`, Part B — before C1 can run;
see Phase 4 below.

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

Stage B is closed: the designed default recipe survived every ablation.

## Phase 4 — Stage C (code complete, C1 unblocked)

`src/verify_dr/{models/evidence.py,models/seg_losses.py,data/segmentation.py,
data/geometry.py,evaluation/{geometry_metrics,segmentation_metrics}.py}`,
`scripts/{train_geometry,train_evidence}.py`, `notebooks/04_phase4.ipynb`.
56 tests pass, including the static AST test that M2 never imports `grading.py`
(rule 4). C2 verified end to end on a synthetic fixture: mean Dice 0.24 -> 0.68.

**C1 was blocked, and the cause was ours.** IDRiD Part A (`IDRiD_01`-`81`, masks) and
Part B (`IDRiD_001`-`516`, grades + Part C centres) are *different image sets*. The
cache held Part A only, and `02_manifests.ipynb` resolved `--coords-source-dir` by
looking for `"segmentation"` in the path — pinning it to Part A. Every coordinate row
missed, and the `--no-grades` fallback then wrote a valid manifest with an empty
geometry column, so nothing looked wrong. Fixed three ways: `01c_idrid_grading.ipynb`
caches Part B, Phase 2 resolves the directory by matching the tables' own IDs, and
`prepare_manifest.py` exits 1 rather than writing a manifest when `--coords` projects
nothing. Verified on a fixture: 0/50 rows before, 50/50 after, C1 then trains.

**C4 is not in the Phase 4 notebook** — it needs M3, which Phase 4 does not build.

**Read C2's Dice as `dice_present`**, over images where the lesion is annotated.
Averaging over every image folds in empty-target/empty-prediction pairs scoring 1.0 by
convention and inflates the headline without anything having been segmented.

### Stage C CLOSED — all four experiments have numbers

| | result | verdict |
|---|---|---|
| C1 | 0.686 DD vs a 0.5 gate | **FAIL** — M3 runs count-only |
| C2 | mean Dice 0.505, MA 0.344 (DDR only) | done |
| C3 | 0.425 on held-out IDRiD, -16% relative | **transfers** |
| C4 | QWK 0.375 vs M1's 0.679 | **informative but weaker** |

**C4 is the premise surviving its own falsification test.** An evidence path as good
as the grader makes the grader redundant; one at chance makes disagreement
meaningless. 0.375 against 0.679 is neither, so F2 in Phase 7 is worth running.

**C3 means the signal is not source-specific.** A 16% relative Dice drop on a model
that never saw IDRiD is ordinary transfer degradation, not the collapse the experiment
was designed to detect. First evidence for H1'.

**Two caveats that belong beside every C4 number.** Only 3 of 5 grades are reachable
(grade 3 needs R4 needs quadrants needs C1; grade 4 needs neovascularisation, which
nothing annotates), so the QWK is a floor. And R3* -- haemorrhage or exudate with no MA
detected -- fired on 35% of the corpus, which is M2a's MA channel missing what its
larger channels find.

**C1: two attempts, both failed.** The coordinate head gives 0.686 DD with 10/83
laterality flips carrying 47% of the error. The heatmap head built to fix that gave
1.028 DD and 27/83 flips -- worse, including on the fovea, where bimodality cannot
apply. That refutes the remedy, not the diagnosis; the cause of the flips is not
established. A weighted-loss retrain (1.3 min) is recorded as declined, deliberately.

Next: **Phase 5 freeze.** The pre-registration's M1 half is filled in; its OD/fovea row
resolves to the coordinate head, and the evidence-data row to DDR-only with IDRiD held
out. Administrative fields and supervisor acknowledgement are the author's.

## Phase 6a — final training, written to be interrupted

`notebooks/06_final_training.ipynb` runs the six final runs (2 variants x 3 seeds).
It does **not** unblind: the single external evaluation is a separate notebook, so
re-running the training one can never spend the one shot.

**epochs is 10, not 12** (D7). 12 was `train_grading.py`'s default leaking into
`frozen_config.yaml`; every Stage B arm that selected this recipe ran 10, so freezing
12 would freeze a configuration no experiment evaluated.

**~24 GPU-h, not 21.7.** The old figure priced both variants at `eyepacs_full`'s
59 842 rows; `eyepacs_ddr_full` carries all 12 522 DDR images in train as well. The
notebook reads each variant's real row count.

**The step that is expensive to skip:** `/kaggle/working` does not survive a session.
Publish results as `verify-dr-phase6` every session and attach it the next, or
section 6 finds nothing and finished runs are retrained -- silently, because a
retrained run writes a perfectly valid `metrics.json`.

The budget guard counts **elapsed session wall-clock**, which is what Kaggle bills,
and refuses to start a run that will not fit rather than have it killed mid-epoch.

**Two limits bind, not one.** The budget is `min(weekly quota, session wall)`. Once the
quota is healthy the *session* wall is the tighter: a session is killed at a fixed age
whatever the quota says, and one killed mid-run may never save its output. The DDR
variant is ~13.1 h across three seeds and therefore needs two sessions regardless of
quota.

**Finish a session with Quick Save, never "Save & Run All".** Save & Run All
re-executes the whole notebook in a fresh container and would retrain from scratch.

### Before the unblinding — two blockers found 2026-09-22

**The leaderboard benchmark cannot run as registered.** Variants were built with
`--eyepacs-split regroup` (all 88,009 images, official train and test mixed), yet
`frozen_config.yaml` says `split: official_eyepacs`. About two-thirds of the official
test images are in the six models' training data. Only that one secondary outcome is
lost; F2, H2, H3 and in-domain triage are unaffected. Decide A (drop the comparison,
0 GPU-h) or C (one official-split model, ~1.8 GPU-h/seed, needs `source_split`) and
record it as **D8 before any locked data is read**. `notebooks/snippets/
phase6_check.py` measures the overlap and says whether C is possible.

**Phase 7 code does not exist** (`calibration/`, `triage/` are empty). Build and test
it on internal data *first*; then one pass over the locked sets writes a label-free
prediction table, and labels are joined once. Unblinding first would mean writing the
analysis after seeing the answer.

**Section 1 once shipped as a comment.** The notebook was built by copying cells from
`04_phase4.ipynb` by index, and the clone step is cell 2 there, not cell 1. The
markdown heading landed in a code cell, where `## 1 - Clone the repo` is a comment: it
ran, succeeded and cloned nothing, and only surfaced six sections later when section 8
wanted `REPO_DIR`. The generator had checked that every cell parsed — and a comment
parses. `tests/test_notebook_integrity.py` now rejects inert code cells and
use-before-definition across cells, on every notebook, and asserts it catches this
exact bug.

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
