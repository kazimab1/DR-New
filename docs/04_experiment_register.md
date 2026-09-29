# VERIFY-DR — Experiment Register

Your running log. Update `Status` and `Result` as you go; this file becomes the
skeleton of the results chapter.

**Status values:** `TODO` · `RUNNING` · `DONE` · `BLOCKED` · `DEVIATED`

---

## Stage A — Infrastructure (no claims)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| A0 | Did the 512 px cache preserve the data? | — | Counts reconcile per grade; crop failures < 0.5%; 100-crop visual audit | **PASS, with limitations** | 3 of 4 exact per grade; EyePACS 0.78% short upstream |
| A1 | Does the pipeline run end to end? | — | Smoke test completes; 1-epoch pilot gives non-trivial QWK | **DONE** | Smoke test passes; B1 512 reached val QWK 0.679 |

### A0 — crop **RESOLVED: benign**. The gate is the wrong test for this source.

`verify-dr-cache-512/cache512/eyepacs/cache_report.json`, read 14 Sep 2026:

| Field | Value |
|---|---|
| source_root | `/kaggle/input/datasets/tantai31124/eyepacs-original` |
| params | size 512, fit **pad**, quality 90, CLAHE on (clip 2.0, 8 tiles) |
| counts | found 88 009 · done 87 809 · skipped 200 · **failed 0** |
| runs / interrupted | 2 / false |
| crop detected | **71** |
| crop fell back to full frame | **87 938** of 88 009 measured |
| **crop fallback_rate** | **0.99919** — gate is **0.005** |
| output | 4287.7 MiB, mean 51 086 B/image |

The retinal-field crop reported success on 71 images out of 88 009. **On the gate as
written, A0 fails for EyePACS by a factor of 200.**

**Two readings, opposite consequences, and the number alone cannot separate them.**

1. *Benign.* The mirror ships images already cropped to the retinal field, so there is
   nothing left to crop and `retinal_bbox` says so correctly. `build_cache.py` documents
   exactly this: corner patches "in an already cropped one … are retina, which drives
   the threshold high, collapses the mask, and trips the area guard below — **the right
   answer there**". The fallback is also returned when the detected box is within 2% of
   the full frame, which is precisely what an already-cropped image produces. Under this
   reading the cache is sound and the 0.005 gate is simply the wrong test for this
   source.
2. *Serious.* The crop is failing on raw, uncropped originals. EyePACS originals are
   typically wide (≈3888×2592) with the fundus centred between black bars. With
   `fit: pad`, a failed crop pads that wide frame to square and resizes to 512 — leaving
   the retina occupying roughly two-thirds of the width, an **effective retinal
   resolution nearer 340 px than 512**. Every Stage B number would then have been
   measured on under-resolved images, and B1's "resolution does not matter" finding would
   have been measured over a range where the retina never filled the frame.

**The evidence leans to reading 1.** If the corners held black surround, the threshold
would sit at ~10–15/255, the fundus would light up cleanly, and the box would come back
far smaller than the frame — detection, not fallback. Getting the opposite on 99.9% of
images implies the corners are *not* dark, which is what a pre-cropped image looks like.
Mean output size of 51 KB per 512² JPEG at q90 also reads as mostly-retina rather than
a third flat black. **None of this is proof.** `contact_sheet.jpg` settles it in one
look, which is exactly why A0 requires a visual audit and not just a rate.

**Also unreconciled:** the mirror reports 88 009 images found, against the 88 702 of the
official EyePACS release — a shortfall of 693. A0 requires counts to reconcile per
grade; that has not been done, and this gap needs an explanation before the freeze.

### Resolution — 14 Sep 2026, from the contact sheets

**Reading 1 holds. The cache is sound and Stage B stands.**

`contact_sheet.jpg` was fetched and inspected for both EyePACS (100 crops) and APTOS
(60 crops). Every tile is a fundus disc filling its frame with only a thin margin; optic
disc and macula are visible in nearly all, vessels are sharp down to fine branches, and
there is not one wide black-barred frame or disc-adrift-in-black among 160 crops.

Measured off the sheets rather than judged by eye, retinal width as a fraction of frame
width:

| Dataset | median | mean | 10th pct | tiles below 0.75 | implied retinal diameter |
|---|---|---|---|---|---|
| EyePACS | **1.000** | 0.990 | 0.999 | **0 of 100** | **512 px** |
| APTOS | **1.000** | 1.000 | 1.000 | **0 of 60** | **512 px** |

Reading 2 predicted ≈0.67 width and ≈340 px. The lit fraction of the frame is 0.74
(EyePACS) and 0.70 (APTOS) against π/4 = 0.785 for a disc inscribed in a square — the
small shortfall is the flat top-and-bottom truncation typical of fundus cameras.
**Reading 2 is ruled out.** The retina fills the frame, so no Stage B result was measured
on under-resolved images and B1's resolution finding stands.

**What this says about the gate, for the write-up.** A crop-fallback rate near 1.0 is not
evidence of a broken crop when the source ships pre-cropped — `retinal_bbox` returns
`full, False` for a box within 2% of the frame, which is the correct answer there and is
what these mirrors produce. The 0.005 threshold in A0 was written assuming raw
uncropped originals and is simply the wrong test for `tantai31124/eyepacs-original` and
the APTOS mirror. `diagnose_cache.py` now says so rather than reporting a flat FAIL. The
rate is kept as recorded evidence, not as a pass/fail gate.

### Visual audit — 4 of 5 datasets, 260 crops, all pass

| Dataset | crops | retinal width / frame (median) | below 0.75 | lit fraction | verdict |
|---|---|---|---|---|---|
| EyePACS | 100 | 1.000 | 0 | 0.741 | pass |
| APTOS | 60 | 1.000 | 0 | 0.702 | pass |
| DDR | 60 | 1.000 | 0 | 0.788 | pass |
| IDRiD | 40 | 1.000 | 0 | 0.730 | pass |
| Messidor-2 | — | — | — | — | **not yet audited** |

Lit fraction against π/4 = 0.785 for a disc inscribed in a square: DDR lands on it
almost exactly, and the others fall slightly under because of the flat top-and-bottom
truncation fundus cameras produce. Optic disc and macula are visible in nearly every
crop, vessels resolve to fine branches, and lesions are legible — hard exudates as
bright yellow clusters, haemorrhages as dark blots — which is what matters for the
Phase 4 masks. IDRiD is notably uniform, as expected from a single-camera single-site
collection.

**One anomaly, recorded and not worth fixing.** DDR row 6 column 8 has dark corners
(median 4) but pure white sides (median 255): that source image carries a white
background, so `retinal_bbox` reads the surround as lit and returns the full frame,
then `fit_square` pads top and bottom with black. One image in 260. It also marks a
limit of the width measure used above — it measures the *lit* region, which equals the
retina only when the surround is dark, so on that one tile it over-reports. Every other
tile in all four sheets has a dark surround (corner median ≤ 20), where the measure is
exact.

### Count reconciliation — **3 of 4 datasets exact to the image**

From `dataset_plan.json` (seed 42, generated 13 Sep 2026), against published figures:

| Dataset | Published | In cache | Diff | Verdict |
|---|---|---|---|---|
| APTOS 2019 train | 3 662 | 3 662 | 0 | **exact, every grade** |
| Messidor-2 adjudicated | 1 744 | 1 744 | 0 | **exact, every grade** |
| DDR minus ungradable | 12 522 | 12 522 | 0 | **exact, every grade** |
| EyePACS train+test | 88 702 | 88 009 | **−693** | 0.78% short |

DDR landing on 12 522 is the published 13 673 less the 1 151 ungradable grade-5 images —
confirming the grade-5 exclusion behaved exactly as intended. APTOS and Messidor-2 match
their published per-grade distributions image for image.

**The EyePACS shortfall is upstream, not ours.** `cache_report.json` records
`found: 88 009`, `cached: 88 009`, `failed: 0` — the pipeline processed every image the
mirror contained and lost none. The 693 were never in `tantai31124/eyepacs-original`.

**The loss is not grade-independent, and that needs saying in the thesis.**

| Grade | Published | In cache | Lost |
|---|---|---|---|
| 0 | 65 343 | 64 909 | 0.66% |
| 1 | 6 205 | 6 177 | 0.45% |
| 2 | 13 153 | 12 994 | 1.21% |
| 3 | 2 087 | 2 053 | 1.63% |
| 4 | 1 914 | 1 876 | **1.99%** |

χ² against proportional loss is **105.6 on 4 df** — decisively non-random. Grade 4 is
lost at three times grade 0's rate. The likely mechanism is benign and worth stating:
mirrors drop unreadable files, and severe DR correlates with media opacity — cataract,
vitreous haemorrhage — which makes those eyes harder to image. The effect is a mild
selection against the hardest-to-photograph severe cases, so the cohort is, if anything,
marginally easier than the true population.

**The magnitude is negligible.** Grade-0 prevalence moves 73.67% → 73.75% (+0.09 pp) and
rDR 19.34% → 19.23% (−0.11 pp). No reported rate is materially affected.

### Structure checks — all pass

| Check | Value | Expected |
|---|---|---|
| Images per patient | **1.989** | ≈2.0 — confirms EyePACS patient parsing |
| Test fraction | 20.01% | 20% |
| Val fraction of pool | 10.00% | 10% |
| Calibration fraction of pool | 4.99% | 5% |
| Splits sum to total | yes | — |
| Held-out rows identical across all three variants | yes | Rule: variants differ in train only |
| Balanced variant shortfall | **none** — 1 000 available in every grade | — |

### A0 — **CLOSED: PASS, with two recorded limitations**

1. **EyePACS is 0.78% short of the official release, non-randomly by grade.** Upstream
   of this project, negligible in prevalence, and to be stated in the thesis limitations.
2. **Messidor-2 has not had its visual audit.** It is locked until unblinding and its
   counts reconcile exactly; the audit should still happen before Phase 6.

### Correction to the Phase 6 projection

Earlier projections used an estimated training-set size of 57 656. `dataset_plan.json`
gives the real figure: **59 842**. Corrected, at the measured 46.0 img/s:

| Configuration | Phase 6 total | vs ~20 h budget |
|---|---|---|
| **Frozen recipe (B0, 512, no fusion)** | **21.7 GPU-h** | slightly over |
| ResNet50 instead | 31.1 GPU-h | 1.6× over |
| With fusion | 47.3 GPU-h | 2.4× over, beyond a week's quota |

The frozen recipe is ~1.7 h over the stated budget rather than the 0.9 h quoted before.
The decisions do not change — but the Phase 6 budget needs trimming (epochs or seeds)
at the freeze, and that is a pre-registration decision, not a mid-run discovery.

## Stage B — Grading pathway (selection; validation only)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| B1 | What resolution is needed? | 384 / 512 / 768 px | Val QWK **and grade-1 F1** | **DONE** | **512 px** — no trend above noise; 768 upsamples |
| B2 | Which encoder? | EfficientNet-B0 / ResNet50 | Val QWK per GPU-hour | **DONE** | **EfficientNet-B0** — QWK tied, 0.70× the cost |
| B3 | Which head? | CE / ordinal / focal-ordinal | Val QWK + MAE | **DONE** | **focal-ordinal** — focal earns its place; CE ruled out |
| B4 | Which sampler? | Natural / stratified exposure / class-balanced | Val QWK at natural prevalence | **DONE (2 arms)** | **weighted** — the third arm was a duplicate |
| B5 | Does eye-pair fusion help? | Single vs left+right fusion | Val QWK, paired across seeds | **DONE** | **no fusion** — best grader, but confounds the thesis signal |
| B6 | Does balancing the dataset help? *(ablation)* | `eyepacs_full` vs `eyepacs_balanced_1000` | QWK on natural-prevalence test | TODO | |
| B7 | Does DDR improve transfer? **(H3)** | `eyepacs_full` vs `eyepacs_ddr_full` | External QWK — single unblinding | **DONE 2026-09-28** | **H3 not supported.** QWK, DDR − EyePACS-only, per seed: Messidor-2 −0.091 / −0.092 / −0.020; APTOS −0.011 / +0.047 / +0.031 |

> **B1 warning.** Judge on grade-1 **F1**, not overall QWK and not recall alone.
> Grade 1 is microaneurysms only; an MA is 10–20 px. A model that silently skips
> grade 1 still scores ~0.97 QWK on a realistic distribution, because grade 1 sits
> one step from grade 0 and quadratic weights barely punish the error.
>
> Recall alone is not enough either: a model predicting grade 1 for *every* image
> scores 1.000 grade-1 recall and is worthless. Check `distinct_predictions` first —
> when it is 1 the run has collapsed and decides nothing. `03_grading_sweeps.ipynb`
> applies both rules and prints the verdict.

### B1 — resolution — **DECIDED: 512 px**

Validation only, `eyepacs_balanced_1000`, EfficientNet-B0, focal-ordinal head,
stratified exposure, no fusion, seed 42, 10 epochs.

| Resolution | QWK | macro-F1 | **grade-1 F1** | g1 recall | g1 precision | distinct preds | MAE | GPU-min | img/s |
|---|---|---|---|---|---|---|---|---|---|
| 384 | 0.676 | 0.445 | 0.151 | 0.293 | 0.102 | 5 / 5 | 0.418 | 19.4 | 43.0 |
| **512** | 0.679 | 0.459 | 0.142 | 0.271 | 0.096 | 5 / 5 | 0.415 | 18.1 | 46.0 |
| 768 | 0.704 | 0.459 | **0.161** | 0.271 | 0.115 | 5 / 5 | 0.395 | 34.3 | 24.3 |

No run collapsed — all five grades predicted at every resolution — so all three are
readable.

**Grade-1 F1 is not monotone in resolution: 0.151 → 0.142 → 0.161.** The worst point
is 512, in the middle. If resolution drove microaneurysm detection the ordering would
be monotone; it is not, and the entire spread is 0.019. On one seed per point that is
noise, not a trend. Grade-1 recall tells the same story: 0.293 at 384, then 0.271 at
both 512 and 768 — the *lowest* resolution has the best recall.

**768 upsamples.** The cache is 512 px (`build_cache.py --size`, default 512) and the
transform is `Resize((image_size, image_size))`, so the 768 run interpolates 512 px
data up to 768. It adds no information. Whatever produced its +0.025 QWK, it cannot be
"seeing microaneurysms that downsampling destroyed" — there are no extra pixels to see.
This is consistent with its recall being unchanged and only its precision improving.

**384 is not cheaper.** 43.0 img/s against 46.0 at 512. Below 512 the job is bound by
JPEG-decoding the 512 px cache files, not by the GPU, so shrinking the target costs
decode work anyway and saves nothing.

**Decision: 512 px.** It is the cache's native resolution, so it is the only setting
that neither discards real pixels nor invents fake ones; it is the fastest measured;
and the evidence for 768 does not survive the non-monotonicity. 768 would also roughly
double everything downstream — Phase 6 projects to 39.6 GPU-h at 768 against 20.9 h at
512, over both the ~20 h budget and the 30 h/week quota — for a gain on grading
accuracy, which is explicitly not this thesis's contribution.

> **Limitation to state in the thesis.** B1 as specified sweeps 384/512/768, but every
> run reads the same 512 px cache, so the sweep cannot test the claim that motivated
> it — that an MA is 10–20 px at native resolution and vanishes under downsampling.
> Testing that honestly needs a cache rebuilt at 768 from the originals (another full
> Phase 1 pass, ~2.25× the storage). Not done; the grading pathway is a component here,
> not the contribution. Report B1 as choosing an operating point under a fixed 512 px
> cache, not as a resolution-sensitivity study.

### B2 — encoder — **DECIDED: EfficientNet-B0**

Same settings as B1 at 512 px; only the encoder varies. The B0 arm *is* the B1 512 run.

| Encoder | QWK | macro-F1 | grade-1 F1 | g1 recall | g1 precision | MAE | GPU-min | img/s | Phase 6 projection |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | 0.679 | 0.459 | 0.142 | 0.271 | 0.096 | **0.415** | **18.1** | **46.0** | **20.9 h** |
| ResNet50 | 0.676 | 0.464 | **0.180** | **0.413** | 0.115 | 0.469 | 26.0 | 32.1 | 30.0 h |

**On the stated criterion this is not close.** QWK differs by −0.0024 and macro-F1 by
+0.0052 — both inside the noise band B1 established. ResNet50 delivers that tie at
1.44× the wall-clock, so on val QWK per GPU-hour B0 wins outright.

**But grade 1 is the first Stage B signal above noise.** ResNet50 reaches grade-1 F1
0.180 against 0.142, a gap of 0.038 — *twice* the entire 0.019 spread B1 produced
across all three resolutions — driven by recall 0.413 against 0.271 (+52%), with
precision also up (0.115 against 0.096).

**This is an operating point, not a quality level.** QWK and macro-F1 are tied and MAE
is *worse* under ResNet50 (0.469 against 0.415): it is not grading better, it is
predicting grade 1 more liberally. That also explains the QWK/MAE split — a more spread
prediction distribution raises QWK's expected-agreement denominator, so QWK holds while
raw distance error grows. What governs that distribution is the loss and the sampler,
which are exactly what **B3** and **B4** vary, at no extra compute.

> **Prediction recorded before running B3/B4.** If `class_balanced` or non-focal
> ordinal moves B0's grade-1 F1 to ≈0.18, then B0 matches ResNet50's grade-1 behaviour
> at 0.70× the cost and B2 is settled. If nothing in B3–B4 shifts grade 1, the encoder
> is contributing something the sampler cannot, and B2 should be reopened as a recorded
> deviation. Stating this now so the follow-up is a test rather than a rationalisation.

**Decision: EfficientNet-B0.** It wins the pre-stated criterion, has the better MAE, and
keeps Phase 6 at 20.9 GPU-h rather than 30.0 — the latter being over the ~20 h budget
and at the ceiling of the 30 h/week quota.

### B3 — output head — **DECIDED: focal-ordinal**

512 px, EfficientNet-B0, stratified exposure, no fusion. `ordinal_focal` is the B1 512 run.

| Head | QWK | macro-F1 | grade-1 F1 | g1 recall | g1 precision | MAE | GPU-min |
|---|---|---|---|---|---|---|---|
| `softmax_ce` | 0.647 | **0.488** | **0.159** | 0.182 | **0.141** | 0.416 | 18.9 |
| `ordinal` (γ=0) | **0.700** | 0.398 | 0.122 | 0.148 | 0.104 | **0.389** | 16.3 |
| **`ordinal_focal` (γ=2)** | 0.679 | 0.459 | 0.142 | **0.271** | 0.096 | 0.415 | 18.1 |

**This is the first Stage B experiment where the choice actually matters.** QWK spans
0.053 across the three heads — nearly twice the 0.028 that B1 produced across every
resolution. The head is a real lever; resolution and encoder were not.

**A correction to how the noise band has been quoted.** "±0.02 QWK" was a round number
used conversationally. B1's actual spread, which we judged non-separable from seed
noise on one seed per point, gives an empirical floor *per metric*: QWK 0.0276,
macro-F1 0.0144, grade-1 F1 0.0190, grade-1 recall 0.0220, MAE 0.0230. Effects are
judged against the floor for their own metric from here on.

**`softmax_ce` is ruled out on the stated criterion.** Its QWK deficit against
`ordinal` is 0.053 — 1.9× the QWK floor, so a real effect, not noise. It has the best
macro-F1 and the best grade-1 F1, but dropping the ordinal head to get them also
discards the CORAL monotonicity guarantee that Phase 7 calibration and the rDR/VTDR
heads are built on. Not worth it.

**`ordinal` versus `ordinal_focal` isolates focal weighting exactly** — same head, same
architecture, only γ differs. Measured against each metric's own floor:

| Metric | focal − ordinal | B1 floor | × floor | Favours |
|---|---|---|---|---|
| QWK | −0.0211 | 0.0276 | 0.8× | ordinal — **inside noise** |
| MAE | +0.0260 | 0.0230 | 1.1× | ordinal — marginal |
| grade-1 F1 | +0.0200 | 0.0190 | 1.1× | focal — marginal |
| macro-F1 | **+0.0611** | 0.0144 | **4.2×** | focal — decisive |
| grade-1 recall | **+0.1230** | 0.0220 | **5.6×** | focal — decisive |

The stated criterion (QWK + MAE) mildly favours plain `ordinal`, but by margins of
0.8× and 1.1× their floors — it does not separate the two. The rare-class metrics
favour focal at 4.2× and 5.6×. On weight of evidence focal wins, and the criterion is
applied rather than overridden: it simply fails to discriminate here.

> **Reportable finding: focal weighting does its stated job.** `docs/03_model_architecture.md`
> justifies γ=2 as handling EyePACS's ~36:1 imbalance without discarding data. Turning it
> off costs **83% of grade-1 recall** (0.271 → 0.148) and 0.061 macro-F1, for a QWK change
> inside the noise floor. That is a clean, isolated confirmation of a design choice, and it
> belongs in the write-up.

**Decision: `ordinal_focal`.** No new baseline is needed — the B1 512 run remains the
reference configuration for B4 and B5.

**B2's recorded prediction is still open.** Neither head reached ResNet50's grade-1 F1
of 0.180: plain `ordinal` moved it *down* to 0.122, and `softmax_ce` reached 0.159 only
by trading 0.053 QWK — and by a different mechanism (precision 0.141 / recall 0.182,
against ResNet50's precision 0.115 / recall 0.413). B4's `class_balanced` is the
remaining test.

### B4 — sampler — **DECIDED: weighted (`stratified_exposure`)**

| Sampler | QWK | macro-F1 | grade-1 F1 | g1 recall | MAE | GPU-min |
|---|---|---|---|---|---|---|
| `natural` | 0.640 | 0.372 | 0.125 | 0.263 | 0.479 | 10.9 |
| **`stratified_exposure`** | **0.679** | **0.459** | 0.142 | 0.271 | **0.415** | 18.1 |
| `class_balanced` | 0.679 | 0.459 | 0.142 | 0.271 | 0.415 | 18.8 |

> ### The third arm was a duplicate. B4 ran as a two-way, not a three-way.
>
> `class_balanced` returned metrics **identical to the B1 512 baseline on every
> figure** — QWK, macro-F1, grade-1 F1, grade-1 recall, MAE and distinct predictions,
> all to the last recorded digit. That is not coincidence: in `make_sampler` both
> weighted strategies weight by inverse grade frequency, and differ only in draws per
> epoch, which default to `len(grades)` for both. With `epoch_samples` unset they are
> one sampler under two names and, given the same seed, emit the same index sequence.
> Verified directly: the two index lists compare equal.
>
> Nothing downstream is invalidated — B1–B3 all used `stratified_exposure`, which is a
> real sampler — but B4 tested two options, not three. `make_sampler` now warns when
> the two coincide, and `tests/test_grading.py` pins the equivalence so it cannot
> resurface silently.

**On the comparison that did run, weighted beats natural clearly:** macro-F1 +0.087
(6.0× its floor), MAE −0.064 (2.8×), QWK +0.039 (1.4×).

**But weighting does not help grade 1.** Grade-1 F1 +0.017 (0.9× floor) and recall
+0.008 (0.4×) are both inside noise. Read alongside B3, the division of labour is
clean: **focal weighting is what moves grade 1** (+83% recall), **the sampler is what
moves overall class balance** (+6 floors of macro-F1). They are not substitutes.

### B5 — eye-pair fusion — **DECIDED: no fusion**

| Configuration | QWK | macro-F1 | grade-1 F1 | g1 recall | MAE | GPU-min | Phase 6 |
|---|---|---|---|---|---|---|---|
| **single image** | 0.679 | 0.459 | 0.142 | 0.271 | 0.415 | **18.1** | **20.9 h** |
| fusion | **0.713** | **0.484** | 0.152 | **0.317** | **0.392** | 39.5 | 45.5 h |

**Fusion produces the best grader in Stage B** — top QWK, macro-F1, grade-1 recall and
MAE. QWK +0.034 (1.2× floor), macro-F1 +0.025 (1.7×), grade-1 recall +0.046 (2.1×).
It is rejected anyway, on three grounds.

1. **It confounds the signal the thesis rests on.** With fusion, M1's grade for image X
   is a function of X *and* its fellow eye; M2's evidence is a function of X alone. When
   they disagree, that can mean "the fellow eye carried disease M2 cannot see" rather
   than "the grade contradicts the evidence in this image". DR is frequently asymmetric,
   so this is not a corner case — it systematically inflates disagreement in exactly the
   cases the referral rule would flag, and those flags would not be errors. `CLAUDE.md`:
   *never optimise for QWK at the expense of the disagreement signal.*
2. **The budget cannot absorb it.** 21.1 img/s against 46.0 puts Phase 6 at **45.5
   GPU-h** against 20.9 — over the ~20 h budget and over a full 30 h/week quota.
3. **The evidence is thin for the criterion.** B5 was specified as *paired across
   seeds*; one seed per arm was run, and ΔQWK is 1.2× its floor. Grade-1 F1 (+0.010,
   0.5×) and MAE (1.0×) are inside noise.

> If the thesis were about grading accuracy, fusion would win. It is not, and buying
> 0.034 QWK by contaminating the referral signal is the specific trade `CLAUDE.md`
> forbids.

### Stage B closed — the frozen recipe

**512 px · EfficientNet-B0 · focal-ordinal head · stratified exposure · no fusion.**

Which is exactly the `B1_res512` configuration, and exactly the default specified in
`docs/03_model_architecture.md`. **Every ablation either lost or tied.** Eight runs,
~2.8 GPU-hours, and the designed recipe survived all of them — a legitimate and
reportable outcome, not a null result to bury.

What actually mattered, ranked by effect size against each metric's own floor:

| Lever | Biggest effect | × floor | Verdict |
|---|---|---|---|
| Head: focal on/off | grade-1 recall +0.123 | 5.6× | **Real and decisive** |
| Sampler: weighted vs natural | macro-F1 +0.087 | 6.0× | **Real and decisive** |
| Fusion on/off | grade-1 recall +0.046 | 2.1× | Real, rejected on confound + cost |
| Encoder: B0 vs ResNet50 | grade-1 F1 +0.038 | 2.0× | Real, rejected on cost |
| Resolution 384/512/768 | grade-1 F1 spread 0.019 | 1.0× | Noise |

**The B2 prediction resolves against reopening.** No configuration except ResNet50
reached grade-1 F1 0.180 — the closest were `softmax_ce` at 0.159 (costing 0.053 QWK)
and fusion at 0.152 (costing 2.18× compute). Note the designated test, `class_balanced`,
was the duplicate arm and so never ran as a distinct condition; what *was* tested is
full inverse-frequency weighting, which every run except `B4_natural` used, and it
yields 0.142. B2 is **not** reopened: ResNet50's advantage is recall-led, costs 9 extra
Phase 6 GPU-hours, and grading accuracy is not the contribution. Recorded here so the
reasoning is auditable rather than assumed.

> **B7 protocol.** Needs external data. Legitimate only because the pre-registration
> declares in advance that exactly these two variants get evaluated in the single
> unblinding pass.

## Stage C — Evidence pathway

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| C1 | Can we locate disc and fovea well enough? | — | Euclidean error on IDRiD test, in disc diameters (**target < 0.5 DD**), against the constant-predictor baseline | **DONE** | **FAIL, 0.686 DD** — fovea fine (0.289), disc is the miss (1.082) |
| C2 | How well do the 4 lesion channels segment? | DDR-seg / +IDRiD / +augmentation | Per-lesion Dice, IoU, **over images where the lesion is annotated** | **DONE (DDR only)** | mean Dice 0.497; MA 0.316 |
| C3 | Does the segmenter transfer? | C2's checkpoint on held-out IDRiD | Dice drop across domains | **DONE** | **0.425 vs 0.505** — −16% relative, no collapse |
| C4 | How informative is evidence alone? | M3 over M2a's predicted lesions | QWK of evidence-grade vs label | **DONE** | **QWK 0.375** vs M1's 0.679 — informative but weaker |

> **C4 is load-bearing.** The evidence path must be *informative but weaker* than the
> grader. As good ⇒ the grader is redundant. Noise ⇒ disagreement means nothing.
> Report it honestly either way — the thesis needs it interpretable, not good.
>
> It is **not in `notebooks/04_phase4.ipynb`**, because it runs the reasoner and
> Phase 4 does not build one. It belongs with M3.

### What the code does, and why

**Read C2's Dice over images where the lesion is annotated**, which is the figure
`dice_present` reports and the notebook prints. Most fundus images carry no soft
exudates, so averaging over every image folds in a long run of empty-target /
empty-prediction pairs scoring 1.0 by convention — a headline that rises without the
model having segmented anything. `dice_all` sits beside it so the gap is visible
rather than assumed.

**Two failure modes are instrumented rather than trusted:**

- `false_positive_images` — images with no such lesion where the model predicted one.
  Over-segmentation is the failure that matters here: a model that finds lesions
  everywhere agrees with every grade, and disagreement carries no information.
- `mean_pred_px` beside `mean_truth_px` — a plausible Dice with predicted area an
  order of magnitude off means the overlap is coincidental.

**C1's baseline is as important as its gate.** Fundus framing is stereotyped, so a
constant predictor scores better than intuition suggests. A model that barely beats it
has learned the average layout, not this image's landmarks, and the gate would then be
passing for the wrong reason. `train_geometry.py` reports the margin; the notebook
warns when it is ≤ 0.

**C1 and C2 are independent.** `docs/03_model_architecture.md` has M2a and M2b sharing
an encoder, so C2 loads C1's via `--encoder-from` when a checkpoint exists. Their
supervision is disjoint, so they are fitted in sequence rather than jointly. When C1 is
blocked, C2 starts from ImageNet weights — **a deviation to record, not a blocker.**

### C1 — RUN, and it FAILS the gate: 0.686 DD against 0.5

413 IDRiD Part B images, 330 train / 83 val, 512 px, 17 epochs (early stopped
from best epoch 6), 1.3 GPU-minutes.

| | mean px | median px | mean DD | median DD |
|---|---|---|---|---|
| Optic disc | 62.0 | 26.6 | **1.082** | 0.354 |
| Fovea | 21.0 | 18.5 | **0.289** | 0.246 |
| **Both** | | | **0.686** | 0.300 |

Constant-predictor baseline: 1.440 DD. Within 0.5 DD: 74.7% of landmarks,
against 11.4% for the baseline. Mean disc diameter 75.4 px at 512.

**The gate fails, and that is the recorded result.** The criterion was fixed
before the run and it is a mean over both landmarks.

**Three things the aggregate hides, all of which matter for what M3 can do:**

1. **The fovea passes comfortably** at 0.289 DD; the disc alone misses at
   1.082. They are separate outputs of one head, and the failure is not shared.
2. **The disc error is heavy-tailed** — mean 62.0 px against median 26.6 px, a
   ratio of 2.3. On the median image the disc lands at 0.354 DD, inside the
   gate. A minority of images are far enough out to carry the mean past it.
   "Most images slightly off" and "a few placed elsewhere entirely" have
   different causes and different fixes, and the aggregate cannot tell them
   apart.
3. **It is not the baseline-matching failure** the gate exists to catch: it
   beats the constant predictor by 0.754 DD and lifts within-0.5 from 11.4% to
   74.7%. The model has learnt something real about where landmarks are.

Also visible in the history: train loss falls from 0.084 to 0.014 while val
loss sits at ~0.030 from epoch 6 on. With 330 training images this overfits
early, and early stopping halted while the *fovea* error was still improving
(36.4 → 18.1 px) because the disc term dominates the stopping metric.

**Diagnostics added, not a changed gate.** `train_geometry.py` now writes
`val_errors.csv` — every validation image with its predicted and true
coordinates, per-landmark error in pixels and disc diameters, sorted worst
disc error first — and reports the median beside the mean plus a `side_flipped`
count. The disc sits nasal to the macula, so a disc predicted on the wrong side
of the fovea is a laterality error rather than an imprecise one, and would mean
the tail has one systematic cause. **The median is a diagnostic. The gate stays
the mean, and C1 stays FAILED, whatever `val_errors.csv` shows.**

> **What this costs M3, pending the diagnosis.** The 4-2-1 rule needs quadrants,
> and quadrants need both landmarks. The fovea is reliable; the disc is not, on
> a minority of images. If the tail turns out to be laterality, it is fixable
> and quadrant reasoning survives. If the disc is simply imprecise everywhere in
> the tail, M3 falls back to count-only rules — which
> `docs/00_START_HERE.md` names as the designed fallback, not a failure to hide.

### C1 diagnosed — the tail is laterality, and it is 47% of the error

`val_errors.csv`, 83 validation images, best epoch 6.

| | median | p75 | p90 | max | > 0.5 DD |
|---|---|---|---|---|---|
| Optic disc | 0.354 | 0.717 | 3.461 | 4.501 | 32 (39%) |
| Fovea | 0.246 | 0.391 | 0.515 | 1.099 | 9 (11%) |

**10 of 83 images (12%) place the disc on the wrong side of the fovea.** On those
the median disc error is **3.875 DD**; on the other 73 it is **0.333 DD** — 11.6×
worse. The eight worst images are all flips and carry **47% of the total disc
error** between them. Their fovea error is unaffected (0.06–0.46 DD), and the
predicted coordinates are mirror images of the truth:

| image | OD error | predicted x | true x |
|---|---|---|---|
| IDRiD_076 | 4.50 DD | 405.0 | 65.8 |
| IDRiD_054 | 4.16 DD | 72.9 | 386.1 |
| IDRiD_077 | 4.07 DD | 104.2 | 407.3 |

This is not imprecision. The model locates the fovea correctly, then puts the disc
on the opposite side of it — a **laterality error**, the single systematic cause the
`side_flipped` flag was added to look for.

**What it implies.** Excluding the flips, the disc median is 0.333 DD and the worst
non-flipped image is 1.473 DD, so a model that got laterality right would very likely
clear the 0.5 gate. The failure is one recoverable mode, not a diffuse inability to
find the disc.

**The gate still FAILS at 0.686 DD.** This diagnosis does not change the recorded
result, and no post-hoc statistic may replace the pre-specified mean.

> **Why direct coordinate regression struggles here.** The disc sits nasal to the
> macula, so its position is bimodal in x — left for one eye, right for the other —
> and which mode applies is a property of the image that a single regressed
> coordinate must commit to. A heatmap head can represent both modes and let the
> evidence choose; a regressor cannot. That, not capacity, is the likely reason a
> model with a 0.333 DD median flips outright on 12% of images.

### C2 — DONE, but on DDR alone: mean Dice 0.497

604 train / 151 val, 512 px, best epoch 32 of 48, 17.9 GPU-minutes.

| lesion | Dice (present) | IoU | images | FP images | pred px | true px |
|---|---|---|---|---|---|---|
| microaneurysm | 0.3159 | 0.2014 | 120 | 26 | 82 | 90 |
| haemorrhage | 0.4808 | 0.3437 | 116 | 18 | 793 | 1088 |
| hard_exudate | 0.5417 | 0.3993 | 104 | 32 | 557 | 451 |
| soft_exudate | 0.6504 | 0.5178 | 55 | 42 | 325 | 300 |
| **mean** | **0.4972** | 0.3656 | | | | |

**Microaneurysm segments at 0.316**, which is the number that matters most for this
thesis: grade 1 is microaneurysms only, and Stage B established that the grader
reaches just 0.142 grade-1 F1. The evidence pathway sees MA *better than the grader
does*, which is the asymmetry the disagreement rule is built on. Predicted MA area
(82 px) also tracks the truth (90 px), so the overlap is not coincidental.

**Over-segmentation is the weak point, and it rises as lesions get rarer.** Soft
exudate scores the best Dice (0.650) on only 55 annotated images while firing falsely
on **42** of the 96 images that have none — 44%. Hard exudate: 32 false-positive
images. A channel that fires on nearly half the negatives inflates evidence counts and
makes disagreement fire for the wrong reason, so this belongs in M3's confidence
handling rather than being read as a good Dice.

**`dice_all` runs *below* `dice_present` here** (MA 0.284 vs 0.316, EX 0.472 vs 0.542),
the opposite of the inflation the all-image average usually produces. With this many
false-positive images, the empty-target images score 0 rather than the conventional
1.0. Both figures are reported for exactly this reason: which way the gap runs is a
property of the data, not something to assume.

> **C2 ran on DDR only — IDRiD never entered.** The notebook chose its population from
> the Phase 2 manifests and fell back to the cache only if none had masks.
> `ddr_manifest.csv` has them, so the fallback never fired, and IDRiD's manifest has
> none: `prepare_manifest.py` builds it from the Part B grading tables, which name the
> 516 graded images and not one of Part A's 81 — and Part A is the whole IDRiD mask
> set. Fixed: C2/C3 now take their population from the cache directly, since the
> manifests answer "which images have grades" and this experiment needs "which images
> have lesion annotations". In IDRiD those sets are disjoint.
>
> **So C2's numbers are a DDR-only result and must be reported as such**, and the
> "+IDRiD" arm the register specifies has not run.

### C3 — restructured: DDR trains, IDRiD is held out

First attempt: `SKIPPED - cross-domain needs both DDR and IDRiD, found {'ddr'}`,
because IDRiD was absent from C2's population.

**The design changed rather than just the bug being fixed.** C2 now trains on DDR
with IDRiD held out entirely, which makes IDRiD a genuine external domain for the
evidence pathway — and means **the cross-domain measurement needs no second training
run**. Evaluating C2's own checkpoint on IDRiD *is* the DDR→IDRiD answer;
`train_evidence.py --eval-only` does exactly that and trains nothing.

| | |
|---|---|
| C2 | trains DDR (~757 annotated images), IDRiD never seen |
| C3a | C2's checkpoint evaluated on IDRiD — free, and the headline transfer number |
| C3b | train IDRiD → test DDR — **off by default** |

**No leakage between C1 and C2/C3.** C1 trains on IDRiD **Part B** (413 graded images
with Part C centres); C2/C3 use IDRiD **Part A** (81 images with lesion masks). Those
are different image sets — the same Part A / Part B split that blocked C1 in the first
place now works in the project's favour.

**C3b is off deliberately.** IDRiD carries roughly 81 annotated images, so a weak
result training on it would be confounded by sample size rather than domain shift and
would not answer the question either way. `RUN_REVERSE = True` runs it; the image count
must be reported beside any number it yields.

**Deviation to record:** C2's specified `+IDRiD` training arm does not run under this
design. The trade is deliberate — an external domain for C3 is worth more to this
thesis than 81 extra training images, because the thesis claims something about
disagreement generalising, not about segmentation accuracy.

### C4 — the premise survives: QWK 0.386 against M1's 0.679

12 522 graded DDR images, M2a's predicted masks through M3's rules, count-only
(R4 declined — C1 failed its gate).

| | |
|---|---|
| QWK vs true grade | **0.3858** |
| exact agreement | 55.6% |
| within one grade | 65.5% |
| distinct evidence grades | **3 of 5** |
| rules fired | R1 2 599 · R2 661 · R3 5 089 · R3\* 4 173 |

**This is the result the project needed.** `docs/00_START_HERE.md` requires the
evidence path to be *informative but weaker* than the grader: matching M1 would make
M1 redundant, and noise would mean disagreement carries no signal and H1 cannot hold
for the stated reason. 0.386 against 0.679 is squarely in between. **The premise
holds and F2 in Phase 7 is worth running.**

Four qualifications, all of which belong beside the number:

1. **It is a floor, not an estimate.** Only three of five grades are reachable.
   Grade 3 needs R4, which needs quadrants, which need C1 — so every severe case can
   at best read as moderate. Grade 4 needs neovascularisation, which nothing
   annotates. A reasoner that cannot emit two of the five classes is being scored
   against all five.
2. **R3\* fired on 4 173 images — a third of the set.** That is haemorrhage or
   exudate found with *no* microaneurysm detected, a combination ICDR has no rung
   for. It is M2a's MA channel (Dice 0.344) missing what its larger-lesion channels
   find. The rule exists so those images are not graded 0, but a third of the
   corpus resting on a fallback rung is a limitation, not a detail.
3. **The split is not M1's.** M1's 0.679 is validation on EyePACS; this is DDR. The
   comparison says "same order of magnitude, clearly weaker", which is what the
   premise needs, and it is not a like-for-like contest.
4. **Contaminated, mildly.** All 12 522 DDR images were scored, and 757 of them are
   the annotated subset M2a was fitted and validated on — about 6%. Fixed for the
   next run with `--exclude-manifest`, which leaves only images the segmenter has
   never seen. The effect is small at 6% but the figure should be regenerated
   before it reaches the thesis.

### C1b — the heatmap head is WORSE: 1.028 DD against 0.686

| head | mean DD | OD | fovea | within 0.5 DD | vs constant |
|---|---|---|---|---|---|
| coordinate regression | **0.686** | 1.082 DD | 0.289 DD | 74.7% | +0.754 |
| heatmap | 1.028 | 1.594 DD | 0.461 DD | 66.3% | +0.412 |

| head | mean DD | fovea DD | **laterality flips** |
|---|---|---|---|
| coordinate regression | **0.686** | 0.289 | **10 / 83** |
| heatmap | 1.028 | 0.461 | **27 / 83** |

**The hypothesis is falsified, and the flip count is decisive.** The heatmap head was
introduced specifically to fix laterality: a regressed coordinate must commit to one
mode, a heatmap can hold both peaks and let argmax choose. It produced **more than
twice as many flips** — 33% of images against 12% — as well as worse error everywhere.

**But this refutes the remedy, not the diagnosis, and the distinction matters.** The
heatmap head is worse on the *fovea* too (0.461 against 0.289), and the fovea has no
bimodality to resolve. A head that is worse where the hypothesis cannot apply is not
evidence about the hypothesis; it is evidence that the head is badly fitted. The extra
flips are most likely a symptom of that — a weak, noisy heatmap gives a noisy argmax —
rather than a verdict on bimodality.

So the honest position is: **C1 fails at 0.686 DD; its error tail is laterality flips;
the cause of those flips is not established; one remedy was tried and refuted.**

> **Why the head is badly fitted, as a hypothesis and not tested.** The loss is plain
> MSE against a Gaussian with sigma 2 on a 128x128 grid: roughly 25 pixels of signal
> against 16 000 of background, so predicting all-zero is a good local minimum and the
> positives get almost no gradient. Standard heatmap regression weights the positives
> or uses focal MSE for exactly this reason, and this implementation does neither.
>
> **Not pursued.** A weighted-loss retrain costs 1.3 GPU-minutes, but two attempts at
> this component have now failed and the project has a designed fallback for exactly
> this case. Landmark accuracy is not the contribution; spending further attempts here
> buys a rung on the reasoner's ladder at the cost of the work that is actually novel.
> Recorded as an option, deliberately declined.

**A methodological error made this comparison harder than it needed to be.** `--head`
defaults to `heatmap`, and section 6 of the notebook did not state a head, so the
baseline arm silently became a second heatmap run and section 6b compared heatmap
with heatmap. Same defect as B4's duplicate sampler arm: an experimental arm riding
on a default. Both cells now state their head explicitly.

**C1 stays FAILED at 0.686 DD**, which remains the recorded result — the better of
the two and the pre-specified architecture. M3 runs count-only.

### A config file that misdescribed its own directory

The failed `--head regress --resume` attempt exposed a separate defect:
`config.json` was written **before** the resume guard ran. A refused resume therefore
left `C1_geometry/config.json` claiming `head=regress` beside a `best.pt`,
`metrics.json` and `val_errors.csv` that were all the heatmap run's. Two result
directories then printed identical numbers under different head labels, and the config
— the one file a reader would trust to say what produced the artefacts — was the thing
that was wrong.

Fixed in both training scripts: the config is written only after the guard passes.
Verified by running heatmap, then attempting a refused regress resume, and confirming
the config still reads `heatmap`.

### C3 — the evidence pathway TRANSFERS: 0.425 against 0.505

C2's checkpoint, trained on DDR and having never seen IDRiD, evaluated on IDRiD's
81 annotated Part A images. No second training run.

| | mean Dice (present) |
|---|---|
| C2, in-domain (DDR val) | 0.5045 |
| C3, DDR → IDRiD | **0.4254** |
| drop | −0.0792, **−15.7% relative** |

**This is the result that makes the disagreement rule worth testing off-domain.** The
experiment was specified to detect a collapse: if Dice halved across sources, then
disagreement measured on one source would say little about another and the thesis
would have to state that plainly. It did not halve. A 16% relative drop on a model
that has never seen the target domain is ordinary transfer degradation, not failure.

It bears directly on **H1′** — that the H1 ranking survives dataset shift. A signal
that only works in-domain is a curiosity; this is the first evidence that the evidence
pathway is not one.

**Qualifications.** IDRiD contributes 81 images, so each lesion channel is evaluated on
a few dozen at most and the per-channel figures in `C3_ddr_to_idrid/metrics.json`
carry wide uncertainty — the mean is the defensible number here. The reverse direction
(train IDRiD → test DDR) was deliberately not run: with 81 training images a weak
result would be confounded by sample size rather than domain shift.

### C4 — final, uncontaminated: QWK 0.375

Re-run with `--exclude-manifest`, so the 755 images M2a was fitted on are gone and
**11 767** images remain that the segmenter has never seen.

| | contaminated | clean |
|---|---|---|
| n | 12 522 | **11 767** |
| QWK | 0.3858 | **0.3754** |
| exact agreement | 55.6% | 54.3% |
| within one grade | 65.5% | 63.9% |

The contamination was worth about **+0.010 QWK** — small, as expected at 6%, and now
removed. **0.375 against M1's 0.679 is the recorded figure.**

Rules fired: R1 2 596 · R2 634 · R3 4 429 · **R3\* 4 108**.

**R3\* remains the largest caveat: 35% of the corpus.** That is haemorrhage or exudate
detected with *no* microaneurysm, a combination ICDR has no rung for, and it is M2a's
MA channel (Dice 0.344) missing what its larger-lesion channels find. The rule exists
so those images are not graded 0, but a third of the corpus resting on a fallback rung
is a limitation to state, not a detail.

### The C2/C3 population, for the record

`segmentation_manifest.csv`, built from the cache and published inside
`verify-dr-stage-c`, is the provenance record for which images Stage C used.

| source | images | resolved from |
|---|---|---|
| DDR | 757 | `verify-dr-cache-512` |
| IDRiD (Part A only, `IDRiD_01`–`81`) | 81 | `verify-dr-idrid` |
| **total** | **838** | no duplicates |

`grade` is −1 throughout: this is the set of images carrying lesion annotations, not a
grading manifest, which is why C4 reads `ddr_manifest.csv` instead. IDRiD resolving to
`verify-dr-idrid` rather than to the Phase 1 build is the split-cache fix working —
that mis-resolution is what blocked C3 on two separate runs.

## Stage C closed

| | result | verdict |
|---|---|---|
| C1 | 0.686 DD against a 0.5 gate | **FAIL** — M3 runs count-only |
| C2 | mean Dice 0.505; MA 0.344 | done, DDR only |
| C3 | 0.425 on held-out IDRiD, −16% | **transfers** |
| C4 | QWK 0.375 against M1's 0.679 | **informative but weaker — the premise holds** |

**The premise the thesis rests on survived its own falsification test.** C4 was
specified to be capable of killing the project: an evidence path as good as the grader
makes the grader redundant, and one at chance makes disagreement meaningless. 0.375
against 0.679 is neither.

**Total cost: ~0.9 GPU-hours** (C1 1 min, C2 18 min, C3 1 min, C4 ~10 min, plus the
refuted heatmap attempt at 3 min).

**Phase 4's exit condition is met** — C1 against its target, C2–C4 logged with real
numbers — with C1's failure and its documented fallback as the one deviation to carry
into the freeze.


The split-cache fix (`repath_to_cache` preferring the mask-bearing root) landed, but
C3 did not produce metrics in this run either. Diagnosis needs section 5's
per-dataset counts and section 8's output, which have not been seen.

### C1 was blocked on first run — cause found and fixed

The first C1 attempt had **no training targets**, and every run before it still
reported success. Two independent faults:

1. **Phase 1 cached the wrong IDRiD part.** IDRiD ships Part A (81 images with lesion
   masks, `IDRiD_01`–`81`) and Part B (516 graded images, `IDRiD_001`–`516`) as
   *different image sets*; the Part C centre tables cover Part B. `IDRiD_01` is not
   `IDRiD_001`, so a Part A-only cache joins to neither.
2. **`02_manifests.ipynb` resolved `--coords-source-dir` by path keyword**, requiring
   `"segmentation"` in it — which pins it to Part A. Every coordinate row missed,
   `project_coords` returned nothing, and the notebook's `--no-grades` fallback then
   wrote a perfectly valid manifest with an empty geometry column.

Fixes, all verified against a fixture reproducing the real directory layout:

| Where | Change |
|---|---|
| `notebooks/01c_idrid_grading.ipynb` | new — caches Part B beside Part A's masks, carrying the existing cache forward so republishing cannot delete the masks. Gates on *images named by the Part C tables*, the number that decides whether C1 can run. |
| `notebooks/02_manifests.ipynb` | resolves the coordinate source by **matching the tables' own IDs**, scoring each candidate directory and its parent (Part B splits Training/Testing into siblings). Prints the match counts. |
| `scripts/prepare_manifest.py` | `--coords` given but nothing projected is now **exit 1 with no manifest written**, naming the Part A/Part B confusion and the fix. |

On the fixture the old discovery resolved **0 of 50** coordinate rows and the new one
**50 of 50**; `prepare_manifest.py` then reported `coordinates projected for 50 images`
and C1 trained end to end. The fixture's coordinates are random, so C1 correctly
*fails* its gate there at 3.726 DD and beats the constant baseline by only +0.059 DD —
which is the "learned the average layout" case the baseline exists to expose.

> **The general lesson, worth a line in the thesis' methods.** Every one of these
> silent failures had the same shape: a pipeline stage that could not do its job still
> produced a well-formed artefact. The countermeasure that keeps working is to gate on
> *the quantity the next stage actually consumes* — images carrying coordinates, not
> images cached — and to make a zero there fatal rather than merely printed.

**Verified end to end** on a synthetic fixture (75 images, MA 67 / HE 51 / EX 43 /
SE 14, SE deliberately sparse): mean Dice 0.24 → 0.68 over 12 epochs, rising on every
channel with enough images to support a figure. That establishes the code trains — it
says nothing about real lesions, and the fixture's numbers must never reach the thesis.

## Stage D — Calibration (H2)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| D1 | Calibrated internally? | Uncalibrated vs temperature scaling | ECE, NLL, Brier | **DONE 2026-09-28** | Yes. In-domain ECE 0.24–0.27 raw → 0.039–0.062 with stages 0+1; 0.016–0.020 with BCTS (D13) |
| D2 | Does internal temperature transfer? | Internal T applied to APTOS / Messidor-2 | External ECE — **expected to fail** | **DONE 2026-09-28** | Partly. Stages 0+1 externally: APTOS 0.04–0.11, Messidor-2 0.09–0.13, against 0.18–0.25 raw and 0.04–0.06 in-domain |
| D3 | Does prior-shift correction fix it? | Saerens–Decock EM vs oracle prior | External ECE, oracle as upper bound | **DONE 2026-09-28** | **No — H2 not supported.** EM worsens ECE in all 12 model × set cells (by 0.04–0.25); amended, 11 of 12. The oracle prior helps Messidor-2 (0.04–0.07) but hurts APTOS (0.07–0.16) |
| D4 | How much unlabelled target data does EM need? | 50 / 200 / 500 / all | ECE vs sample size | **DONE 2026-09-29** | **More data does not help: the failure is bias, not noise.** Registered ECE after EM, n = 50 → all: APTOS 0.164 → 0.168, Messidor-2 0.183 → 0.184. Amended: APTOS 0.195 → 0.241 (worse); Messidor-2 0.266 → 0.118, still above BCTS alone (0.093) |

> D2 failing is the *finding*, not a bug. Grade-0 prevalence moves ~73% → ~49%
> between EyePACS and APTOS; a temperature fitted on one prior cannot transfer.

## Stage E — Verification / faithfulness

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| E1 | Does removing cited lesions change the grade? | Inpaint predicted lesion regions | Δgrade distribution | **DONE 2026-09-29** | Yes. Median Δ expected grade 0.041 (EyePACS test), 0.206 (APTOS), 0.145 (Messidor-2) |
| E2 | **Is that shift lesion-specific?** | Random / non-lesion regions, equal area | Δ(lesion) vs Δ(random) | **DONE 2026-09-29** | **Yes.** Lesion removal beats the mean of 19 equal-area random removals in 83% / 86% / 91% of lesion images, and beats **all 19** in 67% / 70% / 78%, against about 5% by chance |
| E3 | Does faithfulness hold at every severity? | Stratified by true grade | Δgrade by grade — grade 1 is the hard case | **DONE 2026-09-29** | Strongest on grades 1–3 (faithful 75–96%). Weakest on grade 0 (47–67%: M3's lesions there are mostly false) and grade 4 (59–74%) |

> **E2 is mandatory.** Without the control, E1 shows the model reacts to inpainting,
> not that it uses the lesions. Every reviewer asks for this.

## Stage F — Selective triage (H1 — the headline)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| F1 | Best gating signal, internally? | none / confidence / OOD-z / **disagreement** / combined | Coverage–accuracy AUC; accuracy @ 80% and 90% coverage | **DONE 2026-09-28** | **Confidence** (AUC 0.917). Disagreement 0.746 = no gate (0.747); amended 0.801. **H1 not supported** |
| F2 | **Does the ranking survive shift?** | Same five arms on APTOS + Messidor-2 | Same metrics, externally | **DONE 2026-09-28** | Confidence best on both (0.860, 0.841); amended, the combined policy ties it on APTOS (0.860). Disagreement 0.549 and 0.659, at or below no gate; amended 0.672 and 0.716. **H1′ not supported** |
| F3 | What does it mean clinically? | Same arms | rDR sensitivity/specificity at fixed coverage | **DONE 2026-09-29** | At 80% coverage, confidence raises the accepted set's rDR **sensitivity** (EyePACS 0.695 → 0.835, Messidor-2 0.834 → 0.917). Amended disagreement raises **specificity** instead (EyePACS 0.978 → 0.987, APTOS 0.827 → 0.867, Messidor-2 0.932 → 0.951), not sensitivity |
| F4 | Which form of the signal works? | Binary / magnitude / magnitude + unobservable set | Coverage–accuracy AUC | **DONE 2026-09-29** | Registered: binary ≥ magnitude (EyePACS 0.773 vs 0.746); the frozen form was the weaker one. Reported, not selected (§6.3). Amended: identical to 0.001 |
| F5 | Where do cases actually land? | — | Fraction and accuracy per action bucket | **DONE 2026-09-29** | Registered: DEFER takes 55% of EyePACS test images at 81% accuracy against ACCEPT's 85%, so deferral carries almost no information. Amended: DEFER 22% at 55% against ACCEPT 84% |

> **F2 is the thesis.** A signal that only wins in-domain is a curiosity; one that
> holds under shift is a contribution.

## Stage G — Error analysis (no new training)

| ID | Question | Status | Result |
|---|---|---|---|
| G1 | Which errors does disagreement catch that confidence misses? Set overlap + real examples | **DONE 2026-09-29, exploratory** | Registered disagreement points at *correct* cases (error AUROC 0.38–0.45). Amended, it adds information beyond confidence only under shift (within-quintile AUROC: EyePACS 0.52, APTOS 0.62, Messidor-2 0.56), and it defers more **referral** errors than confidence on both external sets. Figure 1 candidates: 24 / 0 / 4 |
| G2 | Does the quality head flag truly ungradable images? Held-out DDR grade-5 | **NOT RUN** (D10) | M0 was never built |
| G3 | Failure taxonomy: where does the reasoner contradict a *correct* grade? | **DONE 2026-09-29, exploratory** | Registered: M3 contradicts 47–65% of M1's correct calls; amended 10–36%. **56% of M1's APTOS errors are ŷ ≥ 3 calls that disagreement cannot see (D1)**, against 7–8% elsewhere |

> G1 produces **figure 1**: a case the grader calls confidently normal while the
> evidence path finds microaneurysms. Go looking for it deliberately.

## Stage H — Protocol-matched benchmark

| ID | Question | Status | Result |
|---|---|---|---|
| H1 | QWK on the **official** EyePACS test split, reported beside the 0.8496 leaderboard reference | **DROPPED** (D8) | The variants regrouped every EyePACS image; no split here is comparable with the leaderboard |

---

## Phase 6a — final training, split across sessions

Six runs: the frozen recipe on both declared variants at three seeds. Executed by
`notebooks/06_final_training.ipynb`.

**DONE 2026-09-22 — all six complete, none collapsed.** Metrics are on the
**validation** split, the one early stopping read, so they are optimistic by
construction; they are recorded to check the runs, not to report.

| run | Status | epochs (best) | val QWK | macro-F1 | g1-F1 | distinct |
|---|---|---|---|---|---|---|
| H1_eyepacs_full_s42 | DONE | 10 (9) | 0.7819 | 0.5657 | 0.2350 | 5 |
| H1_eyepacs_full_s43 | DONE, early-stopped | 7 (3) | 0.7563 | 0.5373 | 0.2073 | 5 |
| H1_eyepacs_full_s44 | DONE | 10 (8) | 0.7839 | 0.5646 | 0.2139 | 5 |
| H1_eyepacs_ddr_full_s42 | DONE | 10 (7) | 0.7666 | 0.5583 | 0.2274 | 5 |
| H1_eyepacs_ddr_full_s43 | DONE, early-stopped | 5 (1) | 0.7455 | 0.5067 | 0.1668 | 5 |
| H1_eyepacs_ddr_full_s44 | DONE | 10 (8) | 0.7816 | 0.5546 | 0.2377 | 5 |

| variant | val QWK mean | spread | macro-F1 mean | spread | g1-F1 mean | spread |
|---|---|---|---|---|---|---|
| eyepacs_full | 0.7740 | 0.0276 | 0.5559 | 0.0284 | 0.2187 | 0.0277 |
| eyepacs_ddr_full | 0.7646 | 0.0361 | 0.5399 | 0.0516 | 0.2106 | 0.0709 |

Paired by seed, `ddr − full` val QWK is −0.0153 / −0.0108 / −0.0023 (mean −0.0095):
same sign three times, every one inside both the seed spread and B1's 0.0276 floor.
**No in-domain difference is claimable, and none is claimed.** This is descriptive
only — nothing is selected on it — and H3 is about *external* transfer, which this
split cannot speak to. Adding a second domain costing a little in-domain is the
ordinary expectation, not evidence against H3.

### Two seeds early-stopped, and neither is retrained

`s43` stopped early in both variants; `eyepacs_ddr_full_s43` peaked at **epoch 1**,
inside the two-epoch warmup, and stopped at epoch 4. The resume path was checked
first, because a resume bug that lost `best_epoch` would make early stopping fire at
the wrong time and would be a code defect, not an outcome: `train_grading.py`
restores `best_qwk`, `best_epoch`, `history`, optimiser, scheduler and scaler
together. So this is the frozen rule — patience 3 on val QWK — doing what it says.

**It is not bad luck — it is a mechanism.** The four runs that trained to the end found
their best at epochs **9, 8, 7 and 8**: all in the low-learning-rate tail of the
10-epoch cosine schedule. The two early-stopped runs were halted at epochs 6 and 4,
before that tail arrived. Patience 3 on a noisy validation QWK can end a run during
its plateau, which on this schedule means before the phase where every completed run
peaked. Seed 43 is the worst run in *both* variants for that reason. The four
full-length runs span 0.7666–0.7839 (spread 0.0173); the early stops account for most
of the rest.

Whether Stage B ever saw early stopping fire is not known: the register did not record
its stopping epochs. So the interaction may simply have been invisible at the freeze.
From now on the stopping epoch is recorded with every run.

Retraining a seed because its result is worse is choosing seeds by outcome. The run
stands. The mechanism is reported as a finding about the frozen recipe, not repaired
after the fact. Its cost is honest and specific: it widens the DDR variant's seed spread, and
§7 of the pre-registration forbids claiming any difference smaller than that spread,
so H3 now needs a larger effect to be claimed.

### Why these score ~0.78 when Stage B scored 0.679

Not a red flag, and not leakage. Stage B selected the recipe on
`eyepacs_balanced_1000` — a few thousand balanced training images — at batch 24.
Phase 6a trains on all 59,842 `eyepacs_full` rows at the frozen batch 32. Both are
scored on the same validation split (held-out rows are identical across variants, A0).
~12x the data is the difference.

Batch 32 is `train_grading.py`'s default, the same class of leak as D7's 12 epochs.
It is not a deviation: `PREREGISTRATION.md` §2 lists batch size as *default*, not as
selected, and Phase 6a ran what was registered. It belongs beside the data-scale
difference as a condition selection did not test.

Phase 6a's grade-1 F1 (0.21–0.24) **does not settle B2's open prediction**. ResNet50's
0.180 and every Stage B figure were measured on `eyepacs_balanced_1000`; these are at
12x the data. Different scale, different question.

C4's comparison should now read against the final M1: the evidence path's 0.375
against ~0.78, not 0.679. The premise test still passes — neither redundant nor at
chance — with the evidence path relatively weaker than C4 suggested.

### The budget was priced wrong, and by 10%

The 21.7 GPU-h figure prices **both** variants at `eyepacs_full`'s 59 842 training
rows. `eyepacs_ddr_full` is that plus all 12 522 DDR images, which
`build_variants.py` assigns to train wholesale — about 72 364 rows. At 46 img/s and
10 epochs:

| variant | train rows | h/run | × 3 seeds |
|---|---|---|---|
| eyepacs_full | ~59 842 | ~3.6 | ~10.8 |
| eyepacs_ddr_full | ~72 364 | ~4.4 | ~13.1 |
| | | **total** | **~24 h** |

So Phase 6a is **~24 GPU-h, not 21.7** — beyond a single week's 30 h quota once
anything else is run in it. The notebook reads each variant's real row count rather
than reusing one figure for both, so the estimate can no longer be wrong in the same
direction twice.

### Why splitting across sessions costs nothing

The six runs are independent and individually seeded, so *when* each is trained does
not enter any result. What does not survive a session is `/kaggle/working`: the
notebook's section 6 copies completed runs back in from the attached
`verify-dr-phase6` dataset before resuming. **Skipping that step silently retrains
finished runs** — the failure is invisible, because a retrained run produces a
perfectly well-formed `metrics.json`.

That is the same shape as every Stage C failure in this register: a stage that
cannot do its job still emits a valid artefact. The countermeasure is the same —
gate on the quantity the next step consumes. Here the notebook prints each run's
state (`done` / `partial` / `not started`) before doing anything.

### Section 1 shipped as a comment

`06_final_training.ipynb` was assembled by copying cells out of `04_phase4.ipynb` **by
index**. The clone step is cell 2 there; cell 1 is its markdown heading. The generator
took cell 1 and wrote it into a *code* cell, so section 1 of Phase 6 was the single
line `## 1 - Clone the repo` — a Python comment. It ran, it succeeded, it cloned
nothing.

Nothing failed for six sections. Sections 3–7 need only the helpers, the cache mounts
and pandas; the first use of `REPO_DIR` is the training call in section 8, so that is
where it surfaced — after the carry-forward copy and the plan, on paid GPU.

The generator *did* validate every cell: it checked each one parsed. A comment parses.
**Parsing is not doing anything** — the same mistake as gating C2 on "images cached"
rather than "images carrying coordinates", and the same fix: check the thing the next
step consumes.

`tests/test_notebook_integrity.py` now checks every notebook for two properties:

1. no code cell is inert (parses to an empty body while having text), and
2. no cell uses a name that no earlier cell — or its own body — defines.

Both fire on the shipped bug, and the test asserts that they do: it reconstructs the
broken cell and requires the check to reject it. A guard that only ever passes proves
nothing. The cells are now located by **what they define** (`REPO_DIR = Path(`), not by
index.

### Two limits, not one — and the session wall is the tighter

The first version of the guard checked the **weekly quota** only. That is the wrong
limit once the quota is healthy: a Kaggle session is killed at a fixed wall-clock age
whatever the quota says, and **a session killed mid-run may never save its output** —
losing the session's work, not just the current epoch.

With a fresh 30 h quota the session wall is what binds. `eyepacs_ddr_full` is ~4.4 h
per seed, so its three runs are ~13.1 h: they cannot be done in one sitting however
much quota is available. The budget is now `min(quota, session wall)`, and the notebook
prints which of the two binds.

| | h/run | 3 seeds | sessions at 10.5 h usable |
|---|---|---|---|
| eyepacs_full (~59 842 rows) | ~3.6 | ~10.8 | 1 |
| eyepacs_ddr_full (~72 364 rows) | ~4.4 | ~13.1 | 2 |

### "Save & Run All" would have retrained everything

Section 10 said to finish each session with **Save Version → Save & Run All (Commit)**.
That re-executes the notebook top to bottom in a fresh container — it would have spent
the entire budget a second time to save the results of the first. The correct action
after an interactive run is **Quick Save**, which keeps the run that just happened.

Nothing about the notebook's logic was wrong here; the instruction beside it was. It is
the same failure mode as the rest of this register in a different register: the
artefact was well-formed and the step it described did the opposite of what was needed.

### The guard counts wall-clock, not training time

Kaggle bills the whole session while the GPU is attached, so cache extraction and
the carry-forward copy draw on the same quota as training. Counting only the seconds
inside a training call would under-report the spend and start a run that cannot
finish. Elapsed session time also self-corrects: if 46 img/s proves optimistic, the
overrun appears after the first run instead of after the third.

`tests/test_phase6_notebook.py` executes the notebook's own cell text against a
fixture with an injected clock — not a re-implementation of the arithmetic. The C2
channel-vocabulary bug survived because the test restated the code's wrong
assumption and therefore agreed with it.

---

## Before the unblinding — the leaderboard benchmark cannot run as registered

**Found 2026-09-22, after Phase 6a training and before any locked data was read.**
Recorded now because the same finding written after the unblinding would read as an
excuse.

The protocol (Rule 6, `02_research_protocol.md`) and the pre-registration (§4, §6) put
the leaderboard benchmark on the **official** EyePACS test split, 53,576 images,
"directly comparable to the published leaderboard (0.8496)". The six Phase 6a models
cannot be scored on it:

- `02_manifests.ipynb` built the variants with `--eyepacs-split regroup`: a fresh
  patient-grouped split over all 88,009 EyePACS images, official train and test mixed.
- `build_variants.py` knew, and recorded `leaderboard_comparable: False` in
  `dataset_plan.json`.
- `frozen_config.yaml` nonetheless reads `split: official_eyepacs  # 35,126 train /
  53,576 test`. It describes a split the six models were **not** trained on. The
  train-row count gives it away: 59,842 rows cannot come from a 35,126-image train set.
- The regroup ignored the competition's partition, so any independently chosen subset
  shares the whole cache's proportions: **about 80% of the official test set is in the
  six models' train (68%), val (8%) or calibration (4%) splits.** An earlier note here
  said "roughly two-thirds"; that counted train alone.

**Scope.** This removes one *secondary* outcome. The primary outcome (F2, APTOS and
Messidor-2), H2 (external calibration), H3 (DDR transfer) and in-domain triage on our
own patient-grouped test split are all untouched: none depends on the official split.

**RESOLVED — D8, option A (2026-09-22): the leaderboard comparison is dropped.** In-domain
results use our own patient-grouped EyePACS test split (17,615 images), labelled
custom-split, never beside 0.8496.

Option C — train one model on the official split and score it on the official test —
turned out to be impossible, and the check that was meant to establish that said the
opposite. The mirror's `source_split` is its **own** 70/15/15 re-split:

| mirror's split | train | val | test |
|---|---|---|---|
| mirror | 61,606 | 13,201 | 13,202 |
| competition | 35,126 | — | 53,576 |

`06_check_runs.ipynb` printed that mismatch — "mirror marks 13,202 images as test (the
official test set is 53,576)" — and on the next line announced "OPTION C is possible",
because it had tested only that a `source_split` column *existed*, never that it was
the competition's partition. It now compares the counts first, and is verified against
the real crosstab cell for cell: the old version repeats the false claim, the new one
refuses it and still offers C for a mirror that does ship the official partition.

Same failure shape as C1's empty geometry column: a check that cannot answer its
question still printed an answer.

Recovering the partition from the competition's own `trainLabels.csv` was weighed and
declined in D8: it would still not be protocol-matched (693 images missing
non-randomly by grade, A0) and the winners' eye-pair ensembles are exactly what B5
rejected.

### The analysis plan — FINAL 2026-09-23

`preregistration/ANALYSIS_PLAN.md` pins every implementation choice the
pre-registration left open. Drafted 2026-09-22, marked final by the author 2026-09-23
with no choice changed, and binding from then: D9 (the plan), D10 (REACQUIRE removed)
and D11 (the sampling-prior correction) are recorded in `PREREGISTRATION.md`. No locked
label had been read.

### Step 1 of the plan: the internal pass — DONE 2026-09-24

`scripts/predict.py` is the single pass the plan's §9 describes, and the same script
will run the locked sets in Phase 6b. `notebooks/07_internal_pass.ipynb` runs it over
internal data only: 5,000-image OOD references per variant, then calibration and val.

- **Label-free by construction.** Label columns go on reading; a static test finds
  every place a label is named or a manifest is read, and was checked against a
  mutated copy of the script that reads `grade` — both checks fired.
- **Locked data refused** without `--locked`, before any image is opened; tested with
  an image loader that fails if called.
- **Faithfulness exactly as §5.4 pins it**, in `src/verify_dr/triage/faithfulness.py`:
  the inpainted region is the components M3 counted (`lesion_region_mask`, beside the
  counting rule in `facts.py`), dilated 3 px; 19 seeded controls move each connected
  region of that dilated union intact, so area is equal even where channels overlap. A
  simulated lesion-blind network passes 1 time in 20, as the plan claims.
- **Resumable** by shard, and a stopped run resumes to byte-identical output.
- **It checks itself against training.** The notebook recomputes each model's val QWK
  from the pass's own grades and compares it with `metrics.json`; a mismatch means the
  pass is not scoring the model it thinks it is.

Rehearsed end to end by executing the notebook's own cells against a fake `/kaggle`
tree: dataset mounts, a cache needing repathing, six checkpoints, a decoy empty
`results/`, and the verification cell tested both ways.

**Result: the pass reproduces training exactly.** Section 9 recomputed each model's val
QWK from the pass's own grades:

| model | n | QWK (pass) | QWK (training) | difference | distinct |
|---|---|---|---|---|---|
| H1_eyepacs_full_s42 | 7040 | 0.7823 | 0.7819 | +0.0004 | 5 |
| H1_eyepacs_full_s43 | 7040 | 0.7563 | 0.7563 | +0.0001 | 5 |
| H1_eyepacs_full_s44 | 7040 | 0.7840 | 0.7839 | +0.0001 | 5 |
| H1_eyepacs_ddr_full_s42 | 7040 | 0.7665 | 0.7666 | −0.0001 | 5 |
| H1_eyepacs_ddr_full_s43 | 7040 | 0.7455 | 0.7455 | 0.0000 | 5 |
| H1_eyepacs_ddr_full_s44 | 7040 | 0.7820 | 0.7816 | +0.0004 | 5 |

Every model within 0.0004, against a 0.005 tolerance: same weights, same transform,
same decision rule. The residue is float16 autocast on a different batch composition.

Section 10, what the evidence pathway found:

| | calibration (3,512) | val (7,040) |
|---|---|---|
| evidence grade 0 / 1 / 2 | 651 / 243 / 2,618 | 1,326 / 472 / 5,242 |
| rules R1 / R2 / R3 / R3\* | 651 / 243 / 1,532 / 1,086 | 1,326 / 472 / 3,134 / 2,108 |
| with lesions | 2,861 (81%) | 5,714 (81%) |
| faithfulness determined / undetermined / none | 2,849 / 12 / 651 | 5,686 / 28 / 1,326 |
| controls placed | median 19 of 19, min 0 | median 19 of 19, min 0 |

The machinery works: the randomisation test decided 99.6% of the images it was asked
about, and the median image got all 19 controls.

### M3 calls three in four EyePACS images diseased

**Evidence grade 2 goes to 74.5% of the calibration split. About 19% of EyePACS is
truly grade ≥ 2** (19.23% after A0's 693-image loss). No label is needed to see the
gap, only the known prevalence: with ~73.5% of images grade 0 and 81.5% carrying
evidence, **at least 74% of grade-0 images get evidence of disease**, and at least
two thirds of non-referable images get evidence grade 2.

**It is not a bug in the pass.** C4 on DDR gave the same split — R1 22.1%, R2 5.4%,
R3 + R3\* 72.5% — on a dataset that is 50% grade 0, against 18.5% / 6.9% / 74.5% on
EyePACS at ~73% grade 0. M3's output barely moves with the true prevalence. C4's QWK
of 0.375 was read as "informative but weaker", and R3\* (35%) was attributed to the
MA channel missing lesions. The other reading was never tested: **the haemorrhage and
exudate channels fire on healthy retinas.** The likely cause is M2's training set: the
757 DDR segmentation images were annotated *for* their lesions, so M2 has probably
never seen a healthy fundus, and a 4-pixel component counts as a finding.

**Why it matters: disagreement is only as specific as M3.** Where M1 correctly calls
grade 0 and M3 reports grade 2, d_evidence = 2 — the strongest disagreement there is —
and the disagreement arm defers exactly the cases M1 gets most reliably right. H1 and
H1′ are at risk for a reason unrelated to M1.

**Nothing is changed.** The frozen M2 operating point (threshold 0.5, ≥ 4 px) stands
until the rehearsal on val measures the consequence with val's labels (step 2, below).
Any change after that is a dated deviation made before the locked data are read, and
must be chosen by a criterion other than the H1 outcome.

### Step 2 of the plan: fit and rehearsal — code built, ready to run

`scripts/fit_params.py` fits T (and T for stage 1 alone), the OOD statistics, τ_ood,
τ_conf and r on the calibration split; `scripts/analyse.py` computes every table in
§§4–8 for one dataset and the §8 verdicts across datasets.
`notebooks/08_fit_and_rehearse.ipynb` runs both on CPU and rehearses on val.

- **numpy only.** Ledoit–Wolf is written out and matches scikit-learn to 1e-9; the
  temperature search is a log grid refined by golden section. The same numbers come out
  on Kaggle and in the tests.
- **Ties by expectation, exactly.** The coverage–accuracy AUC replaces each case's
  correctness by its tie block's mean; tests check it against brute-force enumeration of
  every ordering, and that the "none" arm scores exactly the overall accuracy.
- **The unblinding cannot come early.** `analyse.py` refuses locked data without
  `--unblind`, before any label is read, and `--unblind` refuses a `fitted_params.json`
  that git does not track unmodified. The file carries a digest over its canonical form,
  so it can be pasted through a chat and re-indented, but not edited.
- **Every guard was mutated and caught:** ties broken by position, `>` loosened to `≥`
  in the faithfulness rule, the prior correction removed from the temperature fit (T
  then doubles, 1.8 → 3.65 — D11 in miniature).

Details the plan left open, pinned in code now, before any locked label:

| | Pinned as | Why |
|---|---|---|
| H2's bootstrap | EM re-run inside every resample | its uncertainty belongs in the interval; holding it fixed would favour H2 (rule 3) |
| a coverage point | k = ⌈c·n⌉, the smallest k with coverage ≥ c | 80% of 17,615 = 14,092 exactly |
| F4's binary form | [d_evidence > 0] + r · d_faith, same r | F4 varies the evidence term only |
| E2 | images with no successful control excluded, counted | a mean of zero controls is undefined |
| D4's draws | `default_rng([size, draw])` | reproducible, independent of order |
| r's tie | AUCs within 1e-12 | identical rankings give bit-identical AUCs |

**One new diagnostic: EM's premise.** EM assumes the mean calibrated posterior equals
the prior it was calibrated under. The synthetic fixture shows what happens when it
does not: EM run on unshifted data drove grades 1–3 to zero. So the fit records EM run
on the calibration split itself, where nothing has shifted; if it drifts there, it will
drift on the externals and H2 fails for that reason, not for the shift.

**At real scale** (3,512 / 7,040 / 5,000 × 512-d, 2,000 resamples) the fit takes seconds
and the rehearsal about four minutes, CPU only.

### Step 2: the rehearsal — DONE 2026-09-24. The frozen analysis fails in-domain, for two diagnosed reasons

**The fit (calibration split, per model).** Digest `8fc2b5f8…`; the parameters are kept
as a record in `docs/rehearsal/2026-09-24_fitted_params_frozen.json`. They are not
`preregistration/fitted_params.json`, because nothing is committed for the unblinding yet.

| | range over the six models |
|---|---|
| T (stages 0+1) | 0.41–0.51. Below 1: the prior-corrected posteriors are under-confident |
| ECE raw → stage 1 alone → stages 0+1 | 0.23–0.27 → 0.16–0.25 → **0.03–0.06**. D11's correction does most of the work |
| τ_ood | 1.85–2.08 (5.0% of calibration flagged, by construction) |
| τ_conf | 0.90–0.92 |
| r | **2.5 for all six**. Calibration AUC rises monotonically with r |

π_src comes from 59,842 EyePACS training rows: 0.737 / 0.070 / 0.148 / 0.023 / 0.021.

**The rehearsal on val (7,040 images).** Coverage–accuracy AUC per arm:

| model | none | confidence | OOD | disagreement | combined |
|---|---|---|---|---|---|
| eyepacs_full s42 | 0.7719 | **0.9195** | 0.8606 | 0.7677 | 0.8322 |
| eyepacs_full s43 | 0.7169 | **0.9015** | 0.8483 | 0.7189 | 0.7875 |
| eyepacs_full s44 | 0.7541 | **0.9167** | 0.8463 | 0.7536 | 0.8221 |
| eyepacs_ddr_full s42 | 0.7153 | **0.9106** | 0.8130 | 0.7163 | 0.7936 |
| eyepacs_ddr_full s43 | 0.7205 | **0.8978** | 0.8482 | 0.7167 | 0.7887 |
| eyepacs_ddr_full s44 | 0.7659 | **0.9189** | 0.8608 | 0.7623 | 0.8257 |

**Disagreement is indistinguishable from no gate at all** (within ±0.005 of "none" for
every model), and confidence beats it by 0.15–0.19. The rehearsal verdicts are "not
supported" for everything. For H2 and H3 that is expected on val, since there is no
shift and no external set. **For H1 it is decisive.**

**Why, from the d_evidence blocks** (eyepacs_full s42; the other five match):

| d_evidence | share of images | M1 correct |
|---|---|---|
| 0 | 35% | 81% |
| 1 | 19% | **35%** |
| 2 | 47% | **91%** |

The signal exists: a one-grade disagreement picks out M1's errors. **But the ordering is
inverted.** The largest disagreement is mostly M3 reporting grade 2 on healthy eyes that
M1 correctly calls 0, so the arm defers M1's safest calls first.

**M3 against the truth on val**, descriptive:

| true grade | evidence 0 | evidence 1 | evidence 2 |
|---|---|---|---|
| 0 | 1,243 | 363 | 3,560 |
| 1 | 53 | 84 | 364 |
| 2 | 27 | 24 | 1,013 |
| 3 | 2 | 0 | 153 |
| 4 | 1 | 1 | 152 |

Evidence is found in **75.9% of grade-0 images** and missing in only 2.2% of referable
ones: very sensitive, almost unspecific. M3's QWK on val is **0.138**, against 0.375 on
DDR in C4. EyePACS's larger healthy share exposes the false positives that DDR's mix hid.

**EM's premise fails.** The mean stage 0+1 posterior for grade 0 is 0.79–0.82 against a
true 0.744. EM run on the calibration split, where nothing has shifted, drives grade 1 to
~0 and grade 0 to 0.85–0.90 in every model. Temperature and a fixed prior correction match
the probability of the predicted grade (ECE 0.03–0.06) but not the grade mix, and EM needs
the grade mix. H2 would fail for that reason, not because of any shift.

Faithfulness: 59–72% of determined lesion images are faithful. 28 undetermined, 20 of them
with no control placeable.

### The decision: amend, with the registered analysis primary — D12, D13 · 2026-09-24

Three options were put to the author: proceed as frozen; amend with the frozen analysis
primary; amend with the amended analysis primary. **Chosen: amend, with the frozen analysis
primary.** Both deviations are recorded in `PREREGISTRATION.md` and specified in the
plan's addendum (A.1–A.3), written *before* the code that implements them and before any
locked image or label was read.

| | Changes | Criterion | Does not touch |
|---|---|---|---|
| **D12** | A lesion type is present for M3 only if its total predicted area reaches a per-type minimum from {4, 16, 64, 256, 1024} px (all-4 is the frozen rule) | M3's own QWK against the true grade on calibration; M1 never consulted | M2's detections, the faithfulness regions |
| **D13** | Bias-corrected temperature scaling (one T plus a bias per grade) for D1–D4 and H2; EM's reference prior becomes the calibration grade mix | NLL on calibration; at the optimum the mean posterior *is* the grade mix | d_conf, τ_conf, the confidence arm (H1's baseline) |

**One round.** The amended analysis is rehearsed once. Then `fitted_params.json` is
committed and the locked pass runs, whatever that rehearsal shows. Both analyses come
from one locked pass and one label join; the registered one decides the registered
verdicts.

**Built and checked:**

- **The registered analysis is unchanged, to the last digit.** The pre-amendment code
  (`aef5d12`) and the amended code, run on the same fixture, give identical fitted values,
  per-model tables, claims and evidence table for the registered analysis.
- **D13's premise holds by construction:** on synthetic data where a fixed prior
  correction makes EM drift by more than 0.05 with no shift at all, BCTS drifts by less than
  1e-6. The mean fitted posterior matches the grade mix to 1e-9. A mutant with the biases
  forced to zero fails the test.
- **D12 is M3 plus one condition, nothing else.** The vectorised ladder agrees with
  `rules.grade` on all 16 presence patterns, and the frozen setting reproduces the real
  pipeline's evidence grade exactly (`extract_facts` → `grade`). `fit_params.py` refuses
  to run if the stored counts do not reproduce M3's stored grade. On the fixture, whose
  healthy eyes carry planted 4–40 px exudate, the fit chose a 64 px exudate minimum and
  M3's QWK rose from 0.32 to 0.81.
- **A bug found by running it:** at the search's extreme temperatures the bias Hessian is
  exactly singular and the undamped solve raised. The damped fallback now takes over, and
  a test pins the case.

### The amended fit on the real calibration split — 2026-09-25

Section 4 of the re-run `08_fit_and_rehearse.ipynb`. The registered values reproduced
exactly: the same T, ECE and EM drift as the first run.

| | registered | amended |
|---|---|---|
| ECE on calibration | 0.03–0.06 (stages 0+1) | **0.012–0.018** (BCTS, T 0.42–0.48) |
| EM drift where nothing shifted | 0.107–0.151 | **~1e-11**: EM returns the true grade mix exactly |
| M3's QWK on calibration | 0.133 | **0.441** |
| evidence grades 0 / 1 / 2 on calibration | 651 / 243 / 2,618 | 2,113 / 930 / 469 |
| r | 2.5 for all six | 2.5 for all six |

**D12's area minimums:** microaneurysm 16 px, haemorrhage 256 px, hard exudate 256 px,
soft exudate **1,024 px**. That is the top of the pre-specified grid, so soft exudate now
counts only when it is extensive. The grid is not widened: that would be a second round.
**The new weakness:** evidence grade 1 now goes to 26% of calibration images against ~7%
truly grade 1. The ladder's MA-only rung catches images whose larger lesions fell below
their minimums.

### Steps 4 and 5 built, and the whole chain rehearsed on toy data

`notebooks/09_locked_pass.ipynb` (GPU) and `notebooks/10_unblinding.ipynb` (CPU). Both open
with `step3_record`, which refuses unless `preregistration/fitted_params.json` exists,
verifies against its digest, is tracked by git and unmodified, and has its digest
recorded in `PREREGISTRATION.md`. 09 also checks all seven checkpoints' SHA-256 against
the fitted ones before any GPU time is spent. It runs the externals first.

**Rehearsed end to end, 07 → 08 → step 3 → 09 → 10**, on toy data in a fake `/kaggle` tree
with step 3 committed in a scratch repository. The rehearsal checks the plumbing, not the
numbers: the three locked passes are written label-free, and the unblinding produces both
analyses for all three sets plus the verdicts. It surfaced one toy-only trap. A randomly
initialised EfficientNet-B0's features vanish (~1e-14), every toy embedding is identical,
and the OOD covariance is singular, so the toy checkpoints use ResNet50. The real fit
(5,000 real embeddings) was never affected.

### Step 3 of the plan: the fitted parameters are committed — DONE 2026-09-27

`preregistration/fitted_params.json` is committed as **`50d5d32`**, digest
**`0df6b029b3600c1f014b38fa9c5310fa84ca3c2355dac3b55a807452b4d1c4a9`**, and both are
recorded in `PREREGISTRATION.md` ("Step 3"). No locked image or label has been read.

- **It is exactly what `fit_params.py` wrote.** It came through a chat paste, and it is
  byte-identical to `write_params`' output for its own values, and it verifies against
  its digest.
- **The amendment changed no registered value.** Its registered block matches the first
  fit (`8fc2b5f8…`, made before D12 and D13 existed) value for value, on identical input
  digests. The OOD statistics' digests are identical too, so either version of
  `verify-dr-fitted` serves 10.
- **The code is the code that fitted it.** Since `b2eca70` the only change under `src/`
  and `scripts/` is `step3_record`, 45 lines added and none removed.
- **Sections 5–7 of the amended re-run are not yet in the record.** Under the one-round
  rule they decide nothing: step 3 happens whatever they show.

Per-model values are tabulated in `PREREGISTRATION.md`. r and r′ are 2.5 (the grid's top)
in all six models, as in the first fit.

**Next: 09, the locked pass (GPU, ~2–4 h), then 10, the unblinding (CPU).** Nothing is
committed between them, so 09 and 10 run on the same commit. 09's label-free outputs
enter the record together with 10's. A bug found on the way would be fixed as a dated
deviation (§9 of the pre-registration), never silently.

### Step 4 of the plan: the locked pass — DONE 2026-09-28, label-free

`09_locked_pass.ipynb` on commit `1d9a764`, published as `verify-dr-locked`. **87
minutes** of pass time on two T4s, externals first: Messidor-2 7.4, APTOS 17.1, EyePACS test 62.8.
Every `run.json` reads `locked: true, labels_written: false`. 10 re-checks each pass's
recorded checkpoints and constants against the fitted ones before it joins a label.

| | Messidor-2 | APTOS | EyePACS test (custom, D8) |
|---|---|---|---|
| images | 1,744 | 3,662 | 17,615 |
| M3 finds a lesion | 90.3% | 81.5% | 80.4% |
| evidence grade 0 / 1 / 2 (frozen rule) | 9.7 / 13.3 / 76.9% | 18.5 / 3.9 / 77.6% | 19.6 / 6.9 / 73.5% |
| R3* (haemorrhage or exudate, no MA) | 17.0% | 21.0% | 29.7% |
| faithfulness decided, of lesion images | 99.2% | 97.2% | 99.6% |

**Every count reconciles.** The rules sum to the images. R3 + R3* equals evidence grade 2,
and R2 equals grade 1. R1 equals grade 0, which equals the images with no faithfulness
test. Decided plus undetermined equals the images with lesions, and the per-shard lesion
counts sum to the totals. EyePACS test matches calibration's evidence split (18.5 / 6.9 /
74.5%) to within 1.1 points.

Three observations, recorded before any label is joined:

- **The in-domain rows arrive in grade-correlated order.** The per-shard lesion rate runs
  near 75%, but 89–100% in shards 5, 24–29 and 34–35. `prepare_manifest.py` sorts by
  image path, the cache keeps the mirror's folder layout, and the mirror's path order
  follows grade. This is harmless, and the code was checked before the join: every
  statistic is order-invariant (ties by expectation in `selective.py`, uniform bootstrap
  resamples, labels joined by ID with duplicates and gaps refused), and nothing parses a
  path for a grade.
- **Messidor-2 stands out.** M3 finds a lesion in 90% of its images, against 80–82%
  elsewhere, and evidence grade 0 in 9.7% against 18.5–19.6%. The registered analysis
  uses these grades as they are. The amended one re-derives them with D12's minimums from
  the stored per-type counts and areas.
- **Messidor-2's preprocessing was never recorded.** Dataset card Q2 (is
  `messidor2preprocess` already contrast-normalised or Ben-Graham processed?) is open,
  and A0's visual audit left Messidor-2 out. The mirror's page could not be read from the
  analysis environment. The audit follows the unblinding, as planned. Whatever it finds
  goes into the limitations, and the verdicts stand as registered.

**This record is committed before 10 runs, not with its results as step 3 said.** The
analysis environment is ephemeral, and an uncommitted record can be lost with it. The
commit changes only this register, `CLAUDE.md` and the report, so 10 runs on the code,
fitted parameters and pre-registration 09 ran on: `git diff 1d9a764` touches nothing
under `src/`, `scripts/`, `notebooks/` or `preregistration/`. `results.json` names the
commit 10 cloned; any commit between `1d9a764` and it must pass that same check.

### Step 5 of the plan: the unblinding — DONE 2026-09-28. Every hypothesis: NOT SUPPORTED

`10_unblinding.ipynb`, run once, published as `verify-dr-results`. The labels of all
three sets were joined in one step. Sections 5 and 6 are kept verbatim in
`docs/unblinding/2026-09-28_readout.txt`. The true grade counts match A0's reconciliation
exactly: Messidor-2 1,017 / 270 / 347 / 75 / 35, APTOS 1,805 / 370 / 999 / 193 / 295, and
EyePACS test 13,002 / 1,226 / 2,593 / 414 / 380 at natural prevalence.

**The readout checks out: 380 cross-checks and none fails.** Every effect is the
difference of the arms it compares and lies inside its interval. Every confusion matrix
reproduces the true grade counts and 09's label-free evidence counts. M3's QWK,
recomputed independently from each confusion matrix, matches the printed value. The arms
the amendment does not touch (none, confidence, OOD) are identical in both analyses.

| Hypothesis | Registered analysis (primary) | Amended analysis (D12, D13) |
|---|---|---|
| H1 — disagreement beats confidence in-domain | not supported, both variants | not supported, both variants |
| H1′ — …and on both external sets (the thesis) | not supported | not supported |
| H2 — EM improves external calibration | not supported | not supported |
| H3 — DDR in training improves external QWK | not supported | not supported |

H1 and H1′ also fail §1's own falsification criterion, "confidence-gating matches or beats
it": confidence beats disagreement in every seed of both variants, on all three sets, in
both analyses, and every 95% interval lies below −0.09.

**Coverage–accuracy AUC, primary variant (`eyepacs_full`), mean of three seeds:**

| | none | confidence | OOD | disagreement, registered → amended | combined, registered → amended |
|---|---|---|---|---|---|
| EyePACS test (custom, D8) | 0.747 | **0.917** | 0.858 | 0.746 → 0.801 | 0.815 → 0.890 |
| APTOS | 0.615 | **0.860** | 0.816 | 0.549 → 0.672 | 0.775 → 0.860 |
| Messidor-2 | 0.669 | **0.841** | 0.768 | 0.659 → 0.716 | 0.727 → 0.805 |

The replication variant (`eyepacs_ddr_full`) orders the arms the same way; its
confidence AUCs are 0.912 / 0.889 / 0.803.

**H1/H1′ effect (disagreement − confidence), per seed, over both variants:**

| | registered | amended |
|---|---|---|
| EyePACS test | −0.156 to −0.191 | −0.104 to −0.134 |
| APTOS | −0.228 to −0.367 | −0.114 to −0.236 |
| Messidor-2 | −0.167 to −0.217 | −0.116 to −0.156 |

1. **Under the registered analysis, disagreement carries no signal.** In-domain it ranks
   within 0.011 of no gating in every model, and on Messidor-2 within 0.037. On APTOS it
   is *worse* than no gating in all six models (by 0.030–0.076). The val rehearsal
   predicted the in-domain figure to within 0.002: for s42, 0.7677 against 0.7719 on val
   and 0.7662 against 0.7696 on test.
2. **The amendment restores a real signal, but the gap to confidence remains.** D12's
   minimums were fitted on EyePACS calibration alone, and they transfer. M3's QWK rises
   from 0.141 to 0.444 in-domain, from 0.366 to **0.755** on APTOS and from 0.152 to 0.505
   on Messidor-2. The share of grade-0 images given evidence falls from 62–85% to 10–36%.
   Disagreement then beats no gating in every model on every set, by 0.037–0.084, but
   confidence stays 0.10–0.24 ahead.
3. **The combined policy comes closest.** It sets its level from the evidence,
   faithfulness and OOD signals, then orders by confidence within each level. In the
   amended analysis it trails confidence by 0.02–0.05 in-domain and on Messidor-2, and
   ties it on APTOS (mean 0.860 against 0.860). It is ahead in 4 of 6 models there, by at
   most 0.016. This is descriptive only: no registered claim concerns the combined arm.
4. **H2: EM makes external calibration worse.** It does so in all 12 model × set cells
   of the registered analysis, by 0.04–0.25 ECE. In the amended analysis, whose EM
   premise held on calibration (D13), it does so in 11 of 12; the twelfth moves by
   +0.0003. The oracle prior helps on Messidor-2, taking ECE from 0.09–0.13 to 0.04–0.07,
   but hurts on APTOS, taking it from 0.04–0.11 to 0.07–0.16. So on APTOS even the true
   prior does not fix the posteriors, and the shift there is not the label shift EM
   assumes. On Messidor-2, it is EM's estimate of the prior that fails.
5. **H3: DDR in training does not improve transfer.** External QWK, DDR minus
   EyePACS-only, per seed: Messidor-2 −0.091, −0.092, −0.020 (worse in every seed), and
   APTOS −0.011, +0.047, +0.031.

**What it means.** The result is pre-registered, negative and robust. It holds in every
seed, both variants, all three sets and both analyses. The premise check of §8 passed:
the evidence pathway is informative (C4, and M3 reaches 0.755 QWK on APTOS after D12). But
disagreement with it is a weaker signal of M1's errors than M1's own calibrated
confidence, in-domain and under shift. The pre-registration requires this to be reported
as a negative result with analysis, never worked around.

**Candidate explanations: post hoc, for Phase 8 (G1, G3), labelled exploratory:**

- **Coarseness does not explain it by itself.** Disagreement takes at most six values
  against confidence's continuum, and the plan names that coarseness as a property of the
  method (§6.2). But the combined policy, which orders by confidence *within* its levels,
  still trails confidence alone except on APTOS. So the levels themselves rank M1's errors
  worse than confidence does.
- **M3 stops at grade 2 (D1),** so no disagreement can question M1's grade-3 or grade-4
  calls.
- **M3's own misses become disagreements M1 wins.** Under D12, 21% of referable in-domain
  images get no evidence at all, so where M1 correctly calls them referable, d_evidence is
  2.

**Still to transcribe:** F3–F5, D4 and E1–E3 are computed and stored in `results.json`
(`verify-dr-results`), but the readout does not print them.

**From here, §9 of the pre-registration applies.** These are the numbers retained. A bug
found later is fixed, and the fix and any re-run are recorded as dated deviations beside
these numbers, never in place of them.

### Phase 8 — the error analysis: declared 2026-09-28, before it first runs

**Exploratory, labelled post hoc throughout.** Nothing in Phase 8 can change a verdict:
those are fixed by the one label join and retained under §9. The analyses are declared
here and committed with `scripts/phase8.py` before `notebooks/11_phase8.ipynb` runs, so
the tables are chosen before their numbers are seen. G2 cannot run (no M0, D10).

| # | What | Data | Definition |
|---|---|---|---|
| P8.0 | **Reproduction gate** | locked passes, labels | Recompute every per-image signal as `analyse.py` did. Refuse unless all five arms' AUCs match `results.json` for every model and both analyses (to 1e-12). Refuse unless `results.json` is the unblinded one for the same pass and parameters. |
| P8.1 | **Secondary outcomes, transcribed** | `results.json` only | Accuracy at 80% and 90% coverage, F3, F4, F5, D4 and E1–E3, all computed at the unblinding. Nothing new is computed. |
| P8.2 | **G1 — error overlap** | as P8.0 | At 80% and 90% coverage, split M1's errors into those deferred by confidence only, by disagreement only, by both, or by neither. Expected counts, with ties broken independently at random, so the result does not depend on order. The same split for rDR errors (ŷ ≥ 2 ≠ y ≥ 2). Also the error rate within each arm's deferred set. |
| P8.3 | **G1 — does disagreement add to confidence?** | as P8.0 | Split each model's images into quintiles of d_conf. Within each quintile, report M1's accuracy at each d_evidence level, and the AUROC of disagreement for error, weighted across quintiles (0.5 means no information beyond confidence). Also each signal's error-detection AUROC, with ties counted as 0.5. |
| P8.4 | **G3 — failure taxonomy** | as P8.0 | M1's errors that disagreement cannot see: ŷ ≥ 3 (M3 abstains, D1), and e = ŷ (M3 agrees with a wrong grade). Also M1's correct calls that M3 contradicts: e > ŷ, by rule and lesion type, and e < ŷ. |
| P8.5 | **Figure 1 candidates** | EyePACS test, then the externals; model `eyepacs_full_s42`; amended evidence | Three categories. **A** (figure 1): M1 predicts 0 among its most confident quarter of grade-0 calls, the truth is ≥ 1, and M3 finds microaneurysms. **B**: the same confident 0, a true 0, and M3 evidence 2 (false evidence). **C**: M1 correct at grade 2 and M3 evidence 0 (M3 misses). Up to 8 per category, ordered by confidence and then image ID, and drawn with M2's lesion outlines. The figure may use any candidate, and the caption says it was selected. |
| P8.6 | **Messidor-2 preprocessing audit** | images only, label-free | 200 cached images per source (EyePACS test, APTOS, Messidor-2, DDR). Field-of-view fraction, background level, colour and saturation, near-grey share, sharpness and high-pass energy. Raw Messidor-2 files: size, mode and background. Lesion-detection rates per source from the locked passes. Contact sheets. |

**Built and tested 2026-09-28, not yet run.** The World fixture's 11 tests cover a real
`--unblind` run of `analyse.py` followed by Phase 8. There, the gate reproduces every
AUC, catches one changed by 1e-9, and refuses another pass's results and a locked pass
with no unblinding on record. Notebook 11 ran end to end on the toy Kaggle tree
(07 → 08 → step 3 → 09 → 10 → 11). One fix came from that run: a set where no image
qualifies for a figure category is now reported as a finding, not a failure.


### Phase 8 — results, 2026-09-29 (exploratory, labelled post hoc)

`11_phase8.ipynb`, published as `verify-dr-phase8`. Sections 3, 4 and 6 are kept
verbatim in `docs/phase8/2026-09-29_readout.txt`, with the audit's two images.

- **The readout checks out:** 76 cross-checks and none fails. G1's four-way splits and
  F5's shares each sum to one, and E3's per-grade counts sum exactly to the lesion images
  09 found before any label was read.
- **P8.0, the gate, passed on all three sets.** The recomputation reproduces every
  published AUC (6 models × 2 analyses × 5 arms per set) to 1e-12.
- **P8.1.** The registered secondary outcomes are transcribed into Stages D, E and F above.

**What Phase 8 adds. All of it is post hoc; none of it changes a verdict.**

1. **The registered disagreement points at correct cases.** Its error-detection AUROC is
   0.44 on EyePACS test, 0.38 on APTOS and 0.45 on Messidor-2, below chance; confidence
   scores 0.87, 0.92 and 0.79. At 80% coverage, only 10% of the in-domain cases it
   deferred were errors, against 25% overall. It deferred M1's safest calls.
2. **Amended, it adds information beyond confidence, and only under shift.** Within
   quintiles of confidence, its AUROC for error is 0.52 in-domain, 0.62 on APTOS and 0.56
   on Messidor-2.
3. **On referral errors, the amended disagreement beats confidence on both external
   sets.** At 80% coverage it defers 38.7% of APTOS's referral errors against
   confidence's 17.0%, and 31.1% of Messidor-2's against 26.5%. At 90%: 8.0% against 5.4%,
   and 10.7% against 8.7%. In-domain it is the other way round: 28.4% against 55.3%. F3 shows which errors these are.
   The arm raises the accepted set's specificity, not its sensitivity, so it catches
   false referrals, not missed ones.
4. **APTOS has a structural blind spot.** 56% of M1's APTOS errors are grade-3 or grade-4
   calls, which disagreement cannot see because M3 stops at grade 2 (D1). In-domain and
   on Messidor-2 the share is 7–8%.
5. **The registered failure, counted.** M3 contradicted 65% of M1's correct in-domain
   calls that it could judge. By rule, 48% of those were R3* and 42% were R3. After D12
   the rate falls to 31%, and 76% of the over-calls are microaneurysm-only (R2).
   Messidor-2 has adjudicated grades, and its over-call rate is similar (20.1% against
   22.5% in-domain). That argues against missed microaneurysms in EyePACS's single-grader
   labels as the main cause.
6. **Faithfulness holds (E1–E3).** Removing M2's lesion regions changes the grader more
   than removing all 19 equal-area random regions in 67–78% of lesion images, against
   about 5% by chance, and in 75–96% on grades 1–3. The grader does use the lesions the
   evidence pathway finds.
7. **The figure-1 case exists, but it is rare.** Category A (M1 confidently normal, truth ≥ 1,
   M3 finds microaneurysms) has 24 images in-domain (0.14%), none on APTOS and 4 on
   Messidor-2. The strongest candidate on paper is `EyePACS::219_left`: truth 2, M1 says
   0 with confidence 0.982, and M2 finds 30 microaneurysms covering 425 px.

   **The sheets, seen 2026-09-29** (`docs/phase8/2026-09-29_figure1_candidates_*.webp`).
   This is a visual reading at 512 px by a non-clinician, not a measurement. Where a
   detection can be judged, it is more often an artefact than a lesion:
   - `769_right`: the largest haemorrhage outlines lie outside the field of view.
   - `219_left`: its outlines sit on grey, rod-shaped specks that look like debris. Its
     soft exudate sits on a pale spot at the frame centre, and the same spot recurs there
     in two other patients' images, which points to a camera reflex.
   - Messidor-2: the haemorrhage outlines of `IM003603` and `20060523_49449_0100_PP` sit
     on vessels, and D12's 256 px minimum removes both.

   **M3 counts detections outside the retina.** The evidence path never applies the
   field-of-view mask. The aggregate share was not
   measured.

   **Figure 7.1, decided 2026-09-29** (the author delegated the choice). It has three rows:
   - (a) `Messidor2::20051021_39482_0100_PP`: adjudicated grade 1, and M3 gives grade 1
     by R2 in both analyses; two interior microaneurysms, area 17 px against D12's 16;
   - (b) `769_right`, outlines outside the retina;
   - (c) `219_left`, outlines on debris-like specks.

   Built from the sheets by `docs/thesis/figures/make_fig7_1.py`; chapter 7, §7.1.2.

   **Not run, by decision:**
   - measuring the share of M3's false evidence that lies outside the retina;
   - re-running P8.6's raw check with the mirror attached.

   Neither can change a verdict. Both are stated as limitations.

**P8.6: Messidor-2's preprocessing.** This answers dataset card Q2.
- **Not Ben-Graham processed and not grey-normalised.** The cached images have black
  padding (median 6.3), no near-grey pixels, and saturation like APTOS's (0.79 against
  0.80). Such processing would have survived this project's pipeline.
- **The least sharp source.** Median Laplacian variance is 106, against 161–204 for the
  other three.
- **M2 reports microaneurysms in 73.2% of Messidor-2 images** (median 23 px), against
  50.7% for EyePACS and 60.5% for APTOS. Whether soft, compressed images produce microaneurysm-like
  specks is a hypothesis, not a finding.
- **EyePACS is the least saturated source** (0.47, against 0.70–0.80). The training domain
  differs in colour from every other source, which is a fact to set beside H2.
- **A bug: the raw-file check measured the cache.** The cache keeps each source's folder
  names, so a `preprocess` folder exists inside `verify-dr-cache-512`, and it sorted
  first. The fix is a guard in `phase8.py audit`, which skips a raw root inside a cache
  root and has a test, plus discovery that excludes the cache. Rehearsed on the toy
  tree. The mirror's native resolution and compression are therefore still unobserved.
  Re-running section 6 with the mirror attached would observe them.

### Phase 7's code was written before the unblinding — the order this required

Before step 1, `src/verify_dr/calibration/` and `src/verify_dr/triage/` were empty.
Temperature scaling, prior-shift EM, OOD-z, disagreement gating and the
coverage–accuracy curves were all unwritten.

If the unblinding runs first, every one of those is implemented *after* the external
labels have been seen. The pre-registration froze the procedures (gating arms,
threshold fitted on internal calibration, coverage points) but not their
implementations — how disagreement is scored, what the OOD score is computed on — and
each unwritten detail is a fork that could be taken knowing the answer.

So the order becomes: **build and test the whole analysis on internal data, then read
the locked sets once**, with one pass producing a label-free prediction table and the
labels joined in a single final step.

---

## Summary

**26 experiments.** Selection happens only in B1–B5. External data is touched exactly
once, in Phase 6b, after the freeze. It was, on 2026-09-28, and **every registered
hypothesis (H1, H1′, H2, H3) was not supported**, in the registered analysis and in the
amended one reported beside it. See "Step 5 of the plan: the unblinding".

Phase 6a (the six training runs) touches no locked data and can therefore be split
across as many sessions as the quota needs. The unblinding is a separate notebook for
exactly that reason: re-running the training notebook must never be able to spend the
one shot.
